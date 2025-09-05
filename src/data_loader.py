import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch.nn.functional as F
import json
import random
import re
import os

# ---------------- 新增：ICD9 层级推断 & 构图 ----------------
def _infer_parent(code: str, code_set: set[str]) -> str | None:
    """
    基于 ICD-9 代码字符串前缀推断最近父级：
    - 逐步去掉末尾字符；若末尾是 '.' 则一并去掉；
    - 一旦得到的前缀在 code_set 中，返回该前缀作为最近父级；
    - 找不到则返回 None。
    """
    cand = code.strip()
    while len(cand) > 1:
        cand = cand[:-1]
        if cand.endswith('.'):
            cand = cand[:-1]
        if cand in code_set:
            return cand
    return None

def build_hierarchy_edges(code2idx: dict, direction: str = "parent_to_child") -> torch.LongTensor:
    """
    构建 1-hop 层级边（代码级节点）：
    - direction = 'parent_to_child' 或 'child_to_parent'
    - 仅返回 edge_index（权重默认为 1，在下游可选用）
    """
    codes = list(code2idx.keys())
    code_set = set(codes)
    rows, cols = [], []
    for c in codes:
        p = _infer_parent(c, code_set)
        if p is None:
            continue
        ci = code2idx[c]
        pi = code2idx[p]
        if direction == "parent_to_child":
            rows.append(pi); cols.append(ci)
        else:
            rows.append(ci); cols.append(pi)
    if len(rows) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([rows, cols], dtype=torch.long)

def build_hierarchy_adjs(code2idx: dict, device: str = "cpu") -> dict:
    """
    返回用于消息传递的双向邻接（不含自环；残差在GNN内部做）：
      - 'up':   child→parent
      - 'down': parent→child
    """
    edge_up = build_hierarchy_edges(code2idx, direction="child_to_parent").to(device)
    edge_down = build_hierarchy_edges(code2idx, direction="parent_to_child").to(device)
    weight_up = torch.ones(edge_up.size(1), device=device, dtype=torch.float)
    weight_down = torch.ones(edge_down.size(1), device=device, dtype=torch.float)
    return {
        "up": (edge_up, weight_up),
        "down": (edge_down, weight_down),
    }

# ---------------------------------------------
# 替换原有的 get_headersandindex、get_subnote
def get_headersandindex(input_str: str):
    """
    在无任何标点、无换行的连续文本中，
    按 headers_to_select 中的关键子串查找首次出现的位置，
    并按其在原文中的顺序切分。
    """
    low_text = input_str.lower()
    headers_to_select = [
        "chief complaint",
        "major surgical or invasive procedure",
        "history of present illness",
        "past medical history",
        "social history",
        "procedure",
        "family history",
        "physical exam",
        "pertinent results",
        "brief hospital course",
        "medications on admission",
        "discharge disposition",
        "discharge diagnosis",
        "discharge diagnoses",
        "discharge condition",
        "discharge instructions",
        "followup instructions",
    ]
    # 收集每个标题首次出现的位置
    positions = []
    for hdr in headers_to_select:
        idx = low_text.find(hdr)
        if idx != -1:
            positions.append((idx, hdr))
    # 按位置排序
    positions.sort(key=lambda x: x[0])
    if not positions:
        return []
    # 构造 (hdr, start, end) 区间列表
    intervals = []
    for i, (start, hdr) in enumerate(positions):
        end = positions[i+1][0] if i+1 < len(positions) else len(input_str)
        intervals.append((hdr, start, end))
    return intervals

def get_subnote(input_str: str, headers_pos: list):
    """
    将 headers_pos 指定的所有区间依次拼接，
    只保留这些重要区块。
    """
    return "".join(input_str[start:end] for _, start, end in headers_pos)
# ---------------------------------------------

class TextLoader:
    """文本 Tokenizer，不变"""
    def __init__(self, pretrained_model_name: str = "Clinical-Longformer", max_length: int = 4096):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.max_length = max_length

    def __call__(self, text: str) -> dict:
        # 1. 清理掉特殊标记

        # 2. tokenize 判断长度
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_length:
            # 优先按标题提取
            headers_pos = get_headersandindex(text)
            keep_headers = [
                "chief complaint",
                "major surgical or invasive procedure",
                "history of present illness",
                "past medical history",
                "procedure",
                "brief hospital course",
                "discharge diagnosis",
                "discharge diagnoses"
                "discharge condition",
            ]
            headers_pos = [h for h in headers_pos if h[0] in keep_headers]
            if headers_pos:
                sub = get_subnote(text, headers_pos)
                tokens_sub = self.tokenizer.tokenize(sub)
                if len(tokens_sub) <= self.max_length:
                    raw_text = sub
                else:
                    # 提取后仍超长，fallback 到前后截断
                    half = self.max_length // 2
                    head = tokens_sub[:half]; tail = tokens_sub[-half:]
                    raw_text = self.tokenizer.convert_tokens_to_string(head + tail)
            else:
                # 没匹配到标题，直接前后截断
                half = self.max_length // 2
                head = tokens[:half]; tail = tokens[-half:]
                raw_text = self.tokenizer.convert_tokens_to_string(head + tail)
        else:
            raw_text = text
        # 3. 最终固定长度 encoding
        encoding = self.tokenizer(
            raw_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}

class LabelLoader:
    """标签描述 Encoder，不变"""
    def __init__(
        self,
        codes_file: str = "data/filtered_icd_codes_with_desc.feather",
        pretrained_model_name: str = "Bio_ClinicalBERT",
        max_length: int = 128
    ):
        df = pd.read_feather(codes_file)
        self.codes = df['code'].astype(str).tolist()
        self.descriptions = df['description'].tolist()
        self.code2idx = {code: idx for idx, code in enumerate(self.codes)}
        self.idx2code = {idx: code for code, idx in self.code2idx.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.model = AutoModel.from_pretrained(pretrained_model_name)
        self.max_length = max_length
        self.input_ids = None
        self.attention_mask = None

    @property
    def num_labels(self) -> int:
        return len(self.codes)

    def __call__(self) -> torch.Tensor:
        """
        对所有标签描述做 tokenizer 编码并通过预训练模型计算 embedding，返回形状为
        (num_labels, hidden_size) 的 Tensor。
        """
        # 1. tokenize
        enc = self.tokenizer(
            self.descriptions,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = enc['input_ids']
        attention_mask = enc['attention_mask']
        # 2. 通过模型计算 embedding
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, 'pooler_output'):
                label_embs = outputs.pooler_output    # [num_labels, hidden_size]
            else:
                label_embs = outputs.last_hidden_state[:, 0]  # CLS token 表示
        return label_embs


def build_multihot_y(targets: list, code2idx: dict, num_labels: int) -> np.ndarray:
    """
    将 List[List[code_str]] → dense multi-hot 矩阵 np.ndarray(shape=(n_samples, num_labels))
    """
    y = np.zeros((len(targets), num_labels), dtype=np.float32)
    for i, codes in enumerate(targets):
        for c in codes:
            code_str = str(c)
            if code_str in code2idx:
                idx = code2idx[code_str]
                y[i, idx] = 1.0
            else:
                print(f"警告: 代码 '{code_str}' 不在标签字典中，将被跳过")
    return y


class MultiHotLoader:
    """多热标签 Loader，实现 __call__ 接口"""
    def __init__(self, code2idx: dict, num_labels: int):
        self.code2idx = code2idx
        self.num_labels = num_labels

    def __call__(self, codes: list) -> np.ndarray:
        y = np.zeros(self.num_labels, dtype=np.float32)
        for c in codes:
            code_str = str(c)
            if code_str in self.code2idx:
                y[self.code2idx[code_str]] = 1.0
            else:
                print(f"警告: 代码 '{code_str}' 不在标签字典中，将被跳过")
        return y


class ICDMultiLabelDataset(Dataset):
    """
    普通多标签 Dataset，训练时返回 (x_dict, y_multihot),
    验证/测试时只返回 x_dict。
    x_dict 中包含:
      - 'input_ids':  Tensor[seq_len]
      - 'attention_mask': Tensor[seq_len]
    y_multihot: np.ndarray[num_labels]
    """
    def __init__(self, data_file: str, text_loader: TextLoader, label_loader: LabelLoader, mode="train"):
        df = pd.read_feather(data_file)
        texts = df['text'].tolist()
        targets = df['target'].tolist()
        self.texts = texts
        self.targets = targets
        self.df = df  # 保存原始DataFrame以便访问split等列
        self.text_loader = text_loader
        self.code2idx = label_loader.code2idx
        self.num_labels = label_loader.num_labels
        self.training = (mode == "train")
        # 初始化多热标签 loader
        self.multihot_loader = MultiHotLoader(self.code2idx, self.num_labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.text_loader(self.texts[idx])
        x = {
            'input_ids': enc['input_ids'],
            'attention_mask': enc['attention_mask']
        }
        if self.training:
            # 直接调用 MultiHotLoader 的 __call__ 接口
            y = self.multihot_loader(self.targets[idx])
            return x, y
        return x


class SynonymLabelLoader(LabelLoader):
    """
    支持多类型术语的 LabelLoader：
    - 从 synonyms_file 加载同义词字典（ICD code -> [同义词列表]）
    - 为每个 code 准备 term_count 个术语
    - 支持缩写开关 use_abbreviations：
      * 开启时：第1个术语=原描述，第2个=缩写组合，第3个=通用表达组合，剩余=同义词
      * 关闭时：第1个术语=原描述，剩余=同义词
    - 支持 'random'/'max'/'mean' 三种同义词排序策略
    """
    def __init__(
        self,
        codes_file: str = "data/filtered_icd_codes_with_desc.feather",
        synonyms_file: str = "data/icd_synonyms.json",
        abbreviations_file: str = None,
        pretrained_model_name: str = "Bio_ClinicalBERT",
        max_length: int = 128,
        term_count: int = 4,
        sort_method: str = 'random',
        use_abbreviations: bool = False
    ):
        super().__init__(
            codes_file=codes_file,
            pretrained_model_name=pretrained_model_name,
            max_length=max_length
        )
        # 加载同义词表
        with open(synonyms_file, 'r') as f:
            self.icd_syn = json.load(f)
        
        # 加载缩写文件（可选）
        self.icd_abbr = {}
        if abbreviations_file and os.path.exists(abbreviations_file):
            with open(abbreviations_file, 'r', encoding='utf-8') as f:
                self.icd_abbr = json.load(f)
        
        self.term_count = term_count
        self.sort_method = sort_method
        self.abbreviations_file = abbreviations_file
        self.use_abbreviations = use_abbreviations

    def __call__(self) -> torch.Tensor:
        """
        对每个标签（含同义词）做 tokenizer 编码并通过模型计算 embedding，返回形状为
        (num_labels * term_count, hidden_size) 的 Tensor。
        """
        terms = []
        # 对每个 code 构建术语列表
        for code, desc in zip(self.codes, self.descriptions):
            # 构建术语列表
            code_terms = [desc]  # 原始描述始终在第一位
            
            if self.use_abbreviations:
                # 获取缩写和通用表达（如果存在）
                abbreviations = []
                common_expressions = []
                if code in self.icd_abbr:
                    abbr_data = self.icd_abbr[code]
                    abbreviations = abbr_data.get("abbreviations", [])
                    common_expressions = abbr_data.get("common_expressions", [])
                
                # 添加缩写组合作为第二个术语（如果有的话）
                if abbreviations:
                    abbr_combined = " ".join(abbreviations)
                    code_terms.append(abbr_combined)
                
                # 添加通用表达组合作为第三个术语（如果有的话）
                if common_expressions:
                    expr_combined = " ".join(common_expressions)
                    code_terms.append(expr_combined)
            
            # 获取同义词填充剩余位置
            syns = self.icd_syn.get(code, [])
            
            # 对同义词应用排序策略
            if self.sort_method == 'random':
                random.shuffle(syns)
            elif self.sort_method == 'max':
                syns = sorted(syns, key=lambda x: len(x), reverse=True)
            elif self.sort_method == 'mean':
                syns = sorted(syns, key=lambda x: len(x))
            
            # 用同义词填充剩余的 term_count 位置
            remaining_slots = self.term_count - len(code_terms)
            if remaining_slots > 0:
                if len(syns) >= remaining_slots:
                    selected_syns = syns[:remaining_slots]
                elif len(syns) > 0:
                    # 循环补齐不足的同义词
                    repeat = int(remaining_slots / len(syns)) + 1
                    selected_syns = (syns * repeat)[:remaining_slots]
                else:
                    # 如果完全没有同义词，用原始描述填充
                    selected_syns = [desc] * remaining_slots
                
                code_terms.extend(selected_syns)
            
            terms.extend(code_terms[:self.term_count])

        # 对所有术语一起做 tokenizer
        enc = self.tokenizer(
            terms,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = enc['input_ids']         # [num_labels*term_count, max_length]
        attention_mask = enc['attention_mask']
        # 通过模型计算 embedding
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, 'pooler_output'):
                label_embs = outputs.pooler_output
            else:
                label_embs = outputs.last_hidden_state[:, 0]
        return label_embs
    
def build_adj_matrix(dataset, num_labels: int, term_count: int = 1,
                     mode: str = "ppmi", add_self_loop: bool = True,
                     topk: int = 20, device: str = "cpu") -> tuple[torch.LongTensor, torch.FloatTensor]:
    """
    构建稀疏标签共现图 edge_index 和 edge_weight，适用于 GATConv。

    Args:
      dataset: 包含 df['split'], df['target'] 多热向量 (N, C)
      num_labels: 标签总数 C
      term_count: 同义词数 k
      mode: "binary"/"count"/"ppmi"
      add_self_loop: 是否添加自环
      topk: 每行保留 topk 边
      device: 'cpu' or 'cuda'

    Returns:
      edge_index: torch.LongTensor [2, E]
      edge_weight: torch.FloatTensor [E]
    """
    # 1) 过滤生成样本
    df = dataset.df
    if 'split' in df.columns:
        df = df[df['split'] != 'generate']
    # 2) 聚合标签矩阵：先把每行 code 列表转成长度为 C 的 multi-hot 向量
    targets = df['target'].tolist()  # List[List[code_str]]
    # build_multihot_y 已在本文件中定义，返回 shape=(N, C) 的 np.ndarray
    labels_np = build_multihot_y(targets, dataset.code2idx, num_labels)  # (N, C)
    L = torch.from_numpy(labels_np).float().to(device)  # (N, C)

    # 3) 计算共现矩阵
    co_occ = L.t() @ L                                  # (C, C)

    # 4) 构造初步权重矩阵
    if mode == 'binary':
        W = (co_occ > 0).float()
    elif mode == 'count':
        W = co_occ
    elif mode == 'ppmi':
        N = L.size(0)
        p_ij = co_occ / N
        p_i  = co_occ.diag() / N
        pmi  = torch.log(p_ij / (p_i.unsqueeze(1) * p_i.unsqueeze(0) + 1e-9) + 1e-9)
        W    = F.relu(pmi)
        # 增加阈值筛选，只保留大于0.1的边
        W = W.masked_fill(W < 0.1, 0.0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 5) top-k 稀疏化
    C = num_labels
    vals, idx = torch.topk(W, k=topk, dim=1)            # (C, topk)
    row = torch.arange(C, device=device).unsqueeze(1).repeat(1, topk).reshape(-1)
    col = idx.reshape(-1)
    weight = vals.reshape(-1)

    # 6) 同义词扩展
    if term_count > 1:
        k = term_count
        offsets = torch.arange(k, device=device)
        row = (row.unsqueeze(1) * k + offsets).reshape(-1)
        col = (col.unsqueeze(1) * k + offsets).reshape(-1)
        weight = weight.unsqueeze(1).repeat(1, k).reshape(-1)
        N = C * k
    else:
        N = C

    # 7) 添加自环
    edge_index = torch.stack([row, col], dim=0)          # (2, E)
    edge_weight = weight                                # (E,)
    if add_self_loop:
        loops = torch.arange(N, device=device)
        self_loop_idx = torch.stack([loops, loops], dim=0)
        edge_index = torch.cat([edge_index, self_loop_idx], dim=1)
        edge_weight = torch.cat([edge_weight, torch.ones(N, device=device)], dim=0)

    return edge_index, edge_weight

# ---------------- 新增：ICD9 层级推断 & 构图 ----------------
def _infer_parent(code: str, code_set: set[str]) -> str | None:
    """
    基于 ICD-9 代码字符串前缀推断最近父级：
    - 逐步去掉末尾字符；若末尾是 '.' 则一并去掉；
    - 一旦得到的前缀在 code_set 中，返回该前缀作为最近父级；
    - 找不到则返回 None。
    """
    cand = code.strip()
    while len(cand) > 1:
        cand = cand[:-1]
        if cand.endswith('.'):
            cand = cand[:-1]
        if cand in code_set:
            return cand
    return None

def build_hierarchy_edges(code2idx: dict, direction: str = "parent_to_child") -> torch.LongTensor:
    """
    构建 1-hop 层级边（代码级节点）：
    - direction = 'parent_to_child' 或 'child_to_parent'
    - 仅返回 edge_index（权重默认为 1，下游可选用）
    """
    codes = list(code2idx.keys())
    code_set = set(codes)
    rows, cols = [], []
    for c in codes:
        p = _infer_parent(c, code_set)
        if p is None:
            continue
        ci = code2idx[c]
        pi = code2idx[p]
        if direction == "parent_to_child":
            rows.append(pi); cols.append(ci)
        else:
            rows.append(ci); cols.append(pi)
    if len(rows) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([rows, cols], dtype=torch.long)

def build_hierarchy_adjs(code2idx: dict, device: str = "cpu") -> dict:
    """
    返回用于消息传递的双向邻接（不含自环；残差在GNN内部做）：
      - 'up':   child→parent
      - 'down': parent→child
    """
    edge_up = build_hierarchy_edges(code2idx, direction="child_to_parent").to(device)
    edge_down = build_hierarchy_edges(code2idx, direction="parent_to_child").to(device)
    weight_up = torch.ones(edge_up.size(1), device=device, dtype=torch.float)
    weight_down = torch.ones(edge_down.size(1), device=device, dtype=torch.float)
    return {
        "up": (edge_up, weight_up),
        "down": (edge_down, weight_down),
    }



