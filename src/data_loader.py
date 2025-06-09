import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import json
import random

class TextLoader:
    """文本 Tokenizer，不变"""
    def __init__(self, pretrained_model_name: str = "Clinical-Longformer", max_length: int = 4096):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.max_length = max_length

    def __call__(self, text: str) -> dict:
        encoding = self.tokenizer(
            text,
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

    def get_label_encodings(self) -> dict:
        return self()

    def __call__(self) -> dict:
        """
        对所有标签描述做 tokenizer 编码并返回，
        同时缓存到 self.input_ids / self.attention_mask
        """
        enc = self.tokenizer(
            self.descriptions,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        self.input_ids = enc['input_ids']
        self.attention_mask = enc['attention_mask']
        return {'input_ids': self.input_ids, 'attention_mask': self.attention_mask}


def build_multihot_y(targets: list, code2idx: dict, num_labels: int) -> np.ndarray:
    """
    将 List[List[code_str]] → dense multi-hot 矩阵 np.ndarray(shape=(n_samples, num_labels))
    """
    y = np.zeros((len(targets), num_labels), dtype=np.float32)
    for i, codes in enumerate(targets):
        for c in codes:
            idx = code2idx[str(c)]
            y[i, idx] = 1.0
    return y


class MultiHotLoader:
    """多热标签 Loader，实现 __call__ 接口"""
    def __init__(self, code2idx: dict, num_labels: int):
        self.code2idx = code2idx
        self.num_labels = num_labels

    def __call__(self, codes: list) -> np.ndarray:
        y = np.zeros(self.num_labels, dtype=np.float32)
        for c in codes:
            y[self.code2idx[str(c)]] = 1.0
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
    支持同义词的 LabelLoader：
    - 从 synonyms_file 加载同义词字典（ICD code -> [同义词列表]）
    - 为每个 code 准备 term_count 个术语（第一个是原始描述，后面补充同义词）
    - 支持 'random'/'max'/'mean' 三种同义词排序策略
    """
    def __init__(
        self,
        codes_file: str = "data/filtered_icd_codes_with_desc.feather",
        synonyms_file: str = "data/icd_synonyms.json",
        pretrained_model_name: str = "Bio_ClinicalBERT",
        max_length: int = 128,
        term_count: int = 4,
        sort_method: str = 'random'
    ):
        super().__init__(
            codes_file=codes_file,
            pretrained_model_name=pretrained_model_name,
            max_length=max_length
        )
        # 加载同义词表
        with open(synonyms_file, 'r') as f:
            self.icd_syn = json.load(f)
        self.term_count = term_count
        self.sort_method = sort_method

    def __call__(self) -> dict:
        """
        返回一个 dict：
          'input_ids': Tensor[(num_labels * term_count), max_length]
          'attention_mask': Tensor[(num_labels * term_count), max_length]
        """
        terms = []
        # 对每个 code 构建一个 [原始描述 + 同义词列表]
        for code, desc in zip(self.codes, self.descriptions):
            syns = self.icd_syn.get(code, [])
            # 排序策略
            if self.sort_method == 'random':
                random.shuffle(syns)
            elif self.sort_method == 'max':
                syns = sorted(syns, key=lambda x: len(x), reverse=True)
            elif self.sort_method == 'mean':
                syns = sorted(syns, key=lambda x: len(x))
            # 取 term_count-1 个同义词，不足时循环补齐
            if len(syns) >= self.term_count - 1:
                sel = syns[:self.term_count - 1]
            else:
                sel = syns
                if sel:
                    repeat = int((self.term_count - 1) / len(sel)) + 1
                    sel = (sel * repeat)[:self.term_count - 1]
            # 原始描述总是在第一个
            terms.extend([desc] + sel)

        # 对所有术语一起做 tokenizer
        enc = self.tokenizer(
            terms,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'],       # 形状: [num_labels*term_count, max_length]
            'attention_mask': enc['attention_mask']
        }



