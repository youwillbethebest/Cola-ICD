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

# ---------------- ICD9 Hierarchy Inference & Graph Building ----------------
def _infer_parent(code: str, code_set: set[str]) -> str | None:
    """
    Infer nearest parent based on ICD-9 code string prefix:
    - Progressively remove trailing characters; if trailing char is '.' remove it too;
    - Once a prefix is found in code_set, return that prefix as nearest parent;
    - Return None if not found.
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
    Build 1-hop hierarchy edges (code-level nodes):
    - direction = 'parent_to_child' or 'child_to_parent'
    - Only returns edge_index (weight defaults to 1, optional for downstream use)
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
    Return bidirectional adjacency for message passing (no self-loops; residual done inside GNN):
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
# Replacement for original get_headersandindex, get_subnote
def get_headersandindex(input_str: str):
    """
    In continuous text without punctuation or newlines,
    find first occurrence positions of key substrings in headers_to_select,
    and split according to their order in original text.
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
    # Collect first occurrence positions of each header
    positions = []
    for hdr in headers_to_select:
        idx = low_text.find(hdr)
        if idx != -1:
            positions.append((idx, hdr))
    # Sort by position
    positions.sort(key=lambda x: x[0])
    if not positions:
        return []
    # Build (hdr, start, end) interval list
    intervals = []
    for i, (start, hdr) in enumerate(positions):
        end = positions[i+1][0] if i+1 < len(positions) else len(input_str)
        intervals.append((hdr, start, end))
    return intervals

def get_subnote(input_str: str, headers_pos: list):
    """
    Concatenate all intervals specified by headers_pos,
    keep only these important sections.
    """
    return "".join(input_str[start:end] for _, start, end in headers_pos)
# ---------------------------------------------

class TextLoader:
    """Text Tokenizer, unchanged"""
    def __init__(self, pretrained_model_name: str = "Clinical-Longformer", max_length: int = 4096):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.max_length = max_length

    def __call__(self, text: str) -> dict:
        # 1. Clean up special tokens

        # 2. Tokenize and check length
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_length:
            # Priority extraction by headers
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
                    # Still too long after extraction, fallback to head+tail truncation
                    half = self.max_length // 2
                    head = tokens_sub[:half]; tail = tokens_sub[-half:]
                    raw_text = self.tokenizer.convert_tokens_to_string(head + tail)
            else:
                # No matching headers, directly truncate head+tail
                half = self.max_length // 2
                head = tokens[:half]; tail = tokens[-half:]
                raw_text = self.tokenizer.convert_tokens_to_string(head + tail)
        else:
            raw_text = text
        # 3. Final fixed length encoding
        encoding = self.tokenizer(
            raw_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}

class LabelLoader:
    """Label Description Encoder, unchanged"""
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
        Tokenize all label descriptions and compute embeddings through pretrained model,
        return Tensor of shape (num_labels, hidden_size).
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
        # 2. Compute embedding through model
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, 'pooler_output'):
                label_embs = outputs.pooler_output    # [num_labels, hidden_size]
            else:
                label_embs = outputs.last_hidden_state[:, 0]  # CLS token representation
        return label_embs


def build_multihot_y(targets: list, code2idx: dict, num_labels: int) -> np.ndarray:
    """
    Convert List[List[code_str]] → dense multi-hot matrix np.ndarray(shape=(n_samples, num_labels))
    """
    y = np.zeros((len(targets), num_labels), dtype=np.float32)
    for i, codes in enumerate(targets):
        for c in codes:
            code_str = str(c)
            if code_str in code2idx:
                idx = code2idx[code_str]
                y[i, idx] = 1.0
            else:
                print(f"Warning: code '{code_str}' not in label dictionary, will be skipped")
    return y


class MultiHotLoader:
    """Multi-hot Label Loader, implements __call__ interface"""
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
                print(f"Warning: code '{code_str}' not in label dictionary, will be skipped")
        return y


class ICDMultiLabelDataset(Dataset):
    """
    Standard multi-label Dataset, returns (x_dict, y_multihot) during training,
    only returns x_dict during validation/testing.
    x_dict contains:
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
        self.df = df  # Save original DataFrame to access split and other columns
        self.text_loader = text_loader
        self.code2idx = label_loader.code2idx
        self.num_labels = label_loader.num_labels
        self.training = (mode == "train")
        # Initialize multi-hot label loader
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
            # Directly call MultiHotLoader's __call__ interface
            y = self.multihot_loader(self.targets[idx])
            return x, y
        return x


class SynonymLabelLoader(LabelLoader):
    """
    LabelLoader supporting multiple types of terms:
    - Load synonym dictionary from synonyms_file (ICD code -> [synonym list])
    - Prepare term_count terms for each code
    - Support abbreviation switch use_abbreviations:
      * When enabled: 1st term=original description, 2nd=abbreviation combination, 3rd=common expression combination, rest=synonyms
      * When disabled: 1st term=original description, rest=synonyms
    - Support 'random'/'max'/'mean' three synonym sorting strategies
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
        # Load synonym dictionary
        with open(synonyms_file, 'r') as f:
            self.icd_syn = json.load(f)
        
        # Load abbreviation file (optional)
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
        Tokenize each label (including synonyms) and compute embedding through model,
        return Tensor of shape (num_labels * term_count, hidden_size).
        """
        terms = []
        # Build term list for each code
        for code, desc in zip(self.codes, self.descriptions):
            # Build term list
            code_terms = [desc]  # Original description always first
            
            if self.use_abbreviations:
                # Get abbreviations and common expressions (if exist)
                abbreviations = []
                common_expressions = []
                if code in self.icd_abbr:
                    abbr_data = self.icd_abbr[code]
                    abbreviations = abbr_data.get("abbreviations", [])
                    common_expressions = abbr_data.get("common_expressions", [])
                
                # Add abbreviation combination as second term (if available)
                if abbreviations:
                    abbr_combined = " ".join(abbreviations)
                    code_terms.append(abbr_combined)
                
                # Add common expression combination as third term (if available)
                if common_expressions:
                    expr_combined = " ".join(common_expressions)
                    code_terms.append(expr_combined)
            
            # Get synonyms to fill remaining positions
            syns = self.icd_syn.get(code, [])
            
            # Apply sorting strategy to synonyms
            if self.sort_method == 'random':
                random.shuffle(syns)
            elif self.sort_method == 'max':
                syns = sorted(syns, key=lambda x: len(x), reverse=True)
            elif self.sort_method == 'mean':
                syns = sorted(syns, key=lambda x: len(x))
            
            # Fill remaining term_count positions with synonyms
            remaining_slots = self.term_count - len(code_terms)
            if remaining_slots > 0:
                if len(syns) >= remaining_slots:
                    selected_syns = syns[:remaining_slots]
                elif len(syns) > 0:
                    # Repeat to fill insufficient synonyms
                    repeat = int(remaining_slots / len(syns)) + 1
                    selected_syns = (syns * repeat)[:remaining_slots]
                else:
                    # If no synonyms at all, fill with original description
                    selected_syns = [desc] * remaining_slots
                
                code_terms.extend(selected_syns)
            
            terms.extend(code_terms[:self.term_count])

        # Tokenize all terms together
        enc = self.tokenizer(
            terms,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = enc['input_ids']         # [num_labels*term_count, max_length]
        attention_mask = enc['attention_mask']
        # Compute embedding through model
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
    Build sparse label co-occurrence graph edge_index and edge_weight, suitable for GATConv.

    Args:
      dataset: contains df['split'], df['target'] multi-hot vectors (N, C)
      num_labels: total number of labels C
      term_count: number of synonyms k
      mode: "binary"/"count"/"ppmi"
      add_self_loop: whether to add self-loops
      topk: keep topk edges per row
      device: 'cpu' or 'cuda'

    Returns:
      edge_index: torch.LongTensor [2, E]
      edge_weight: torch.FloatTensor [E]
    """
    # 1) Filter generated samples
    df = dataset.df
    if 'split' in df.columns:
        df = df[df['split'] != 'generate']
    # 2) Aggregate label matrix: convert each row's code list to length C multi-hot vector
    targets = df['target'].tolist()  # List[List[code_str]]
    # build_multihot_y defined in this file, returns np.ndarray of shape=(N, C)
    labels_np = build_multihot_y(targets, dataset.code2idx, num_labels)  # (N, C)
    L = torch.from_numpy(labels_np).float().to(device)  # (N, C)

    # 3) Compute co-occurrence matrix
    co_occ = L.t() @ L                                  # (C, C)

    # 4) Build preliminary weight matrix
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
        # Add threshold filtering, only keep edges greater than 0.1
        W = W.masked_fill(W < 0.1, 0.0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 5) Top-k sparsification
    C = num_labels
    vals, idx = torch.topk(W, k=topk, dim=1)            # (C, topk)
    row = torch.arange(C, device=device).unsqueeze(1).repeat(1, topk).reshape(-1)
    col = idx.reshape(-1)
    weight = vals.reshape(-1)

    # 6) Synonym expansion
    if term_count > 1:
        k = term_count
        offsets = torch.arange(k, device=device)
        row = (row.unsqueeze(1) * k + offsets).reshape(-1)
        col = (col.unsqueeze(1) * k + offsets).reshape(-1)
        weight = weight.unsqueeze(1).repeat(1, k).reshape(-1)
        N = C * k
    else:
        N = C

    # 7) Add self-loops
    edge_index = torch.stack([row, col], dim=0)          # (2, E)
    edge_weight = weight                                # (E,)
    if add_self_loop:
        loops = torch.arange(N, device=device)
        self_loop_idx = torch.stack([loops, loops], dim=0)
        edge_index = torch.cat([edge_index, self_loop_idx], dim=1)
        edge_weight = torch.cat([edge_weight, torch.ones(N, device=device)], dim=0)

    return edge_index, edge_weight

# ---------------- ICD9 Hierarchy Inference & Graph Building (duplicate) ----------------
def _infer_parent(code: str, code_set: set[str]) -> str | None:
    """
    Infer nearest parent based on ICD-9 code string prefix:
    - Progressively remove trailing characters; if trailing char is '.' remove it too;
    - Once a prefix is found in code_set, return that prefix as nearest parent;
    - Return None if not found.
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
    Build 1-hop hierarchy edges (code-level nodes):
    - direction = 'parent_to_child' or 'child_to_parent'
    - Only returns edge_index (weight defaults to 1, optional for downstream use)
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
    Return bidirectional adjacency for message passing (no self-loops; residual done inside GNN):
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



