import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from src.data_loader import TextLoader, LabelLoader, ICDMultiLabelDataset
from src.model import ClinicalLongformerLabelAttention
from src.metric import MetricCollection, Precision, Recall, F1Score, MeanAveragePrecision, AUC,Precision_K
from src.trainer import Trainer


text_loader = TextLoader(pretrained_model_name="Clinical-Longformer", max_length=4096)
print("text_loader loaded")
label_loader = LabelLoader(codes_file="data/filtered_icd_codes_with_desc.feather", pretrained_model_name="Bio_ClinicalBERT", max_length=128)
print("label_loader loaded")
test_dataset = ICDMultiLabelDataset(data_file="data/mimiciv_icd9_test.feather", text_loader=text_loader, label_loader=label_loader)
print("test_dataset loaded")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
print("test_loader loaded")