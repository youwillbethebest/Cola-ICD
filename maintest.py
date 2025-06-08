import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from src.data_loader import TextLoader, LabelLoader, ICDMultiLabelDataset
from src.model import ClinicalLongformerLabelAttention
from src.metric import MetricCollection, Precision, Recall, F1Score, MeanAveragePrecision, AUC, Precision_K, LossMetric
from src.trainer import Trainer

# 配置
train_file = "data/mimiciv_icd9_train.feather"
val_file = "data/mimiciv_icd9_val.feather"
test_file = "data/mimiciv_icd9_test.feather"
codes_file = "data/filtered_icd_codes_with_desc.feather"
pretrained_model_name = "Clinical-Longformer"
label_model_name = "Bio_ClinicalBERT"
max_length = 4096
label_max_length = 128
batch_size = 8
epochs = 1
lr = 2e-5
weight_decay = 0.0
warmup_steps = 0
gradient_accumulation_steps = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = True
use_wandb = False
output_dir = "checkpoints_test"
best_metric_name = "map"

os.makedirs(output_dir, exist_ok=True)

print("Initializing loaders...")
text_loader = TextLoader(pretrained_model_name=pretrained_model_name, max_length=max_length)
label_loader = LabelLoader(codes_file=codes_file, pretrained_model_name=label_model_name, max_length=label_max_length)

print("Loading datasets and sampling first 1000 examples...")
train_full = ICDMultiLabelDataset(data_file=train_file, text_loader=text_loader, label_loader=label_loader)
val_full = ICDMultiLabelDataset(data_file=val_file, text_loader=text_loader, label_loader=label_loader)
test_full = ICDMultiLabelDataset(data_file=test_file, text_loader=text_loader, label_loader=label_loader)

sample_size = 1000
train_dataset = Subset(train_full, range(min(sample_size, len(train_full))))
val_dataset = Subset(val_full, range(min(sample_size, len(val_full))))
test_dataset = Subset(test_full, range(min(sample_size, len(test_full))))
print(f"Sampled {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test examples")

print("Creating data loaders...")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

print("Initializing model...")
model = ClinicalLongformerLabelAttention(
    longformer_path=pretrained_model_name,
    codes_file=codes_file,
    label_model_name=label_model_name
)

print("Setting up optimizer and scheduler...")
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
total_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
criterion = nn.BCEWithLogitsLoss()

print("Initializing metrics...")
metrics = MetricCollection([
    Precision(number_of_classes=label_loader.num_labels, average="macro"),
    Precision(number_of_classes=label_loader.num_labels, average="micro"),
    F1Score(number_of_classes=label_loader.num_labels, average="macro"),
    F1Score(number_of_classes=label_loader.num_labels, average="micro"),
    AUC(number_of_classes=label_loader.num_labels, average="macro"),
    AUC(number_of_classes=label_loader.num_labels, average="micro"),
    Precision_K(k=10),
    Precision_K(k=8),
    Precision_K(k=5),
    MeanAveragePrecision(),
    LossMetric()
])
metrics.to(device)

print("Initializing trainer...")
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    metrics=metrics,
    device=device,
    epochs=epochs,
    gradient_accumulation_steps=gradient_accumulation_steps,
    output_dir=output_dir,
    best_metric_name=best_metric_name,
    use_amp=use_amp,
    use_wandb=use_wandb
)

print("Training started")
trainer.train()
print("Training completed")