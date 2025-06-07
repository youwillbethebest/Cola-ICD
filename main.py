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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description="Trainer for ICD multi-label classification using label attention model")
    parser.add_argument("--train_file",default="data/mimiciv_icd9_train.feather", type=str, help="Path to training data (.feather)")
    parser.add_argument("--val_file",default="data/mimiciv_icd9_val.feather", type=str, help="Path to validation data (.feather)")
    parser.add_argument("--test_file",default="data/mimiciv_icd9_test.feather", type=str, help="Path to test data (.feather)")  
    parser.add_argument("--codes_file", type=str, default="data/filtered_icd_codes_with_desc.feather", help="Path to ICD codes and descriptions file")
    parser.add_argument("--pretrained_model_name", type=str, default="Clinical-Longformer", help="Pretrained model for text encoding")
    parser.add_argument("--label_model_name", type=str, default="Bio_ClinicalBERT", help="Pretrained model for label encoding")
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length for text")
    parser.add_argument("--label_max_length", type=int, default=128, help="Max sequence length for label descriptions")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--best_metric_name", type=str, default="MeanAveragePrecision", help="Metric name to select best model")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 数据加载和模型初始化
    print("Loading text tokenizer...")
    text_loader = TextLoader(pretrained_model_name=args.pretrained_model_name, max_length=args.max_length)
    print("Loading label tokenizer and model...")
    label_loader = LabelLoader(codes_file=args.codes_file, pretrained_model_name=args.label_model_name, max_length=args.label_max_length)
    print(f"Number of labels: {label_loader.num_labels}")

    print("Creating training dataset...")
    train_dataset = ICDMultiLabelDataset(data_file=args.train_file, text_loader=text_loader, label_loader=label_loader)
    print("Creating validation dataset...")
    val_dataset = ICDMultiLabelDataset(data_file=args.val_file, text_loader=text_loader, label_loader=label_loader)
    test_dataset = ICDMultiLabelDataset(data_file=args.test_file, text_loader=text_loader, label_loader=label_loader)
    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    print("Initializing model...")
    model = ClinicalLongformerLabelAttention(
        longformer_path=args.pretrained_model_name,
        codes_file=args.codes_file,
        label_model_name=args.label_model_name
    )

    print("Setting up optimizer and scheduler...")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_steps)

    criterion = nn.BCEWithLogitsLoss()
    print("Initializing metrics...")
    metrics = MetricCollection([
        Precision(number_of_classes=label_loader.num_labels, average="macro"),
        Precision(number_of_classes=label_loader.num_labels, average="micro"),
        F1Score(number_of_classes=label_loader.num_labels, average="macro"),
        F1Score(number_of_classes=label_loader.num_labels, average="micro"),
        AUC(number_of_classes=label_loader.num_labels,average="macro"),
        AUC(number_of_classes=label_loader.num_labels,average="micro"),
        Precision_K(k=10),
        Precision_K(k=8),
        Precision_K(k=5),
        MeanAveragePrecision()
    ])
    metrics.to(device)
    
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,  # 使用验证集作为测试集
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        device=device,
        epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
        best_metric_name=args.best_metric_name
    )
    print("Training started")
    trainer.train()
    print("Training completed")


if __name__ == "__main__":
    main()
