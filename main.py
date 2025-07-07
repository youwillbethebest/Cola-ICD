import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from torch.optim import AdamW
from src.data_loader import TextLoader, LabelLoader, ICDMultiLabelDataset, SynonymLabelLoader
from src.model import ClinicalLongformerLabelAttention
from src.metric import MetricCollection, Precision, Recall, F1Score, MeanAveragePrecision, AUC, Precision_K, LossMetric
from src.trainer import Trainer
import wandb
from datetime import datetime
from src.module import JaccardWeightedSupConLoss, LabelWiseContrastiveLoss

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_ddp(rank, world_size):
    """初始化DDP环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """清理DDP环境"""
    dist.destroy_process_group()

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
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--best_metric_name", type=str, default="map", help="Metric name to select best model")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Whether to use mixed precision training")
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Whether to enable Weights & Biases logging")
    parser.add_argument("--term_count",type=int, default=1, help="Whether to use synonym")
    parser.add_argument("--use_ddp", action="store_true", default=False, help="Whether to use DDP for distributed training")
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use for DDP")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for metrics")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint for resuming training")
    parser.add_argument("--scheduler_type", type=str, default="cosine", choices=["linear", "cosine", "cosine_restart", "polynomial", "constant"], help="Learning-rate scheduler strategy")
    
    # 添加早停相关参数
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Number of epochs to wait before early stopping if no improvement")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0001, help="Minimum change required to qualify as an improvement")
    parser.add_argument("--early_stopping", action="store_true", default=False, help="Whether to use early stopping")
    
    parser.add_argument("--eval_codes_file", type=str, default=None,
                        help="txt/json 列表文件；若给定则只在这些 code 上计算指标")
    
    # ---- 对比学习相关参数 ----
    parser.add_argument("--use_contrastive", action="store_true", default=False,
                        help="是否启用对比学习")
    parser.add_argument("--contrastive_loss_weight", type=float, default=0.1,
                        help="对比学习的损失权重")
    parser.add_argument("--contrastive_temperature", type=float, default=0.1,
                        help="对比学习的温度 τ (for LabelWiseContrastiveLoss)")

    # ---- GNN和共现矩阵相关参数 ----
    parser.add_argument("--use_gnn", action="store_true", default=False,
                        help="是否启用GNN更新标签嵌入")
    parser.add_argument("--adj_matrix_mode", type=str, default="ppmi", 
                        choices=["binary", "count", "ppmi"],
                        help="邻接矩阵构建模式")
    parser.add_argument("--adj_matrix_topk", type=int, default=20,
                        help="邻接矩阵每行保留的边数 topk")
    parser.add_argument("--adj_matrix_self_loop", action="store_true", default=True,
                        help="是否在邻接矩阵中添加自环")
    parser.add_argument("--adj_matrix_device", type=str, default="cpu",
                        help="构建邻接矩阵时使用的设备")

    return parser.parse_args()

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_warmup_steps: int,
    num_training_steps: int,
    *,
    num_cycles: int = 1,
    power: float = 2.0,
):
    if scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif scheduler_type == "cosine_restart":
        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, num_cycles=num_cycles
        )
    elif scheduler_type == "polynomial":
        return get_polynomial_decay_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, power=power
        )
    elif scheduler_type == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

def main_worker(rank, args):
    """DDP训练的工作进程"""
    if args.use_ddp:
        setup_ddp(rank, args.world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device(args.device)
    
    # W&B初始化（只在rank 0进行）
    if args.use_wandb and (not args.use_ddp or rank == 0):
        now = datetime.now().strftime("%m-%d_%H-%M")
        wandb.init(project="Attentionicd", name=f"Attentionicd_{now}")
        wandb.config.update(vars(args), allow_val_change=True)
    else:
        print("W&B logging disabled.")
    device = torch.device(args.device)
    print(f"Use AMP: {args.use_amp}")
    print(f"Using device: {device}")

    # 数据加载和模型初始化
    print("Loading text tokenizer...")
    text_loader = TextLoader(pretrained_model_name=args.pretrained_model_name, max_length=args.max_length)
    print("Loading label tokenizer and model...")
    label_loader = SynonymLabelLoader(codes_file=args.codes_file, pretrained_model_name=args.label_model_name, max_length=args.label_max_length,term_count=args.term_count)
    print(f"Number of labels: {label_loader.num_labels}")

    print("Creating training dataset...")
    train_dataset = ICDMultiLabelDataset(data_file=args.train_file, text_loader=text_loader, label_loader=label_loader)
    print("Creating validation dataset...")
    val_dataset = ICDMultiLabelDataset(data_file=args.val_file, text_loader=text_loader, label_loader=label_loader)
    test_dataset = ICDMultiLabelDataset(data_file=args.test_file, text_loader=text_loader, label_loader=label_loader)
    
    # 创建分布式采样器
    if args.use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=args.world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
    
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=8 // args.world_size if args.use_ddp else 8, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        sampler=val_sampler,
        shuffle=False, 
        num_workers=8 // args.world_size if args.use_ddp else 8, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        sampler=test_sampler,
        shuffle=False, 
        num_workers=8 // args.world_size if args.use_ddp else 8, 
        pin_memory=True
    )
    print("Initializing model...")
    
    # 构建邻接矩阵（如果启用GNN）
    adj_matrix = None
    if args.use_gnn:
        print("Building adjacency matrix for GNN...")
        from src.data_loader import build_adj_matrix
        edge_index, edge_weight = build_adj_matrix(
            dataset=train_dataset,
            num_labels=label_loader.num_labels,
            term_count=args.term_count,
            mode=args.adj_matrix_mode,
            add_self_loop=args.adj_matrix_self_loop,
            topk=args.adj_matrix_topk,
            device=args.adj_matrix_device
        )
        adj_matrix = (edge_index, edge_weight)
        print(f"Adjacency matrix shape: {edge_index.shape}, {edge_weight.shape}")
    
    model = ClinicalLongformerLabelAttention(
        longformer_path=args.pretrained_model_name,
        term_counts=args.term_count,
        label_loader=label_loader,
        use_gnn=args.use_gnn,
        adj_matrix=adj_matrix
    )
    
    model.to(device)
    
    # 包装为DDP模型
    if args.use_ddp:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    # W&B监控（只在rank 0）
    if args.use_wandb and (not args.use_ddp or rank == 0):
        wandb.watch(model)

    print("Setting up optimizer and scheduler...")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    scheduler = build_scheduler(
        optimizer,
        args.scheduler_type,
        args.warmup_steps,
        total_steps,
    )

    criterion = nn.BCEWithLogitsLoss()
    if args.use_contrastive:
        print("Contrastive learning enabled (Label-Wise).")
        contrastive_criterion = LabelWiseContrastiveLoss(
            temperature=args.contrastive_temperature,
            neg_samples=100
        )
    else:
        print("Contrastive learning disabled.")
        contrastive_criterion = None
    print("Initializing metrics...")
    
    # ---------- 生成要评估的 code_indices ----------
    if args.eval_codes_file:                 # 开关打开
        # 支持 .txt 一行一个 code，也支持 json list
        if args.eval_codes_file.endswith(".json"):
            import json, pathlib
            subset_codes = json.load(open(args.eval_codes_file))
        else:
            subset_codes = [l.strip() for l in open(args.eval_codes_file) if l.strip()]
    else:                                    # 默认：把训练集实际出现过的 code 拿出来
        subset_codes = list({str(c) for t in train_dataset.targets for c in t})

    if subset_codes:                         # 显式传给 MetricCollection
        code_indices = torch.tensor(
            [label_loader.code2idx[c] for c in subset_codes if c in label_loader.code2idx],
            dtype=torch.long
        )
    else:
        code_indices = None

    metrics = {
        "train": MetricCollection([LossMetric()]),          # 训练阶段通常不需要裁剪
        "val":   MetricCollection([
                    Precision(number_of_classes=label_loader.num_labels, average="macro"),
                    Precision(number_of_classes=label_loader.num_labels, average="micro"),
                    F1Score(number_of_classes=label_loader.num_labels, average="macro"),
                    F1Score(number_of_classes=label_loader.num_labels, average="micro"),
                    AUC(number_of_classes=label_loader.num_labels,average="macro"),
                    AUC(number_of_classes=label_loader.num_labels,average="micro"),
                    Precision_K(k=15),
                    Precision_K(k=10),
                    Precision_K(k=8),
                    Precision_K(k=5),
                    MeanAveragePrecision(),
                    LossMetric()
                ], code_indices),     # 只评估这 50 个 code
        "test":  MetricCollection([
                    Precision(number_of_classes=label_loader.num_labels, average="macro"),
                    Precision(number_of_classes=label_loader.num_labels, average="micro"),
                    F1Score(number_of_classes=label_loader.num_labels, average="macro"),
                    F1Score(number_of_classes=label_loader.num_labels, average="micro"),
                    AUC(number_of_classes=label_loader.num_labels,average="macro"),
                    AUC(number_of_classes=label_loader.num_labels,average="micro"),
                    Precision_K(k=15),
                    Precision_K(k=10),
                    Precision_K(k=8),
                    Precision_K(k=5),
                    MeanAveragePrecision(),
                    LossMetric()
                ], code_indices)
    }

    for mc in metrics.values():
        mc.set_threshold(args.threshold)
        mc.to(device)
    
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
        epochs=args.epochs,
        contrastive_criterion=contrastive_criterion,
        contrastive_loss_weight=args.contrastive_loss_weight,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
        best_metric_name=args.best_metric_name,
        use_amp=args.use_amp,
        use_wandb=args.use_wandb and (not args.use_ddp or rank == 0),
        use_ddp=args.use_ddp,
        rank=rank if args.use_ddp else 0,
        world_size=args.world_size if args.use_ddp else 1,
        train_sampler=train_sampler,
        resume_checkpoint=args.resume_checkpoint,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,

    )
    print("Training started")
    trainer.train()
    print("Training completed")
    
    if args.use_ddp:
        cleanup_ddp()

def main():
    args = parse_args()
    if not args.output_dir:
        # 为输出目录添加小时级时间戳，格式：YYYYMMDD_HH
        timestamp = datetime.now().strftime("%Y%m%d_%H")
        args.output_dir = os.path.join("checkpoints", timestamp)

    if args.use_ddp and torch.cuda.device_count() > 1:
        print(f"Starting DDP training on {args.world_size} GPUs")
        mp.spawn(main_worker, args=(args,), nprocs=args.world_size, join=True)
    else:
        print("Starting single GPU training")
        main_worker(0, args)

if __name__ == "__main__":
    main()
