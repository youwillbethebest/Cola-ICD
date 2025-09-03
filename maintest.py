import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from src.data_loader import TextLoader, LabelLoader, ICDMultiLabelDataset
from src.model import ClinicalLongformerLabelAttention
from src.metric import MetricCollection, Precision, Recall, F1Score, MeanAveragePrecision, AUC, Precision_K, LossMetric
from src.trainer import Trainer
import wandb
from datetime import datetime
import argparse

def setup_ddp(rank, world_size):
    """初始化DDP环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """清理DDP环境"""
    dist.destroy_process_group()

def create_metrics(label_loader):
    """创建metrics配置"""
    return {
        "train": MetricCollection([LossMetric()]),
        "val": MetricCollection([
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
        ]),
        "test": MetricCollection([
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
    }

def train_single_gpu():
    """单GPU/CPU训练函数"""
    # 配置
    train_file = "data/mimiciv_icd9_train.feather"
    val_file = "data/mimiciv_icd9_val.feather"
    test_file = "data/mimiciv_icd9_test.feather"
    codes_file = "data/filtered_icd_codes_with_desc.feather"
    pretrained_model_name = "Clinical-Longformer"
    label_model_name = "Bio_ClinicalBERT"
    max_length = 1024
    label_max_length = 128
    batch_size = 8
    epochs = 2
    lr = 2e-5
    weight_decay = 0.0
    warmup_steps = 0
    gradient_accumulation_steps = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = True
    use_wandb = True
    output_dir = "checkpoints_test"
    best_metric_name = "map"

    os.makedirs(output_dir, exist_ok=True)
    
    if use_wandb:
        now = datetime.now().strftime("%m-%d_%H-%M")
        wandb.init(project="Attentionicd", name=f"Attentionicd_Single_{now}")
    else:
        print("W&B logging disabled.")
    
    print("Initializing loaders...")
    text_loader = TextLoader(pretrained_model_name=pretrained_model_name, max_length=max_length)
    label_loader = LabelLoader(codes_file=codes_file, pretrained_model_name=label_model_name, max_length=label_max_length)

    print("Loading datasets and sampling first 200 examples...")
    train_full = ICDMultiLabelDataset(data_file=train_file, text_loader=text_loader, label_loader=label_loader)
    val_full = ICDMultiLabelDataset(data_file=val_file, text_loader=text_loader, label_loader=label_loader)
    test_full = ICDMultiLabelDataset(data_file=test_file, text_loader=text_loader, label_loader=label_loader)

    sample_size = 200
    train_dataset = Subset(train_full, range(min(sample_size, len(train_full))))
    val_dataset = Subset(val_full, range(min(sample_size, len(val_full))))
    test_dataset = Subset(test_full, range(min(sample_size, len(test_full))))
    
    print(f"Sampled {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test examples")

    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True
    )

    print("Initializing model...")
    model = ClinicalLongformerLabelAttention(
        longformer_path=pretrained_model_name,
        codes_file=codes_file,
        label_model_name=label_model_name
    )
    
    model = model.to(device)

    print("Setting up optimizer and scheduler...")
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = nn.BCEWithLogitsLoss()

    print("Initializing metrics...")
    metrics = create_metrics(label_loader)

    # 将所有metrics移到对应设备
    for mc in metrics.values():
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
        epochs=epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        output_dir=output_dir,
        best_metric_name=best_metric_name,
        use_amp=use_amp,
        use_wandb=use_wandb,
        save_artifacts=False,  # 禁用wandb artifacts保存
        use_ddp=False,         # 不使用DDP
        rank=0,                # 单GPU时rank固定为0
        world_size=1,          # 单GPU时world_size为1
        train_sampler=None     # 单GPU时不需要sampler
    )

    print("Training started")
    trainer.train()
    print("Training completed")

def train_ddp_worker(rank, world_size):
    """DDP训练工作函数"""
    # 设置DDP
    setup_ddp(rank, world_size)
    
    # 配置
    train_file = "data/mimiciv_icd9_train.feather"
    val_file = "data/mimiciv_icd9_val.feather"
    test_file = "data/mimiciv_icd9_test.feather"
    codes_file = "data/filtered_icd_codes_with_desc.feather"
    pretrained_model_name = "Clinical-Longformer"
    label_model_name = "Bio_ClinicalBERT"
    max_length = 1024
    label_max_length = 128
    batch_size = 8
    epochs = 2
    lr = 2e-5
    weight_decay = 0.0
    warmup_steps = 0
    gradient_accumulation_steps = 1
    device = torch.device(f"cuda:{rank}")
    use_amp = True
    use_wandb = True
    output_dir = "checkpoints_test"
    best_metric_name = "map"

    os.makedirs(output_dir, exist_ok=True)
    
    # 只在rank 0初始化wandb
    if use_wandb and rank == 0:
        now = datetime.now().strftime("%m-%d_%H-%M")
        wandb.init(project="Attentionicd", name=f"Attentionicd_DDP_{now}")
    elif rank == 0:
        print("W&B logging disabled.")
    
    if rank == 0:
        print("Initializing loaders...")
    text_loader = TextLoader(pretrained_model_name=pretrained_model_name, max_length=max_length)
    label_loader = LabelLoader(codes_file=codes_file, pretrained_model_name=label_model_name, max_length=label_max_length)

    if rank == 0:
        print("Loading datasets and sampling first 200 examples...")
    train_full = ICDMultiLabelDataset(data_file=train_file, text_loader=text_loader, label_loader=label_loader)
    val_full = ICDMultiLabelDataset(data_file=val_file, text_loader=text_loader, label_loader=label_loader)
    test_full = ICDMultiLabelDataset(data_file=test_file, text_loader=text_loader, label_loader=label_loader)

    sample_size = 200
    train_dataset = Subset(train_full, range(min(sample_size, len(train_full))))
    val_dataset = Subset(val_full, range(min(sample_size, len(val_full))))
    test_dataset = Subset(test_full, range(min(sample_size, len(test_full))))
    
    if rank == 0:
        print(f"Sampled {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test examples")

    # 创建分布式sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    if rank == 0:
        print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        sampler=val_sampler,
        num_workers=4, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        sampler=test_sampler,
        num_workers=4, 
        pin_memory=True
    )

    if rank == 0:
        print("Initializing model...")
    model = ClinicalLongformerLabelAttention(
        longformer_path=pretrained_model_name,
        codes_file=codes_file,
        label_model_name=label_model_name,
        term_counts=3
    )
    
    # 将模型移到对应的GPU并包装为DDP
    model = model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    if rank == 0:
        print("Setting up optimizer and scheduler...")
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = nn.BCEWithLogitsLoss()

    if rank == 0:
        print("Initializing metrics...")
    metrics = create_metrics(label_loader)

    # 将所有metrics移到GPU
    for mc in metrics.values():
        mc.to(device)

    if rank == 0:
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
        use_wandb=use_wandb,
        save_artifacts=False,  # 禁用wandb artifacts保存
        use_ddp=True,          # 启用DDP
        rank=rank,             # 当前进程rank
        world_size=world_size, # 总进程数
        train_sampler=train_sampler  # 传入训练sampler用于epoch同步
    )

    if rank == 0:
        print("Training started")
    trainer.train()
    if rank == 0:
        print("Training completed")
    
    # 清理DDP环境
    cleanup_ddp()

def main():
    """主函数，处理命令行参数并选择训练模式"""
    parser = argparse.ArgumentParser(description="训练脚本，支持单GPU和DDP模式")
    parser.add_argument('--use_ddp', action='store_true', help='是否使用DDP分布式训练')
    parser.add_argument('--world_size', type=int, default=2, help='DDP模式下使用的GPU数量')
    args = parser.parse_args()
    
    if args.use_ddp:
        # DDP模式
        world_size = args.world_size
        
        # 检查GPU数量
        if world_size > torch.cuda.device_count():
            print(f"Warning: world_size ({world_size}) > available GPUs ({torch.cuda.device_count()})")
            world_size = torch.cuda.device_count()
        
        if world_size < 2:
            print("DDP mode requires at least 2 GPUs. Falling back to single GPU mode.")
            train_single_gpu()
        else:
            print(f"Starting DDP training with {world_size} GPUs")
            # 启动多进程DDP训练
            mp.spawn(train_ddp_worker, args=(world_size,), nprocs=world_size, join=True)
    else:
        # 单GPU/CPU模式
        print("Starting single GPU/CPU training")
        train_single_gpu()

if __name__ == "__main__":
    main()