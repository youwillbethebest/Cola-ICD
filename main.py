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
# Modify import: add new BERT chunk model
from src.model import ClinicalLongformerLabelAttention, ClinicalBERTChunkAttention, ClinicalBERTChunkAttentionV2
from src.metric import MetricCollection, Precision, Recall, F1Score, MeanAveragePrecision, AUC, Precision_K, LossMetric
from src.trainer import Trainer
import wandb
from datetime import datetime
from src.module import JaccardWeightedSupConLoss, LabelWiseContrastiveLoss, PositiveOnlyContrastiveLoss
from src.loss import FocalLossWithLogits, HierarchyConsistencyLoss
from src.data_loader import build_hierarchy_adjs, build_hierarchy_edges

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_ddp(rank, world_size):
    """Initialize DDP environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Clean up DDP environment"""
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description="Trainer for ICD multi-label classification using label attention model")
    parser.add_argument("--train_file",default="data/mimiciv_icd9_train.feather", type=str, help="Path to training data (.feather)")
    parser.add_argument("--val_file",default="data/mimiciv_icd9_val.feather", type=str, help="Path to validation data (.feather)")
    parser.add_argument("--test_file",default="data/mimiciv_icd9_test.feather", type=str, help="Path to test data (.feather)")  
    parser.add_argument("--codes_file", type=str, default="data/filtered_icd_codes_with_desc.feather", help="Path to ICD codes and descriptions file")
    parser.add_argument("--synonyms_file", type=str, default="data/icd_synonyms.json", help="Path to ICD synonyms file")
    parser.add_argument("--abbreviations_file", type=str, default=None, help="Path to ICD abbreviations file")
    parser.add_argument("--use_abbreviations", action="store_true", default=False, help="Whether to use abbreviations and common expressions as separate terms")
    parser.add_argument("--pretrained_model_name", type=str, default="Clinical-Longformer", help="Pretrained model for text encoding")
    parser.add_argument("--label_model_name", type=str, default="Bio_ClinicalBERT", help="Pretrained model for label encoding")
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length for text")
    parser.add_argument("--label_max_length", type=int, default=256, help="Max sequence length for label descriptions")
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
    
    # Add early stopping parameters
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Number of epochs to wait before early stopping if no improvement")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0001, help="Minimum change required to qualify as an improvement")
    parser.add_argument("--early_stopping", action="store_true", default=False, help="Whether to use early stopping")
    
    parser.add_argument("--eval_codes_file", type=str, default=None,
                        help="txt/json list file; if provided, metrics are calculated only on these codes")
    
    # ---- Contrastive learning parameters ----
    parser.add_argument("--use_contrastive", action="store_true", default=False,
                        help="Whether to enable contrastive learning")
    parser.add_argument("--contrastive_loss_weight", type=float, default=0.1,
                        help="Weight for contrastive loss")
    parser.add_argument("--contrastive_temperature", type=float, default=0.1,
                        help="Temperature tau for contrastive learning (for LabelWiseContrastiveLoss)")
    parser.add_argument("--use_contrastive_positive_only", action="store_true", default=False,
                        help="Whether to enable positive-only contrastive learning")

    # ---- GNN and co-occurrence/hierarchy graph parameters ----
    parser.add_argument("--use_gnn", action="store_true", default=False,
                        help="Whether to enable GNN to update label embeddings")
    parser.add_argument("--adj_matrix_mode", type=str, default="ppmi", 
                        choices=["binary", "count", "ppmi", "hierarchy"],
                        help="Adjacency matrix construction mode")
    parser.add_argument("--adj_matrix_topk", type=int, default=20,
                        help="Top-k edges to keep per row for adjacency matrix (only for co-occurrence graph)")
    parser.add_argument("--adj_matrix_self_loop", action="store_true", default=True,
                        help="Whether to add self-loops to adjacency matrix (only for co-occurrence graph)")
    parser.add_argument("--adj_matrix_device", type=str, default="cpu",
                        help="Device used for building adjacency matrix")
    
    # ---- Hierarchy consistency loss ----
    parser.add_argument("--use_hierarchy_loss", action="store_true", default=False,
                        help="Whether to enable hierarchy consistency loss (parent->child)")
    parser.add_argument("--hierarchy_loss_weight", type=float, default=0.05,
                        help="Weight for hierarchy consistency loss")
    parser.add_argument("--hierarchy_margin", type=float, default=0.0,
                        help="Hinge margin for hierarchy consistency")
    
    parser.add_argument("--use_focal_loss", action="store_true", default=False,
                        help="Whether to use Focal Loss")
    parser.add_argument("--gamma", type=float, default=2.0, help="Gamma parameter for Focal Loss")
    parser.add_argument("--alpha", type=float, default=0.25, help="Alpha parameter for Focal Loss")
    
    # ---- New: BERT chunk related parameters ----
    parser.add_argument("--model_type", type=str, default="longformer", 
                        choices=["longformer", "bert_chunk", "bert_chunk_v2"],
                        help="Model type selection")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Chunk size for BERT chunk model")
    parser.add_argument("--overlap", type=int, default=32,
                        help="Overlap size for BERT chunk model (only for bert_chunk_v2)")
    parser.add_argument("--chunk_aggregation", type=str, default="max",
                        choices=["mean", "max", "sum", "weighted"],
                        help="Chunk aggregation method (only for bert_chunk_v2)")
    
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
    """DDP training worker process"""
    if args.use_ddp:
        setup_ddp(rank, args.world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device(args.device)
    
    # W&B initialization (only on rank 0)
    if args.use_wandb and (not args.use_ddp or rank == 0):
        now = datetime.now().strftime("%m-%d_%H-%M")
        wandb.init(project="Attentionicd", name=f"Attentionicd_{now}")
        wandb.config.update(vars(args), allow_val_change=True)
    else:
        print("W&B logging disabled.")
    device = torch.device(args.device)
    print(f"Use AMP: {args.use_amp}")
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")

    # Data loading and model initialization
    print("Loading text tokenizer...")
    text_loader = TextLoader(pretrained_model_name=args.pretrained_model_name, max_length=args.max_length)
    print("Loading label tokenizer and model...")
    label_loader = SynonymLabelLoader(codes_file=args.codes_file, synonyms_file=args.synonyms_file, abbreviations_file=args.abbreviations_file, pretrained_model_name=args.label_model_name, max_length=args.label_max_length,term_count=args.term_count,sort_method='max', use_abbreviations=args.use_abbreviations)
    print(f"Number of labels: {label_loader.num_labels}")

    print("Creating training dataset...")
    train_dataset = ICDMultiLabelDataset(data_file=args.train_file, text_loader=text_loader, label_loader=label_loader)
    print("Creating validation dataset...")
    val_dataset = ICDMultiLabelDataset(data_file=args.val_file, text_loader=text_loader, label_loader=label_loader)
    test_dataset = ICDMultiLabelDataset(data_file=args.test_file, text_loader=text_loader, label_loader=label_loader)
    
    # Create distributed samplers
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
    
    # Build adjacency matrix (if GNN enabled)
    adj_matrix = None
    hierarchy_edges = None
    if args.use_gnn:
        print("Building adjacency matrix for GNN...")
        if args.adj_matrix_mode == "hierarchy":
            # Hierarchy graph: bidirectional adjacency for message passing; do not copy synonym nodes
            adj_matrix = build_hierarchy_adjs(
                code2idx=label_loader.code2idx,
                device=args.adj_matrix_device
            )
            # Directed parent->child edges: for hierarchy consistency loss
            hierarchy_edges = build_hierarchy_edges(
                code2idx=label_loader.code2idx,
                direction="parent_to_child"
            )
            print("Hierarchy adjacency built (up/down).")
        else:
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
    
    # Create different models based on model type
    if args.model_type == "longformer":
        print("Using ClinicalLongformerLabelAttention model...")
        model = ClinicalLongformerLabelAttention(
            longformer_path=args.pretrained_model_name,
            term_counts=args.term_count,
            label_loader=label_loader,
            use_gnn=args.use_gnn,
            adj_matrix=adj_matrix
        )
    elif args.model_type == "bert_chunk":
        print(f"Using ClinicalBERTChunkAttention model with chunk_size={args.chunk_size}...")
        model = ClinicalBERTChunkAttention(
            bert_model_path=args.pretrained_model_name,
            chunk_size=args.chunk_size,
            term_counts=args.term_count,
            label_loader=label_loader,
            use_gnn=args.use_gnn,
            adj_matrix=adj_matrix
        )
    elif args.model_type == "bert_chunk_v2":
        print(f"Using ClinicalBERTChunkAttentionV2 model with chunk_size={args.chunk_size}, overlap={args.overlap}, aggregation={args.chunk_aggregation}...")
        model = ClinicalBERTChunkAttentionV2(
            bert_model_path=args.pretrained_model_name,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            term_counts=args.term_count,
            label_loader=label_loader,
            use_gnn=args.use_gnn,
            adj_matrix=adj_matrix,
            chunk_aggregation=args.chunk_aggregation
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    model.to(device)
    
    # Wrap as DDP model
    if args.use_ddp:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    # W&B watch (only on rank 0)
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

    criterion = FocalLossWithLogits(gamma=args.gamma, alpha=args.alpha) if args.use_focal_loss else nn.BCEWithLogitsLoss()
    if args.use_contrastive:
        print("Contrastive learning enabled (Label-Wise).")
        contrastive_criterion = LabelWiseContrastiveLoss(
            temperature=args.contrastive_temperature,
            neg_samples=50
        )
    elif args.use_contrastive_positive_only:
        print("Contrastive learning enabled (Positive-Only).")
        contrastive_criterion = PositiveOnlyContrastiveLoss(
            temperature=args.contrastive_temperature
        )
    else:
        print("Contrastive learning disabled.")
        contrastive_criterion = None
    hierarchy_criterion = None
    if args.use_hierarchy_loss and args.adj_matrix_mode == "hierarchy":
        hierarchy_criterion = HierarchyConsistencyLoss(margin=args.hierarchy_margin)
    print("Initializing metrics...")
    
    # ---------- Generate code_indices to evaluate ----------
    if args.eval_codes_file:                 # Switch on
        # Support .txt one code per line, also json list
        if args.eval_codes_file.endswith(".json"):
            import json, pathlib
            subset_codes = json.load(open(args.eval_codes_file))
        else:
            subset_codes = [l.strip() for l in open(args.eval_codes_file) if l.strip()]
    else:                                    # Default: extract codes that actually appeared in training set
        subset_codes = list({str(c) for t in train_dataset.targets for c in t})

    if subset_codes:                         # Explicitly pass to MetricCollection
        code_indices = torch.tensor(
            [label_loader.code2idx[c] for c in subset_codes if c in label_loader.code2idx],
            dtype=torch.long
        )
    else:
        code_indices = None

    metrics = {
        "train": MetricCollection([LossMetric()]),          # No cropping usually needed for training phase
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
                ], code_indices),     # Only evaluate these codes
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
        save_artifacts=False,  # Disable wandb artifacts saving
        use_ddp=args.use_ddp,
        rank=rank if args.use_ddp else 0,
        world_size=args.world_size if args.use_ddp else 1,
        train_sampler=train_sampler,
        resume_checkpoint=args.resume_checkpoint,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        hierarchy_criterion=hierarchy_criterion,
        hierarchy_edges=hierarchy_edges,
        hierarchy_loss_weight=args.hierarchy_loss_weight,
    )
    print("Training started")
    trainer.train()
    print("Training completed")
    
    if args.use_ddp:
        cleanup_ddp()

def main():
    args = parse_args()
    if not args.output_dir:
        # Add hourly timestamp to output directory, format: YYYYMMDD_HH
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        args.output_dir = os.path.join("checkpoints", timestamp)

    if args.use_ddp and torch.cuda.device_count() > 1:
        print(f"Starting DDP training on {args.world_size} GPUs")
        mp.spawn(main_worker, args=(args,), nprocs=args.world_size, join=True)
    else:
        print("Starting single GPU training")
        main_worker(0, args)

if __name__ == "__main__":
    main()
