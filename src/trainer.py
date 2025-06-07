
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from src.data_loader import TextLoader, LabelLoader, ICDMultiLabelDataset
from src.model import ClinicalLongformerLabelAttention
from src.metric import MetricCollection, Precision, Recall, F1Score, MeanAveragePrecision, AUC
from tqdm import tqdm

class Trainer:
    """
    负责管理优化器、学习率调度器、损失函数、训练/验证循环、多种度量和最佳模型保存。
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        metrics: MetricCollection,
        device: torch.device,
        epochs: int,
        gradient_accumulation_steps: int = 1,
        output_dir: str = "checkpoints",
        best_metric_name: str = "MeanAveragePrecision"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics
        self.device = device
        self.epochs = epochs
        self.grad_acc_steps = gradient_accumulation_steps
        self.output_dir = output_dir
        self.best_metric_name = best_metric_name
        os.makedirs(self.output_dir, exist_ok=True)
        self.best_metric = None
        self.best_epoch = -1

    def train(self):
        self.model.to(self.device)
        for epoch in range(1, self.epochs + 1):
            print(f"Epoch {epoch}/{self.epochs}")
            self.current_epoch = epoch
            self._train_epoch()
            self._validate_epoch("val")
        self._validate_epoch("test")
        print(f"Best validation {self.best_metric_name}: {self.best_metric:.4f} at epoch {self.best_epoch}")

    def _train_epoch(self):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {self.current_epoch}/{self.epochs}")
        for step, (x_batch, y_batch) in enumerate(pbar, 1):
            input_ids = x_batch['input_ids'].to(self.device)
            attention_mask = x_batch['attention_mask'].to(self.device)
            y_true = y_batch.to(self.device)

            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.criterion(logits, y_true) / self.grad_acc_steps
            loss.backward()

            if step % self.grad_acc_steps == 0 or step == len(self.train_loader):
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # 在进度条中实时显示训练损失
            pbar.set_postfix({"train loss": loss.item()})
            # 手动更新进度条
            if step % 1000 == 0:
                pbar.update(1000)

    def _validate_epoch(self,data_loader="val"):
        if data_loader == "val":
            loader = self.val_loader
        elif data_loader == "train":
            loader = self.train_loader
        else:
            loader = self.test_loader
        self.model.eval()
        self.metrics.reset_metrics()
        all_logits, all_targets = [], []
        pbar=tqdm(loader,desc=f"{data_loader}")
        with torch.no_grad():
            for step, (x_batch, y_batch) in enumerate(pbar, 1):
                input_ids = x_batch['input_ids'].to(self.device)
                attention_mask = x_batch['attention_mask'].to(self.device)
                y_true = y_batch.to(self.device)

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits=torch.sigmoid(logits)

                # 只更新分类度量，不更新 loss
                self.metrics.update({
                    'logits': logits,
                    'targets': y_true
                })
                all_logits.append(logits)
                all_targets.append(y_true)
                if step % 1000 == 0:
                    pbar.update(1000)
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        results = self.metrics.compute(logits=all_logits, targets=all_targets)
        print(f"{data_loader}:  ", {k: float(v) for k, v in results.items()})

        current = results.get(self.best_metric_name)
        if current is not None:
            current_val = float(current)
            if self.best_metric is None or current_val > self.best_metric:
                self.best_metric = current_val
                self.best_epoch = self.current_epoch
                save_path = os.path.join(self.output_dir, f"best_model_epoch{self.current_epoch}.pt")
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved best model to {save_path} ({self.best_metric_name}: {self.best_metric:.4f})")


print("aaa")