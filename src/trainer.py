import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.metric import MetricCollection
from tqdm import tqdm
from typing import Dict
from torch.amp import autocast
from torch.amp import GradScaler
import wandb

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
        metrics: Dict[str, MetricCollection],
        device: torch.device,
        epochs: int,
        gradient_accumulation_steps: int = 1,
        output_dir: str = "checkpoints",
        best_metric_name: str = "map",
        use_amp: bool = False,
        use_wandb: bool = False
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
        self.use_amp = use_amp
        self.use_wandb = use_wandb
        if self.use_amp:
            self.scaler = GradScaler("cuda")
        else:
            self.scaler = None

    def train(self):
        self.model.to(self.device)
        for epoch in range(1, self.epochs + 1):
            print(f"Epoch {epoch}/{self.epochs}")
            self.current_epoch = epoch
            self._train_epoch()
            self._validate_epoch("val")
        self.on_train_end()
        self.test_begin("best_model.pt")
        self._validate_epoch("test")
        print(f"Best validation {self.best_metric_name}: {self.best_metric:.4f} at epoch {self.best_epoch}")

    def _train_epoch(self):
        self.model.train()
        pbar = tqdm(total=len(self.train_loader), desc=f"Training Epoch {self.current_epoch}/{self.epochs}")
        # 重置训练指标收集
        self.metrics["train"].reset_metrics()
        for step, (x_batch, y_batch) in enumerate(self.train_loader, 1):
            # 前向和 loss 计算
            outputs = self.training_step(x_batch, y_batch)
            loss = outputs['loss']  # 直接使用原始 loss，不进行缩放
            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            # 优化器和调度器更新
            if step % self.grad_acc_steps == 0 or step == len(self.train_loader):
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            # 更新指标（使用原始 loss）
            self.update_metrics(outputs,loader_name="train")
            # 在进度条中显示训练损失（原始 loss）
            
            # 手动更新进度条
            if step % 500 == 0:
                pbar.set_postfix({"train loss": float(outputs['loss'].item())})
                pbar.update(500)
        # 训练 epoch 结束，打印训练集指标
        self.on_train_epoch_end()

    def _validate_epoch(self, data_loader="val"):
        if data_loader == "val":
            loader = self.val_loader
        elif data_loader == "train":
            loader = self.train_loader
        else:
            loader = self.test_loader
        self.model.eval()
        self.metrics[data_loader].reset_metrics()
        all_logits, all_targets = [], []
        pbar=tqdm(total=len(loader),desc=f"{data_loader}")
        with torch.no_grad():
            for step, (x_batch, y_batch) in enumerate(loader, 1):
                # 执行验证前向并更新指标
                outputs = self.validation_step(x_batch, y_batch)
                self.update_metrics(outputs,loader_name=data_loader)
                all_logits.append(outputs['logits'])
                all_targets.append(outputs['targets'])
                if step % 500 == 0:
                    pbar.update(500)
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        self.on_val_end(all_logits, all_targets, loader_name=data_loader)
        # 仅在 val 阶段保存最佳模型
        if data_loader == "val":
            best_tensor = self.metrics[data_loader].get_best_metric(self.best_metric_name)
            if best_tensor is not None:
                best_val = float(best_tensor)
                if self.best_metric is None or best_val > self.best_metric:
                    self.best_metric = best_val
                    self.best_epoch = self.current_epoch
                    save_path = os.path.join(self.output_dir, "best_model.pt")
                    self.save_checkpoint(save_path)
                    print(f"Saved best model to {save_path} ({self.best_metric_name}: {self.best_metric:.4f})")
        # 统一交给回调：计算/打印/重置
        

    def save_checkpoint(self, save_path: str):
        """
        保存包含模型、优化器、调度器和AMP scaler的完整checkpoint。
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        if self.use_amp and self.scaler is not None:
            checkpoint['scaler'] = self.scaler.state_dict()
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")

    def on_train_end(self):
        """
        训练结束时调用，保存最终模型
        """
        save_path = os.path.join(self.output_dir, f"final_model.pt")
        self.save_checkpoint(save_path)
        print(f"Saved final model to {save_path}")

    def on_train_epoch_end(self):
        """
        每个训练 epoch 结束时调用，计算并打印训练集上的所有 batch_update 指标
        """
        train_results = self.metrics["train"].compute()
        # 使用统一 log_dict 方法记录训练指标
        self.log_dict({'train': train_results})
        # 重置以便下一阶段使用
        self.metrics["train"].reset_metrics()

    def test_begin(self,file_name:str)-> None:
        checkpoint_path = os.path.join(self.output_dir, file_name)
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        if self.use_amp and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler'])
        print(f"Loaded model from {checkpoint_path}")

    def training_step(self, x_batch, y_batch):
        """
        单个 batch 的前向计算并返回包含 loss、logits、targets 的字典
        """
        input_ids = x_batch['input_ids'].to(self.device)
        attention_mask = x_batch['attention_mask'].to(self.device)
        y_true = y_batch.to(self.device)
        if self.use_amp:
            with autocast('cuda'):
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(logits, y_true)  
        else:
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.criterion(logits, y_true) 
        logits = torch.sigmoid(logits) 
        return {'loss': loss, 'logits': logits, 'targets': y_true}

    def validation_step(self, x_batch, y_batch):
        """
        单个 batch 的验证前向计算并返回包含 logits（已 sigmoid）和 targets 的字典
        """
        input_ids = x_batch['input_ids'].to(self.device)
        attention_mask = x_batch['attention_mask'].to(self.device)
        y_true = y_batch.to(self.device)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # 计算验证集的 loss
        loss = self.criterion(logits, y_true)
        probs = torch.sigmoid(logits)
        return {'loss': loss, 'logits': probs, 'targets': y_true}

    def update_metrics(self, outputs: dict,loader_name:str):
        """
        将 outputs 中的 loss、logits、targets 传给 metrics 集合
        """
        # 直接使用原始 loss，不需要恢复
        if 'loss' in outputs:
            outputs['loss'] = outputs['loss'].detach()
        self.metrics[loader_name].update(outputs)

    def log_dict(self, nested_metrics: dict):
        """
        打印并记录包含 'train'/'val' 子字典的嵌套指标
        """
        logs = {}
        for phase, m in nested_metrics.items():
            metrics_items = {k: float(v) for k, v in m.items()}
            # 打印到控制台
            print(f"{phase} metrics:", metrics_items)
            # 汇总代码到 logs
            logs.update({f"{phase}/{k}": v for k, v in metrics_items.items()})
        # 根据开关同步到 wandb
        if self.use_wandb:
            wandb.log(logs, step=self.current_epoch)

    def on_val_end(self, all_logits, all_targets, loader_name="val"):
        """
        评估结束时调用，计算完整 logits/targets 并打印 loader_name 指标，然后重置
        """
        results = self.metrics[loader_name].compute(logits=all_logits.cpu(), targets=all_targets.cpu())
        # 使用统一 log_dict 方法记录验证指标
        self.log_dict({loader_name: results})
        self.metrics[loader_name].reset_metrics()
