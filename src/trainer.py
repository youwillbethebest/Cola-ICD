import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.metric import MetricCollection
from tqdm import tqdm
from typing import Dict
import json
from torch.amp import autocast
from torch.amp import GradScaler
import wandb
import torch.distributed as dist
import pandas as pd  # 新增，用于保存 feather 文件
import numpy as np

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
        contrastive_criterion: nn.Module = None,
        contrastive_loss_weight: float = 0.0,
        gradient_accumulation_steps: int = 1,
        output_dir: str = "checkpoints",
        best_metric_name: str = "map",
        use_amp: bool = False,
        use_wandb: bool = False,
        save_artifacts: bool = False,
        use_ddp: bool = False,
        rank: int = 0,
        world_size: int = 1,
        train_sampler = None,
        resume_checkpoint: str = None,
        early_stopping: bool = False,
        early_stopping_patience: int = 3,
        early_stopping_min_delta: float = 0.001,
        hierarchy_criterion: nn.Module = None,
        hierarchy_edges: torch.LongTensor = None,
        hierarchy_loss_weight: float = 0.0,
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
        self.contrastive_criterion = contrastive_criterion
        self.contrastive_loss_weight = contrastive_loss_weight
        self.grad_acc_steps = gradient_accumulation_steps
        self.output_dir = output_dir
        self.best_metric_name = best_metric_name
        os.makedirs(self.output_dir, exist_ok=True)
        self.best_metric = None
        self.best_epoch = -1
        self.use_amp = use_amp
        self.use_wandb = use_wandb
        self.save_artifacts = save_artifacts
        self.use_ddp = use_ddp
        self.rank = rank
        self.world_size = world_size
        self.train_sampler = train_sampler
        self.resume_checkpoint = resume_checkpoint
        self.start_epoch = 1
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_counter = 0
        self.early_stopping_triggered = False
        self.prev_best = None

        self.best_db = 0.5
        self.current_epoch = 0  # 初始化 current_epoch 属性


        if self.use_amp:
            self.scaler = GradScaler("cuda")
        else:
            self.scaler = None

        # 层级一致性损失
        self.hierarchy_criterion = hierarchy_criterion
        self.hierarchy_edges = hierarchy_edges  # CPU/任意设备；前向时迁移
        self.hierarchy_loss_weight = hierarchy_loss_weight

    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点以继续训练
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if self.use_ddp:
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        if self.use_amp and self.scaler is not None and 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_db = checkpoint.get('db', 0.5)  # 如果检查点中没有'db'键，使用默认值0.5
        
        
        print(f"Checkpoint loaded successfully. Resuming from epoch {self.start_epoch}")
        return self.start_epoch

    def train(self):
        self.model.to(self.device)
        
        if self.resume_checkpoint:
            self.load_checkpoint(self.resume_checkpoint)
        
        for epoch in range(self.start_epoch, self.epochs + 1):
            print(f"Epoch {epoch}/{self.epochs}")
            self.current_epoch = epoch
            self._train_epoch()
            self._validate_epoch("val")
            
            # 检查早停条件
            if self.early_stopping and self._should_early_stop():
                break
                
        if self.epochs > 0:
            self.on_train_end()
        
        print("\n=== 最终模型评估 ===")
        self.test_begin("best_model.pt")
        self._validate_epoch("val", evaluating_best_model=True)
        self._validate_epoch("test")
        self.on_test_end()

    def _train_epoch(self):
        self.model.train()
        
        if self.use_ddp and self.train_sampler is not None:
            self.train_sampler.set_epoch(self.current_epoch)
        
        if self.rank == 0:
            pbar = tqdm(total=len(self.train_loader), desc=f"Training Epoch {self.current_epoch}/{self.epochs}")
        
        self.metrics["train"].reset_metrics()
        for step, (x_batch, y_batch) in enumerate(self.train_loader, 1):
            outputs = self.training_step(x_batch, y_batch)
            loss = outputs['loss']
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            if step % self.grad_acc_steps == 0 or step == len(self.train_loader):
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            self.update_metrics(outputs,loader_name="train")
            
            if step % 500 == 0 and self.rank == 0:
                pbar.set_postfix({"train loss": float(outputs['loss'].item())})
                pbar.update(500)
        self.on_train_epoch_end()

    def _validate_epoch(self, data_loader="val", evaluating_best_model=False):
        if data_loader == "val":
            loader = self.val_loader
        elif data_loader == "train":
            loader = self.train_loader
        else:
            loader = self.test_loader
        self.model.eval()
        self.metrics[data_loader].reset_metrics()
        all_logits, all_targets = [], []
        pbar = tqdm(total=len(loader), desc=f"{data_loader}")
        with torch.no_grad():
            for step, (x_batch, y_batch) in enumerate(loader, 1):
                outputs = self.validation_step(x_batch, y_batch)
                self.update_metrics(outputs, loader_name=data_loader)
                all_logits.append(outputs['logits'])
                all_targets.append(outputs['targets'])
                if step % 500 == 0:
                    pbar.update(500)
        
        local_logits = torch.cat(all_logits, dim=0)
        local_targets = torch.cat(all_targets, dim=0)
        if self.use_ddp:
            all_logits = self._gather_all(local_logits)
            all_targets = self._gather_all(local_targets)
            self.metrics[data_loader].sync_counters()
        else:
            all_logits = local_logits
            all_targets = local_targets

        # 仅在主进程做完整指标计算／打印／保存
        if self.use_ddp and self.rank != 0:
            return
        self.on_val_end(all_logits.cpu(), all_targets.cpu(), loader_name=data_loader,evaluating_best_model=evaluating_best_model)
        # 仅在 val 阶段保存最佳模型
        if data_loader == "val" and not evaluating_best_model:
            best_tensor = self.metrics[data_loader].get_best_metric(self.best_metric_name)
            if best_tensor is not None:
                best_val = float(best_tensor)
                if self.best_metric is None or best_val > self.best_metric:
                    self.best_metric = best_val
                    self.best_epoch = self.current_epoch
                    save_path = os.path.join(self.output_dir, "best_model.pt")
                    self.save_checkpoint(save_path)
                    print(f"Saved best model to {save_path} ({self.best_metric_name}: {self.best_metric:.4f})")
        # 在测试集上保存预测结果
        if data_loader == "test" and (not self.use_ddp or self.rank == 0):
            # 保存二值化预测到 feather 文件
            self._save_predictions(all_logits, loader.dataset)
            # 额外导出注意力贡献（raw_text、pred_codes、true_codes、每标签 top-N token 贡献）
            try:
                self._export_attention_contributions(loader, top_labels=5, top_tokens=10,
                                                     save_name="test_attention_contrib.jsonl")
            except Exception as e:
                print(f"[WARN] 导出注意力贡献失败: {e}")

    def save_checkpoint(self, save_path: str):
        """
        保存包含模型、优化器、调度器和AMP scaler的完整checkpoint。
        """
        if self.rank == 0:
            if self.use_ddp:
                model_state_dict = self.model.module.state_dict()
            else:
                model_state_dict = self.model.state_dict()
            
            checkpoint = {
                'epoch': self.current_epoch,
                'model': model_state_dict,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'db':self.best_db
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
        if self.use_wandb and self.save_artifacts and (not self.use_ddp or self.rank == 0):
            best_path = os.path.join(self.output_dir, f"best_model.pt")
            best_artifact = wandb.Artifact(
                name="Attentionicd-best-model",      # 在 WandB 上的 artifact 名称
                type="model",                        # artifact 类型
                description="best checkpoint"
            )
            best_artifact.add_file(best_path)
            wandb.log_artifact(best_artifact)
    
    def on_test_end(self):
        save_path = os.path.join(self.output_dir, f"best_model_tuned.pt")
        self.save_checkpoint(save_path)
        print(f"Saved best model tuned to {save_path}")

    def on_train_epoch_end(self):
        """
        每个训练 epoch 结束时调用，计算并打印训练集上的所有 batch_update 指标
        """
        if self.use_ddp:
            self.metrics["train"].sync_counters()
        train_results = self.metrics["train"].compute()
        if not self.use_ddp or self.rank == 0:
            self.log_dict({'train': train_results})
        self.metrics["train"].reset_metrics()

    def test_begin(self,file_name:str)-> None:
        checkpoint_path = os.path.join(self.output_dir, file_name)
        checkpoint = torch.load(checkpoint_path)
        
        if self.use_ddp:
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_db = checkpoint.get('db', 0.5)  # 如果检查点中没有'db'键，使用默认值0.5
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
                logits, per_label_text_feat, label_proto = self.model(input_ids=input_ids, attention_mask=attention_mask)
                classification_loss = self.criterion(logits, y_true)  
                if self.contrastive_criterion is not None and self.contrastive_loss_weight > 0:
                    self.loss_cl = self.contrastive_criterion(per_label_text_feat, label_proto,y_true)
                    loss = classification_loss + self.contrastive_loss_weight * self.loss_cl
                    if self.rank == 0 and self.use_wandb:
                        wandb.log({"train/contrastive_loss": self.loss_cl.item()})
                else:
                    loss = classification_loss
                # 层级一致性损失（在 logits 上，sigmoid 前）
                if self.hierarchy_criterion is not None and self.hierarchy_edges is not None and self.hierarchy_loss_weight > 0:
                    hier_edges = self.hierarchy_edges.to(self.device)
                    self.loss_hier = self.hierarchy_criterion(logits, hier_edges)
                    loss = loss + self.hierarchy_loss_weight * self.loss_hier
                    if self.rank == 0 and self.use_wandb:
                        wandb.log({"train/hierarchy_loss": self.loss_hier.item()})
        else:
            logits, per_label_text_feat, label_proto = self.model(input_ids=input_ids, attention_mask=attention_mask)
            classification_loss = self.criterion(logits, y_true)  
            if self.contrastive_criterion is not None and self.contrastive_loss_weight > 0:
                    self.loss_cl = self.contrastive_criterion(per_label_text_feat, label_proto,y_true)
                    loss = classification_loss + self.contrastive_loss_weight * self.loss_cl
                    if self.rank == 0 and self.use_wandb:
                        wandb.log({"train/contrastive_loss": self.loss_cl.item()})
            else:
                    loss = classification_loss
            if self.hierarchy_criterion is not None and self.hierarchy_edges is not None and self.hierarchy_loss_weight > 0:
                    hier_edges = self.hierarchy_edges.to(self.device)
                    self.loss_hier = self.hierarchy_criterion(logits, hier_edges)
                    loss = loss + self.hierarchy_loss_weight * self.loss_hier
                    if self.rank == 0 and self.use_wandb:
                        wandb.log({"train/hierarchy_loss": self.loss_hier.item()})
        logits = torch.sigmoid(logits) 
        return {'loss': loss, 'logits': logits, 'targets': y_true}

    def validation_step(self, x_batch, y_batch):
        """
        单个 batch 的验证前向计算并返回包含 logits（已 sigmoid）和 targets 的字典
        """
        input_ids = x_batch['input_ids'].to(self.device)
        attention_mask = x_batch['attention_mask'].to(self.device)
        y_true = y_batch.to(self.device)
        logits, per_label_text_feat, label_proto = self.model(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.criterion(logits, y_true)
        probs = torch.sigmoid(logits)
        return {'loss': loss, 'logits': probs, 'targets': y_true}

    def update_metrics(self, outputs: dict,loader_name:str):
        """
        将 outputs 中的 loss、logits、targets 传给 metrics 集合
        """
        if 'loss' in outputs:
            outputs['loss'] = outputs['loss'].detach()
        self.metrics[loader_name].update(outputs)

    def log_dict(self, nested_metrics: dict):
        """
        打印并记录包含 'train'/'val' 子字典的嵌套指标
        """
        if self.rank == 0:
            logs = {}
            for phase, m in nested_metrics.items():
                metrics_items = {k: float(v) for k, v in m.items()}
                print(f"{phase} metrics:", metrics_items)
                logs.update({f"{phase}/{k}": v for k, v in metrics_items.items()})
            if self.use_wandb:
                wandb.log(logs)

    def on_val_end(self, all_logits, all_targets, loader_name="val", evaluating_best_model=False):
        """
        评估结束时调用，计算完整 logits/targets 并打印 loader_name 指标，然后重置
        """
        results = self.metrics[loader_name].compute(logits=all_logits.cpu(), targets=all_targets.cpu())
        if  evaluating_best_model and loader_name == "val":
            best_f1, best_db = self.f1_score_db_tuning(all_logits, all_targets)
            self.best_db = best_db
            print(f"Best F1: {best_f1:.4f}")
            print(f"Best DB: {best_db:.4f}")
            self.metrics["test"].set_threshold(best_db)
        # 使用统一 log_dict 方法记录验证指标
        self.log_dict({loader_name: results})
        self.metrics[loader_name].reset_metrics()

    def _gather_all(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        DDP环境下跨进程收集tensor并拼接，返回全量tensor
        """
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor)
        return torch.cat(tensor_list, dim=0)

    def _should_early_stop(self) -> bool:
        """
        检查是否应该触发早停
        """
        if not self.early_stopping:
            return False
            
        if self.use_ddp and self.rank != 0:
            return False
            
        best_metric = self.metrics["val"].get_best_metric(self.best_metric_name)
        if best_metric is None:
            return False
            
        best_val = float(best_metric)
        
        if self.prev_best is None or best_val > (self.prev_best + self.early_stopping_min_delta):
            self.early_stopping_counter = 0
            self.prev_best = best_val
            return False
        else:
            self.early_stopping_counter += 1
            
            if self.early_stopping_counter >= self.early_stopping_patience:
                self.early_stopping_triggered = True
                print(f"Early stopping: {self.early_stopping_counter} epochs without improvement for {self.best_metric_name}")
                return True
                
        return False
    
    def f1_score_db_tuning(self,logits, targets, average="micro", type="single"):
        if average not in ["micro", "macro"]:
            raise ValueError("Average must be either 'micro' or 'macro'")
        dbs = torch.linspace(0, 1, 100)
        tp = torch.zeros((len(dbs), targets.shape[1]))
        fp = torch.zeros((len(dbs), targets.shape[1]))
        fn = torch.zeros((len(dbs), targets.shape[1]))
        for idx, db in enumerate(dbs):
            predictions = (logits > db).long()
            tp[idx] = torch.sum((predictions) * (targets), dim=0)
            fp[idx] = torch.sum(predictions * (1 - targets), dim=0)
            fn[idx] = torch.sum((1 - predictions) * targets, dim=0)
        if average == "micro":
            f1_scores = tp.sum(1) / (tp.sum(1) + 0.5 * (fp.sum(1) + fn.sum(1)) + 1e-10)
        else:
            f1_scores = torch.mean(tp / (tp + 0.5 * (fp + fn) + 1e-10), dim=1)
        if type == "single":
            best_f1 = f1_scores.max()
            best_db = dbs[f1_scores.argmax()]
            print(f"Best F1: {best_f1:.4f} at DB: {best_db:.4f}")
            return best_f1, best_db
        if type == "per_class":
            best_f1 = f1_scores.max(1)
            best_db = dbs[f1_scores.argmax(0)]
            print(f"Best F1: {best_f1} at DB: {best_db}")
            return best_f1, best_db

    # ------------------------------------------------------------------
    # 新增：保存预测结果
    def _save_predictions(self, logits: torch.Tensor, dataset, save_name: str = "test_predictions.feather"):
        """
        将测试集预测结果按 best_db 阈值二值化后，与原始 text、target 一并保存为 feather 文件。
        - logits: sigmoid 后的概率张量，形状 (n_samples, num_labels)
        - dataset: ICDMultiLabelDataset，对应测试集，用于获取 text 和 target 原始信息
        - save_name: 保存文件名，默认 test_predictions.feather
        """
        if len(logits) != len(dataset):
            raise ValueError("logits 行数与 dataset 大小不一致，无法对齐保存预测结果")

        # 1. 二值化预测
        preds = (logits > self.best_db).int().cpu().numpy()  # shape (n_samples, num_labels)

        # 2. 构造 code 列顺序（idx -> code）
        idx2code = {idx: code for code, idx in dataset.code2idx.items()}
        codes_sorted = [idx2code[idx] for idx in range(len(idx2code))]

        # 3. 组装 DataFrame
        data_dict = {
            "text": dataset.texts,
            "target": dataset.targets,  # 使用分号连接保存原始多标签列表
        }
        # 为每个 code 添加预测列
        for idx, code in enumerate(codes_sorted):
            data_dict[code] = preds[:, idx]

        df_pred = pd.DataFrame(data_dict)

        # 4. 保存 feather
        save_path = os.path.join(self.output_dir, save_name)
        df_pred.to_feather(save_path)
        print(f"Saved predictions to {save_path} (阈值: {self.best_db:.4f})")

    # ------------------------------------------------------------------
    # 新增：导出注意力贡献 JSONL
    def _export_attention_contributions(self, loader, top_labels: int = 5, top_tokens: int = 10,
                                        save_name: str = "test_attention_contrib.jsonl"):
        """
        针对 loader（通常为 test_loader）逐样本导出：
          - raw_text
          - pred_codes（按 best_db 阈值）
          - true_codes（数据集中提供）
          - contributions：对每个预测标签，给出注意力最高的 top-N token 及其权重
        保存为 JSONL（每行一个样本）。
        """
        model = self.model.module if self.use_ddp else self.model
        model.eval()
        device = self.device
        dataset = loader.dataset
        tokenizer = None
        if hasattr(dataset, 'text_loader') and hasattr(dataset.text_loader, 'tokenizer'):
            tokenizer = dataset.text_loader.tokenizer
        if tokenizer is None:
            raise RuntimeError("找不到 tokenizer（dataset.text_loader.tokenizer 不存在）")

        # 代码索引映射
        idx2code = {idx: code for code, idx in dataset.code2idx.items()}

        save_path = os.path.join(self.output_dir, save_name)
        os.makedirs(self.output_dir, exist_ok=True)

        with torch.no_grad(), open(save_path, 'w', encoding='utf-8') as f:
            pbar = tqdm(total=len(loader), desc="export_contrib")
            sample_cursor = 0  # 跟踪已处理样本数量，用于正确索引原始数据
            for x_batch, y_batch in loader:
                input_ids = x_batch['input_ids'].to(device)
                attention_mask = x_batch['attention_mask'].to(device)

                # 前向与注意力
                if hasattr(model, 'chunk_and_encode'):
                    text_hidden, processed_attention_mask = model.chunk_and_encode(input_ids, attention_mask)
                    label_embs_for_attention, _ = model.get_label_embeddings()
                    logits, _ = model.attention(text_hidden, processed_attention_mask, label_embs_for_attention)
                    _, alpha_agg = model.attention.get_attention_alpha(
                        text_hidden, processed_attention_mask, label_embs_for_attention, aggregate="mean"
                    )
                    eff_mask = processed_attention_mask
                else:
                    outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
                    text_hidden = outputs.last_hidden_state
                    label_embs_for_attention, _ = model.get_label_embeddings()
                    logits, _ = model.attention(text_hidden, attention_mask, label_embs_for_attention)
                    _, alpha_agg = model.attention.get_attention_alpha(
                        text_hidden, attention_mask, label_embs_for_attention, aggregate="mean"
                    )
                    eff_mask = attention_mask

                probs = torch.sigmoid(logits)  # [B, C]
                preds_bin = (probs > self.best_db).int()  # [B, C]

                B, L = input_ids.size()
                valid_lens = eff_mask.sum(dim=1).tolist()

                for b in range(B):
                    global_idx = sample_cursor + b

                    # 非 Subset：直接以全局样本下标读取对应文本与标签
                    raw_text = dataset.texts[global_idx] if hasattr(dataset, 'texts') else None
                    true_codes = dataset.targets[global_idx] if hasattr(dataset, 'targets') else []


                    # 将 true_codes 转为可 JSON 序列化的纯 Python 类型
                    if isinstance(true_codes, np.ndarray):
                        true_codes = true_codes.tolist()
                    if isinstance(true_codes, (list, tuple)):
                        sanitized = []
                        for x in true_codes:
                            if isinstance(x, (np.integer,)):
                                sanitized.append(int(x))
                            elif isinstance(x, (np.floating,)):
                                sanitized.append(float(x))
                            elif isinstance(x, (np.bool_,)):
                                sanitized.append(bool(x))
                            else:
                                sanitized.append(x)
                        true_codes = sanitized

                    # 预测标签索引，按概率排序取前 top_labels 再与阈值交集，若为空则退回纯 top-k
                    prob_b = probs[b].detach().cpu()
                    pred_mask = preds_bin[b].bool().detach().cpu()
                    sorted_idx = torch.argsort(prob_b, descending=True).tolist()
                    pred_idx = [i for i in sorted_idx if pred_mask[i].item()]
                    if len(pred_idx) == 0:
                        pred_idx = sorted_idx[:top_labels]
                    else:
                        pred_idx = pred_idx[:top_labels]
                    pred_codes = [idx2code[i] for i in pred_idx]

                    # tokens（对齐到有效长度）
                    ids_b = input_ids[b, :valid_lens[b]].detach().cpu().tolist()
                    tokens = tokenizer.convert_ids_to_tokens(ids_b)

                    # 注意力贡献（按标签）
                    contrib = {}
                    att_b = alpha_agg[b, :, :valid_lens[b]].detach().cpu()  # [C, L]
                    for i, code in zip(pred_idx, pred_codes):
                        att_i = att_b[i]  # [L]
                        # 选择 top_tokens 个 token
                        topk = min(top_tokens, att_i.numel())
                        vals, inds = torch.topk(att_i, k=topk)
                        items = []
                        for score, idx in zip(vals.tolist(), inds.tolist()):
                            items.append({
                                "token_index": int(idx),
                                "token": tokens[idx] if idx < len(tokens) else "",
                                "score": float(score)
                            })
                        contrib[code] = items

                    rec = {
                        "text": raw_text,
                        "pred_codes": pred_codes,
                        "true_codes": true_codes,
                        "topk_token_contrib": contrib
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sample_cursor += B
                pbar.update(1)
            pbar.close()
        print(f"Saved attention contributions to {save_path}")
    