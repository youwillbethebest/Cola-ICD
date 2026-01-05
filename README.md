# AttentionICD: Label Attention-based ICD Code Prediction

基于标签注意力机制的ICD编码自动分类系统。本项目使用预训练语言模型结合标签注意力机制，实现对临床文本的多标签ICD编码分类。

## 📋 目录

- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [参数说明](#参数说明)
- [模型评估](#模型评估)

## 🔧 环境配置

### 1. 创建 Conda 环境

```bash
conda create -n attentionicd python=3.10
conda activate attentionicd
```

### 2. 安装依赖

```bash
# 安装 PyTorch (根据您的CUDA版本调整)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 安装其他核心依赖
pip install transformers==4.51.3
pip install pandas==2.2.3
pip install pyarrow==18.1.0
pip install scikit-learn==1.6.1
pip install wandb==0.21.0
pip install sentence-transformers==3.3.1
pip install torch-geometric==2.6.1

# 安装 PyG 相关库 (根据CUDA版本调整)
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

### 3. 下载预训练模型

项目使用以下预训练模型，需要提前下载：

```bash
# 文本编码器 (选择其一)
# - Clinical-Longformer: https://huggingface.co/yikuan8/Clinical-Longformer
# - SapBERT: https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext

# 标签编码器
# - Bio_ClinicalBERT: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT
# - SapBERT: https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext
```

将模型下载到项目根目录下对应的文件夹：
- `Clinical-Longformer/`
- `Bio_ClinicalBERT/`
- `SapBERT-from-PubMedBERT-fulltext/` (如使用)

## 📁 数据准备

### 数据格式

训练数据需要是 `.feather` 格式的 DataFrame，包含以下列：
- `TEXT`: 临床文本内容
- `LABELS`: ICD编码列表

ICD编码文件需包含：
- `icd_code`: ICD编码
- `long_title`: 编码描述

### 数据目录结构

```
data/
├── mimiciii_full/                    # MIMIC-III 全量数据
│   ├── MIMICIII_train.feather        # 训练集
│   ├── MIMICIII_val.feather          # 验证集
│   ├── MIMICIII_test.feather         # 测试集
│   └── icd9_codes_mimiciii.feather   # ICD编码及描述
├── mimiciii_50/                      # MIMIC-III Top-50 子集
│   ├── mimiciii_50_train.feather
│   ├── mimiciii_50_val.feather
│   ├── mimiciii_50_test.feather
│   └── top50.feather
├── icd_synonyms_enhanced_gemini.json # ICD同义词文件 (可选)
└── icd9_abbreviations_gemini.json    # ICD缩写文件 (可选)
```

## 📂 项目结构

```
AttentionICD_new/
├── main.py                 # 主入口文件
├── eval.sh                 # 评估脚本
├── mainterm.sh             # 训练脚本示例
├── maincontrastive.sh      # 对比学习训练脚本示例
├── src/
│   ├── model.py            # 模型定义
│   ├── module.py           # 注意力机制等模块
│   ├── data_loader.py      # 数据加载器
│   ├── trainer.py          # 训练器
│   ├── metric.py           # 评估指标
│   └── loss.py             # 损失函数
├── data/                   # 数据目录
├── checkpoints/            # 模型检查点保存目录
├── Clinical-Longformer/    # 预训练模型
└── Bio_ClinicalBERT/       # 预训练模型
```

## 🚀 快速开始

### 1. 基础训练

```bash
python main.py \
    --train_file data/mimiciii_full/MIMICIII_train.feather \
    --val_file data/mimiciii_full/MIMICIII_val.feather \
    --test_file data/mimiciii_full/MIMICIII_test.feather \
    --codes_file data/mimiciii_full/icd9_codes_mimiciii.feather \
    --pretrained_model_name SapBERT-from-PubMedBERT-fulltext \
    --label_model_name SapBERT-from-PubMedBERT-fulltext \
    --model_type bert_chunk \
    --chunk_size 256 \
    --batch_size 6 \
    --epochs 20 \
    --lr 2e-5 \
    --warmup_steps 2000 \
    --early_stopping
```

### 2. 使用对比学习训练

```bash
python main.py \
    --train_file data/mimiciii_full/MIMICIII_train.feather \
    --val_file data/mimiciii_full/MIMICIII_val.feather \
    --test_file data/mimiciii_full/MIMICIII_test.feather \
    --codes_file data/mimiciii_full/icd9_codes_mimiciii.feather \
    --pretrained_model_name SapBERT-from-PubMedBERT-fulltext \
    --label_model_name SapBERT-from-PubMedBERT-fulltext \
    --model_type bert_chunk \
    --chunk_size 256 \
    --batch_size 6 \
    --epochs 20 \
    --lr 2e-5 \
    --warmup_steps 2000 \
    --early_stopping \
    --use_contrastive \
    --contrastive_loss_weight 0.001 \
    --contrastive_temperature 0.3
```

### 3. 使用同义词增强

```bash
python main.py \
    --train_file data/mimiciii_full/MIMICIII_train.feather \
    --val_file data/mimiciii_full/MIMICIII_val.feather \
    --test_file data/mimiciii_full/MIMICIII_test.feather \
    --codes_file data/mimiciii_full/icd9_codes_mimiciii.feather \
    --pretrained_model_name SapBERT-from-PubMedBERT-fulltext \
    --label_model_name SapBERT-from-PubMedBERT-fulltext \
    --model_type bert_chunk \
    --chunk_size 256 \
    --batch_size 6 \
    --epochs 20 \
    --term_count 4 \
    --synonyms_file data/icd_synonyms_enhanced_gemini.json \
    --early_stopping
```

### 4. 使用 SLURM 提交作业

如果您使用 SLURM 集群，可以参考以下脚本：

```bash
sbatch mainterm.sh
```

## ⚙️ 参数说明

### 数据相关参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_file` | - | 训练集路径 (.feather) |
| `--val_file` | - | 验证集路径 (.feather) |
| `--test_file` | - | 测试集路径 (.feather) |
| `--codes_file` | - | ICD编码文件路径 (.feather) |
| `--synonyms_file` | - | 同义词文件路径 (.json, 可选) |
| `--abbreviations_file` | - | 缩写文件路径 (.json, 可选) |

### 模型相关参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pretrained_model_name` | Clinical-Longformer | 文本编码器预训练模型 |
| `--label_model_name` | Bio_ClinicalBERT | 标签编码器预训练模型 |
| `--model_type` | longformer | 模型类型: longformer, bert_chunk, bert_chunk_v2 |
| `--chunk_size` | 512 | BERT chunk大小 |
| `--term_count` | 1 | 每个标签使用的同义词数量 |
| `--max_length` | 4096 | 文本最大长度 |

### 训练相关参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 12 | 批次大小 |
| `--epochs` | 5 | 训练轮数 |
| `--lr` | 2e-5 | 学习率 |
| `--warmup_steps` | 0 | 预热步数 |
| `--weight_decay` | 0.0 | 权重衰减 |
| `--early_stopping` | False | 是否启用早停 |
| `--early_stopping_patience` | 5 | 早停耐心值 |
| `--scheduler_type` | cosine | 学习率调度器类型 |

### 对比学习参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_contrastive` | False | 是否启用对比学习 |
| `--contrastive_loss_weight` | 0.1 | 对比学习损失权重 |
| `--contrastive_temperature` | 0.1 | 对比学习温度参数 |

### GNN相关参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_gnn` | False | 是否启用GNN |
| `--adj_matrix_mode` | ppmi | 邻接矩阵模式: binary, count, ppmi, hierarchy |

### 其他参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_wandb` | False | 是否启用W&B日志 |
| `--use_amp` | True | 是否使用混合精度训练 |
| `--output_dir` | checkpoints/{timestamp} | 模型保存目录 |
| `--threshold` | 0.5 | 分类阈值 |

## 📊 模型评估

### 评估已训练的模型

设置 `--epochs 0` 并指定 `--output_dir` 为已保存的模型目录：

```bash
python main.py \
    --train_file data/mimiciii_full/MIMICIII_train.feather \
    --val_file data/mimiciii_full/MIMICIII_val.feather \
    --test_file data/mimiciii_full/MIMICIII_test.feather \
    --codes_file data/mimiciii_full/icd9_codes_mimiciii.feather \
    --batch_size 2 \
    --epochs 0 \
    --term_count 4 \
    --output_dir checkpoints/your_checkpoint_dir
```

### 评估指标

模型使用以下评估指标：
- **Precision (Macro/Micro)**: 精确率
- **Recall (Macro/Micro)**: 召回率
- **F1 Score (Macro/Micro)**: F1分数
- **AUC (Macro/Micro)**: ROC曲线下面积
- **Precision@K** (K=5, 8, 10, 15): Top-K精确率
- **MAP**: 平均精确率均值

## 💡 硬件要求

- **GPU**: 推荐使用 NVIDIA H100 (80GB) 或 A100 (40GB/80GB)
- **内存**: 至少 128GB RAM (推荐 350GB)
- **存储**: 至少 50GB 可用空间

## 📝 注意事项

1. **数据隐私**: MIMIC数据集需要通过PhysioNet申请访问权限
2. **显存管理**: 如果显存不足，可以减小 `batch_size` 或 `chunk_size`
3. **训练时间**: MIMIC-III全量数据集训练约需12-24小时（单GPU）

## 📧 联系方式

如有问题，请联系项目维护者。

## 📄 License

本项目仅供学术研究使用。

