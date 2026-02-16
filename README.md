
## 🔧 Environment Setup

### 1. Create Conda Environment

```bash
conda create -n cola-icd python=3.10
conda activate cola-icd
```

### 2. Install Dependencies

```bash
# Install requirements.txt
pip install -r requirements.txt
```
### Download Pretrained Models

The project uses the following pretrained models that need to be downloaded in advance:

```bash
# - SapBERT: https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext
```


## 📁 Data Preparation
### 1. MIMIC Dataset Access
Due to the privacy regulations associated with clinical data, we cannot provide the MIMIC-III dataset directly.
* Please request access to MIMIC-III (v1.4) via [PhysioNet](https://physionet.org/content/mimiciii/).
* Complete the required CITI training and sign the Data Use Agreement.

### 2. Preprocessing
To ensure a fair comparison with baselines, we follow the standard data preprocessing pipeline and splits established by Edin et al. (2023)** (see their [reproducibility study](https://github.com/JoakimEdin/medical-coding-reproducibility)).

### Data Directory Structure

```
data/
├── mimiciii_full/                        # MIMIC-III full dataset
│   ├── MIMICIII_train.feather            # Training set
│   ├── MIMICIII_val.feather              # Validation set
│   ├── MIMICIII_test.feather             # Test set
│   └── icd9_codes_mimiciii.feather       # ICD codes and descriptions
├── mimiciv_full/                         # MIMIC-IV full dataset
│   ├── mimiciv_icd9_train.feather        # Training set
│   ├── mimiciv_icd9_val.feather          # Validation set
│   ├── mimiciv_icd9_test.feather         # Test set
│   └── filtered_icd_codes_with_desc.feather  # ICD codes and descriptions
├── mimiciii_50/                          # MIMIC-III Top-50 subset
│   ├── mimiciii_50_train.feather
│   ├── mimiciii_50_val.feather
│   ├── mimiciii_50_test.feather
│   └── top50.feather
├── icd_synonyms_enhanced_gemini.json     # ICD synonyms file (optional)
└── icd9_abbreviations_gemini.json        # ICD abbreviations file (optional)


## 🚀 Quick Start

### 1. Basic Training

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

### 2. Training with Contrastive Learning

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

### 3. Training with Synonym Enhancement

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


## ⚙️ Parameter Description

### Data-related Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train_file` | - | Training set path (.feather) |
| `--val_file` | - | Validation set path (.feather) |
| `--test_file` | - | Test set path (.feather) |
| `--codes_file` | - | ICD code file path (.feather) |
| `--synonyms_file` | - | Synonyms file path (.json, optional) |
| `--abbreviations_file` | - | Abbreviations file path (.json, optional) |

### Model-related Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pretrained_model_name` | Clinical-Longformer | Pretrained model for text encoder |
| `--label_model_name` | Bio_ClinicalBERT | Pretrained model for label encoder |
| `--model_type` | longformer | Model type: longformer, bert_chunk, bert_chunk_v2 |
| `--chunk_size` | 512 | BERT chunk size |
| `--term_count` | 1 | Number of synonyms per label |
| `--max_length` | 4096 | Maximum text length |

### Training-related Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 12 | Batch size |
| `--epochs` | 5 | Number of training epochs |
| `--lr` | 2e-5 | Learning rate |
| `--warmup_steps` | 0 | Warmup steps |
| `--weight_decay` | 0.0 | Weight decay |
| `--early_stopping` | False | Whether to enable early stopping |
| `--early_stopping_patience` | 5 | Early stopping patience |
| `--scheduler_type` | cosine | Learning rate scheduler type |

### Contrastive Learning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_contrastive` | False | Whether to enable contrastive learning |
| `--contrastive_loss_weight` | 0.1 | Contrastive learning loss weight |
| `--contrastive_temperature` | 0.1 | Contrastive learning temperature |

### GNN-related Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_gnn` | False | Whether to enable GNN |
| `--adj_matrix_mode` | ppmi | Adjacency matrix mode: binary, count, ppmi, hierarchy |

### Other Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_wandb` | False | Whether to enable W&B logging |
| `--use_amp` | True | Whether to use mixed precision training |
| `--output_dir` | checkpoints/{timestamp} | Model save directory |
| `--threshold` | 0.5 | Classification threshold |

## 📊 Model Evaluation

### Evaluate Trained Models

Set `--epochs 0` and specify `--output_dir` as the saved model directory:

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

### Evaluation Metrics

The model uses the following evaluation metrics:
- **Precision (Macro/Micro)**: Precision rate
- **Recall (Macro/Micro)**: Recall rate
- **F1 Score (Macro/Micro)**: F1 score
- **AUC (Macro/Micro)**: Area Under ROC Curve
- **Precision@K** (K=5, 8, 10, 15): Top-K precision
