import torch
import torch.nn as nn
from transformers import AutoModel
from src.module import LabelAttention
from src.data_loader import LabelLoader

class ClinicalLongformerLabelAttention(nn.Module):
    """
    Clinical-Longformer 主干 + 标签注意力 多标签分类模型。
    Args:
      longformer_path: 本地临床 Longformer 模型路径
      codes_file: ICD 代码及描述文件路径
      label_model_name: 用于加载标签描述的预训练模型名称，默认 Bio_ClinicalBERT
    """
    def __init__(self,
                 longformer_path: str,
                 codes_file: str,
                 label_model_name: str = "Bio_ClinicalBERT"):
        super().__init__()
        # 文本编码器
        self.model= AutoModel.from_pretrained(longformer_path)
        hidden_size = self.model.config.hidden_size

        # 标签加载与编码
        label_loader = LabelLoader(
            codes_file=codes_file,
            pretrained_model_name=label_model_name
        )
        self.num_labels = label_loader.num_labels
        label_enc = label_loader()
        with torch.no_grad():
            label_outputs = label_loader.model(
            input_ids=label_enc['input_ids'],
            attention_mask=label_enc['attention_mask']
            )
            # pooler_output 或 CLS token 表示
        if hasattr(label_outputs, 'pooler_output'):
            label_embs = label_outputs.pooler_output  # (num_labels, H)
        else:
                # 获取每个标签的 [CLS] token 表示作为标签嵌入
            label_embs = label_outputs.last_hidden_state[:, 0]  # (num_labels, H)
        # 注册标签嵌入
        self.register_buffer('label_embs', label_embs)

        # 标签感知注意力模块
        self.attention = LabelAttention(attention_head=1,rep_droupout_num=0.1,head_pooling="concat",att_dropout_num=0.1,attention_dim=hidden_size,num_labels=self.num_labels)
        # 分类头

    def forward(self,   
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        # 文本编码
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        text_hidden = outputs.last_hidden_state  # (N, L, H)

        # 标签注意力生成候选标签上下文表示 (N, C, H)
        logits= self.attention(text_hidden, attention_mask,self.label_embs)
        # 分类得分 (N, C)
        return logits
    
