import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModel
from src.module import LabelAttention, GraphAttentionLayer, LabelGNN, LabelAttentionV2
from src.data_loader import SynonymLabelLoader


class ClinicalLongformerLabelAttention(nn.Module):
    """
    Clinical-Longformer 主干 + 标签注意力 多标签分类模型。
    Args:
      longformer_path: 本地临床 Longformer 模型路径
      codes_file: ICD 代码及描述文件路径
      label_model_name: 用于加载标签描述的预训练模型名称，默认 Bio_ClinicalBERT
      use_gnn: 是否使用GNN更新标签嵌入
      adj_matrix: 标签共现邻接矩阵
    """
    def __init__(self,
                 longformer_path: str,
                 codes_file: str | None = None,
                 label_model_name: str = "Bio_ClinicalBERT",
                 term_counts: int = 1,
                 label_loader: Optional[SynonymLabelLoader] = None,
                 use_gnn: bool = False,
                 adj_matrix: Optional[torch.Tensor] = None):
        super().__init__()
        # 文本编码器
        self.model = AutoModel.from_pretrained(longformer_path)
        hidden_size = self.model.config.hidden_size

        # ---------------- 标签加载 ----------------
        if label_loader is None:          # 若外部未传入，则内部创建（保持兼容）
            assert codes_file is not None, "当未提供 label_loader 时，必须指定 codes_file"
            label_loader = SynonymLabelLoader(
                codes_file=codes_file,
                pretrained_model_name=label_model_name,
                term_count=term_counts
            )

        self.num_labels = label_loader.num_labels
        label_embs = label_loader()       # 预计算标签嵌入
        #self.register_buffer("label_embs", label_embs)
        self.label_embs = nn.Parameter(label_embs)
        self.term_counts = term_counts  # 保存 term_counts
        # ---------------- GNN 模块 (可选) ----------------
        self.use_gnn = use_gnn
        if self.use_gnn:
            assert adj_matrix is not None, "当 use_gnn=True 时，必须提供 adj_matrix。"
            # 兼容两种格式：密集矩阵 Tensor 或稀疏表示 (edge_index, edge_weight)

                # 新接口：直接传入 (edge_index, edge_weight)
            edge_index, edge_weight = adj_matrix
            # 注册为 buffer
            self.register_buffer("edge_index", edge_index)
            self.register_buffer("edge_weight", edge_weight)
            # GNN 模块内部已是两层 GCNConv
            self.gnn = LabelGNN(hidden_size, hidden_size // 2, hidden_size)
            
            # 可以堆叠多层GAT
            # 或者只用一层
            # self.gat_layer = GraphAttentionLayer(hidden_size, hidden_size, dropout=0.2, alpha=0.2, concat=False)


        # 标签感知注意力
        self.attention = LabelAttentionV2(
            attention_head=term_counts,
            rep_droupout_num=0.1,
            head_pooling="max",
            att_dropout_num=0.1,
            attention_dim=hidden_size,
            num_labels=self.num_labels,
            est_cls=1
        )
        # 投影头
        projection_dim = 256
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim)
        )
        self.text_projection_head = self.shared_head
        self.label_projection_head = self.shared_head

    def get_label_embeddings(self):
        """
        获取标签嵌入，如果启用GNN则使用图注意力网络更新
        
        Returns:
            label_embs_for_attention: (C*k, H) - 用于注意力机制的标签嵌入
            label_proto_for_contrastive: (C, H) - 用于对比学习的标签原型
        """
        # 原始标签嵌入已经是 (C*k, H) 的形式
        raw_embs = self.label_embs  # (C*k, H)
        
        if self.use_gnn:
            # 调用 GNN 时传入稀疏表示
            updated_embs = self.gnn(raw_embs, self.edge_index, self.edge_weight)  # (C*k, H)

            # 直接用更新后的嵌入做注意力
            label_embs_for_attention = updated_embs  # (C*k, H)

            # 对比学习时，取每个标签的 k 个同义词平均作为原型
            label_proto_for_contrastive = updated_embs.view(
                self.num_labels, self.term_counts, -1
            ).mean(dim=1)  # (C, H)
            
        else:
            # 如果不使用GNN，直接使用原始嵌入
            label_embs_for_attention = raw_embs  # (C*k, H)
            
            # 对比学习的标签原型：将每个标签的k个同义词取平均
            label_proto_for_contrastive = raw_embs.view(
                self.num_labels, self.term_counts, -1
            ).mean(dim=1)  # (C, H)

        return label_embs_for_attention, label_proto_for_contrastive
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> dict:
        # 文本编码
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        text_hidden = outputs.last_hidden_state  # (N, L, H)
        
        # Mean Pooling for text_feat, using attention_mask to ignore padding
        # attention_mask_expanded = attention_mask.unsqueeze(-1)
        # sum_embeddings = torch.sum(text_hidden * attention_mask_expanded, dim=1)
        # sum_mask = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9)
        # text_feat = sum_embeddings / sum_mask
        label_embs_for_attention, label_proto_for_contrastive = self.get_label_embeddings()
        # 标签注意力生成 logits
        logits, per_label_text_feat = self.attention(text_hidden, attention_mask, label_embs_for_attention)

        # 应用投影头，为对比学习生成隔离的特征
        contrastive_text_feat = self.text_projection_head(per_label_text_feat)
        contrastive_label_proto = self.label_projection_head(label_proto_for_contrastive)

        return logits, contrastive_text_feat, contrastive_label_proto
        
    

class ClinicalLongformerLabelAttentionV2(nn.Module):
    '''
    
    '''
    
    def __init__(self,
                 longformer_path: str,
                 codes_file: str,
                 label_model_name: str = "Bio_ClinicalBERT"):
        super().__init__()
        self.model= AutoModel.from_pretrained(longformer_path)
        hidden_size = self.model.config.hidden_size
        
        