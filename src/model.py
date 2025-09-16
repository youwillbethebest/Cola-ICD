import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModel
from src.module import LabelAttention, GraphAttentionLayer, LabelGNN, LabelAttentionV2, HierLabelGNN
from src.data_loader import SynonymLabelLoader
import torch.nn.functional as F


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
        self.model = AutoModel.from_pretrained(longformer_path,add_pooling_layer=False)
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
        self.hier_mode = False
        if self.use_gnn:
            assert adj_matrix is not None, "当 use_gnn=True 时，必须提供 adj_matrix。"
            # 支持两类：共现图 (edge_index, edge_weight) 或 层级图 {'up':(...), 'down':(...)}
            if isinstance(adj_matrix, dict) and "up" in adj_matrix and "down" in adj_matrix:
                self.hier_mode = True
                up_ei, up_w = adj_matrix["up"]
                down_ei, down_w = adj_matrix["down"]
                self.register_buffer("up_edge_index", up_ei)
                self.register_buffer("up_edge_weight", up_w)
                self.register_buffer("down_edge_index", down_ei)
                self.register_buffer("down_edge_weight", down_w)
                self.gnn_hier = HierLabelGNN(hidden_size, hidden_size // 2, hidden_size)
            else:
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
        self.attention = LabelAttention(
            attention_head=term_counts,
            rep_droupout_num=0.2,
            head_pooling="concat",
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
            if self.hier_mode:
                # 代码级图：先聚合术语为代码级，再图更新，再广播回术语位
                code_pool = raw_embs.view(self.num_labels, self.term_counts, -1).max(dim=1)[0]  # (C, H)
                code_updated = self.gnn_hier(
                    code_pool,
                    self.up_edge_index, self.up_edge_weight,
                    self.down_edge_index, self.down_edge_weight
                )  # (C, H)
                label_embs_for_attention = code_updated.repeat_interleave(self.term_counts, dim=0)  # (C*k, H)
                label_proto_for_contrastive = code_updated  # (C, H)
            else:
                # 调用 GNN 时传入稀疏表示（共现图）
                updated_embs = self.gnn(raw_embs, self.edge_index, self.edge_weight)  # (C*k, H)
                label_embs_for_attention = updated_embs  # (C*k, H)
                label_proto_for_contrastive = updated_embs.view(
                    self.num_labels, self.term_counts, -1
                ).max(dim=1)[0]  # (C, H)
            
        else:
            # 如果不使用GNN，直接使用原始嵌入
            label_embs_for_attention = raw_embs  # (C*k, H)
            
            # 对比学习时，取每个标签的 k 个同义词进行 max pooling
            label_proto_for_contrastive = raw_embs.view(
                self.num_labels, self.term_counts, -1
            ).max(dim=1)[0]  # (C, H)

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
        
        

class ClinicalBERTChunkAttention(nn.Module):
    """
    基于BERT的文本分块注意力模型
    
    处理流程：
    1. 文本分块：(batch_size, total_tokens) → (batch_size, num_chunks, chunk_size)
    2. BERT独立编码：每个chunk通过BERT → (batch_size, num_chunks, chunk_size, hidden_size)
    3. 重新组织：拼接所有chunk的embedding → (batch_size, total_tokens, hidden_size)
    4. 标签注意力：在拼接后的完整序列上进行注意力计算
    
    Args:
        bert_model_path: BERT模型路径
        chunk_size: 每个chunk的token数量，默认128
        codes_file: ICD代码及描述文件路径
        label_model_name: 标签描述的预训练模型名称
        term_counts: 每个标签的同义词数量
        label_loader: 外部传入的标签加载器
        use_gnn: 是否使用GNN更新标签嵌入
        adj_matrix: 标签共现邻接矩阵
    """
    
    def __init__(self,
                 bert_model_path: str,
                 chunk_size: int = 512,
                 codes_file: str | None = None,
                 label_model_name: str = "Bio_ClinicalBERT",
                 term_counts: int = 1,
                 label_loader: Optional[SynonymLabelLoader] = None,
                 use_gnn: bool = False,
                 adj_matrix: Optional[torch.Tensor] = None):
        super().__init__()
        
        # BERT编码器
        self.bert_model = AutoModel.from_pretrained(bert_model_path,add_pooling_layer=False)
        self.chunk_size = chunk_size
        hidden_size = self.bert_model.config.hidden_size
        
        # ---------------- 标签加载 ----------------
        if label_loader is None:
            assert codes_file is not None, "当未提供 label_loader 时，必须指定 codes_file"
            label_loader = SynonymLabelLoader(
                codes_file=codes_file,
                pretrained_model_name=label_model_name,
                term_count=term_counts
            )

        self.num_labels = label_loader.num_labels
        label_embs = label_loader()
        self.label_embs = nn.Parameter(label_embs)
        self.term_counts = term_counts
        
        # ---------------- GNN 模块 (可选) ----------------
        self.use_gnn = use_gnn
        self.hier_mode = False
        if self.use_gnn:
            assert adj_matrix is not None, "当 use_gnn=True 时，必须提供 adj_matrix。"
            if isinstance(adj_matrix, dict) and "up" in adj_matrix and "down" in adj_matrix:
                self.hier_mode = True
                up_ei, up_w = adj_matrix["up"]
                down_ei, down_w = adj_matrix["down"]
                self.register_buffer("up_edge_index", up_ei)
                self.register_buffer("up_edge_weight", up_w)
                self.register_buffer("down_edge_index", down_ei)
                self.register_buffer("down_edge_weight", down_w)
                self.gnn_hier = HierLabelGNN(hidden_size, hidden_size // 2, hidden_size)
            else:
                edge_index, edge_weight = adj_matrix
                self.register_buffer("edge_index", edge_index)
                self.register_buffer("edge_weight", edge_weight)
                self.gnn = LabelGNN(hidden_size, hidden_size // 2, hidden_size)

        # 标签感知注意力
        self.attention = LabelAttention(
            attention_head=term_counts,
            rep_droupout_num=0.2,
            head_pooling="concat",
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

    def chunk_and_encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        文本分块并进行BERT编码
        
        Args:
            input_ids: (batch_size, total_tokens)
            attention_mask: (batch_size, total_tokens)
            
        Returns:
            chunked_embeddings: (batch_size, total_tokens, hidden_size)
            final_attention_mask: (batch_size, total_tokens)
        """
        batch_size, total_tokens = input_ids.shape
        
        # 计算需要的chunk数量
        num_chunks = (total_tokens + self.chunk_size - 1) // self.chunk_size
        
        # 如果总长度不能被chunk_size整除，需要padding
        padded_length = num_chunks * self.chunk_size
        if padded_length > total_tokens:
            # 使用pad_token_id进行padding，通常是0
            pad_length = padded_length - total_tokens
            input_ids = F.pad(input_ids, (0, pad_length), value=0)
            attention_mask = F.pad(attention_mask, (0, pad_length), value=0)
        
        # 第一步：分块重塑
        # (batch_size, total_tokens) → (batch_size, num_chunks, chunk_size)
        chunked_input_ids = input_ids.view(batch_size, num_chunks, self.chunk_size)
        chunked_attention_mask = attention_mask.view(batch_size, num_chunks, self.chunk_size)
        
        # 第二步：BERT独立编码
        # 展平为 (batch_size * num_chunks, chunk_size)
        flat_input_ids = chunked_input_ids.view(-1, self.chunk_size)
        flat_attention_mask = chunked_attention_mask.view(-1, self.chunk_size)
        
        # 每个chunk独立通过BERT编码
        with torch.cuda.device(flat_input_ids.device):
            chunk_outputs = self.bert_model(
                input_ids=flat_input_ids,
                attention_mask=flat_attention_mask
            )
        chunk_embeddings = chunk_outputs.last_hidden_state  # (batch_size * num_chunks, chunk_size, hidden_size)
        
        # 第三步：重新组织embedding
        # 重塑为 (batch_size, num_chunks, chunk_size, hidden_size)
        chunk_embeddings = chunk_embeddings.view(batch_size, num_chunks, self.chunk_size, -1)
        # 拼接所有chunk: (batch_size, num_chunks * chunk_size, hidden_size)
        concatenated_embeddings = chunk_embeddings.view(batch_size, -1, chunk_embeddings.size(-1))
        
        # 同样处理attention_mask
        final_attention_mask = chunked_attention_mask.view(batch_size, -1)
        
        # 截取到原始长度
        if padded_length > total_tokens:
            concatenated_embeddings = concatenated_embeddings[:, :total_tokens, :]
            final_attention_mask = final_attention_mask[:, :total_tokens]
        
        return concatenated_embeddings, final_attention_mask

    def get_label_embeddings(self):
        """获取标签嵌入，与原模型保持一致"""
        raw_embs = self.label_embs
        
        if self.use_gnn:
            if self.hier_mode:
                code_pool = raw_embs.view(self.num_labels, self.term_counts, -1).max(dim=1)[0]
                code_updated = self.gnn_hier(
                    code_pool,
                    self.up_edge_index, self.up_edge_weight,
                    self.down_edge_index, self.down_edge_weight
                )
                label_embs_for_attention = code_updated.repeat_interleave(self.term_counts, dim=0)
                label_proto_for_contrastive = code_updated
            else:
                updated_embs = self.gnn(raw_embs, self.edge_index, self.edge_weight)
                label_embs_for_attention = updated_embs
                label_proto_for_contrastive = updated_embs.view(
                    self.num_labels, self.term_counts, -1
                ).max(dim=1)[0]
        else:
            label_embs_for_attention = raw_embs
            label_proto_for_contrastive = raw_embs.view(
                self.num_labels, self.term_counts, -1
            ).max(dim=1)[0]

        return label_embs_for_attention, label_proto_for_contrastive
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> dict:
        """
        前向传播
        
        Args:
            input_ids: (batch_size, total_tokens) 例如 (1, 2000)
            attention_mask: (batch_size, total_tokens)
            
        Returns:
            logits: 分类logits
            contrastive_text_feat: 用于对比学习的文本特征
            contrastive_label_proto: 用于对比学习的标签原型
        """
        # 文本分块编码
        text_hidden, processed_attention_mask = self.chunk_and_encode(input_ids, attention_mask)
        # text_hidden: (batch_size, total_tokens, hidden_size)
        # processed_attention_mask: (batch_size, total_tokens)
        
        # 获取标签嵌入
        label_embs_for_attention, label_proto_for_contrastive = self.get_label_embeddings()
        
        # 标签注意力生成logits
        logits, per_label_text_feat = self.attention(
            text_hidden, 
            processed_attention_mask, 
            label_embs_for_attention
        )
        
        # 应用投影头，为对比学习生成隔离的特征
        contrastive_text_feat = self.text_projection_head(per_label_text_feat)
        contrastive_label_proto = self.label_projection_head(label_proto_for_contrastive)

        return logits, contrastive_text_feat, contrastive_label_proto


class ClinicalBERTChunkAttentionV2(nn.Module):
    """
    基于BERT的文本分块注意力模型 - 版本2
    
    处理流程（先注意力再拼接）：
    1. 文本分块：(batch_size, total_tokens) → (batch_size, num_chunks, chunk_size)
    2. BERT独立编码：每个chunk通过BERT → (batch_size, num_chunks, chunk_size, hidden_size)
    3. 每个chunk内部进行标签注意力
    4. 拼接各chunk的注意力结果
    
    这个版本在每个chunk内部独立进行注意力计算，可能更适合长文本处理
    """
    
    def __init__(self,
                 bert_model_path: str,
                 chunk_size: int = 512,
                 codes_file: str | None = None,
                 label_model_name: str = "Bio_ClinicalBERT",
                 term_counts: int = 1,
                 label_loader: Optional[SynonymLabelLoader] = None,
                 use_gnn: bool = False,
                 adj_matrix: Optional[torch.Tensor] = None,
                 chunk_aggregation: str = "max"):  # 新增：chunk结果聚合方式
        super().__init__()
        
        # BERT编码器
        self.bert_model = AutoModel.from_pretrained(bert_model_path,add_pooling_layer=False)
        self.chunk_size = chunk_size
        self.chunk_aggregation = chunk_aggregation  # "mean", "max", "sum", "weighted"
        hidden_size = self.bert_model.config.hidden_size
        
        # 标签加载（与V1相同）
        if label_loader is None:
            assert codes_file is not None, "当未提供 label_loader 时，必须指定 codes_file"
            label_loader = SynonymLabelLoader(
                codes_file=codes_file,
                pretrained_model_name=label_model_name,
                term_count=term_counts
            )

        self.num_labels = label_loader.num_labels
        label_embs = label_loader()
        self.label_embs = nn.Parameter(label_embs)
        self.term_counts = term_counts
        
        # GNN模块（与V1相同）
        self.use_gnn = use_gnn
        self.hier_mode = False
        if self.use_gnn:
            assert adj_matrix is not None, "当 use_gnn=True 时，必须提供 adj_matrix。"
            if isinstance(adj_matrix, dict) and "up" in adj_matrix and "down" in adj_matrix:
                self.hier_mode = True
                up_ei, up_w = adj_matrix["up"]
                down_ei, down_w = adj_matrix["down"]
                self.register_buffer("up_edge_index", up_ei)
                self.register_buffer("up_edge_weight", up_w)
                self.register_buffer("down_edge_index", down_ei)
                self.register_buffer("down_edge_weight", down_w)
                self.gnn_hier = HierLabelGNN(hidden_size, hidden_size // 2, hidden_size)
            else:
                edge_index, edge_weight = adj_matrix
                self.register_buffer("edge_index", edge_index)
                self.register_buffer("edge_weight", edge_weight)
                self.gnn = LabelGNN(hidden_size, hidden_size // 2, hidden_size)

        # 标签感知注意力
        self.attention = LabelAttention(
            attention_head=term_counts,
            rep_droupout_num=0.1,
            head_pooling="concat",
            att_dropout_num=0.1,
            attention_dim=hidden_size,
            num_labels=self.num_labels,
            est_cls=1
        )
        
        # chunk权重网络（用于加权聚合）
        if self.chunk_aggregation == "weighted":
            self.chunk_weight_net = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid()
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
        """获取标签嵌入，与V1相同"""
        raw_embs = self.label_embs
        
        if self.use_gnn:
            if self.hier_mode:
                code_pool = raw_embs.view(self.num_labels, self.term_counts, -1).max(dim=1)[0]
                code_updated = self.gnn_hier(
                    code_pool,
                    self.up_edge_index, self.up_edge_weight,
                    self.down_edge_index, self.down_edge_weight
                )
                label_embs_for_attention = code_updated.repeat_interleave(self.term_counts, dim=0)
                label_proto_for_contrastive = code_updated
            else:
                updated_embs = self.gnn(raw_embs, self.edge_index, self.edge_weight)
                label_embs_for_attention = updated_embs
                label_proto_for_contrastive = updated_embs.view(
                    self.num_labels, self.term_counts, -1
                ).max(dim=1)[0]
        else:
            label_embs_for_attention = raw_embs
            label_proto_for_contrastive = raw_embs.view(
                self.num_labels, self.term_counts, -1
            ).max(dim=1)[0]

        return label_embs_for_attention, label_proto_for_contrastive

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> dict:
        """
        前向传播 - V2版本（先注意力再拼接）
        """
        batch_size, total_tokens = input_ids.shape
        
        # 计算需要的chunk数量
        num_chunks = (total_tokens + self.chunk_size - 1) // self.chunk_size
        
        # padding到chunk边界
        padded_length = num_chunks * self.chunk_size
        if padded_length > total_tokens:
            pad_length = padded_length - total_tokens
            input_ids = F.pad(input_ids, (0, pad_length), value=0)
            attention_mask = F.pad(attention_mask, (0, pad_length), value=0)
        
        # 分块
        chunked_input_ids = input_ids.view(batch_size, num_chunks, self.chunk_size)
        chunked_attention_mask = attention_mask.view(batch_size, num_chunks, self.chunk_size)
        
        # 获取标签嵌入
        label_embs_for_attention, label_proto_for_contrastive = self.get_label_embeddings()
        
        # 对每个chunk独立处理
        chunk_logits = []
        chunk_text_feats = []
        
        for i in range(num_chunks):
            # 当前chunk的输入
            chunk_input_ids = chunked_input_ids[:, i, :]  # (batch_size, chunk_size)
            chunk_attention_mask = chunked_attention_mask[:, i, :]  # (batch_size, chunk_size)
            
            # 跳过全是padding的chunk
            if chunk_attention_mask.sum() == 0:
                continue
                
            # BERT编码
            chunk_outputs = self.bert_model(
                input_ids=chunk_input_ids,
                attention_mask=chunk_attention_mask
            )
            chunk_hidden = chunk_outputs.last_hidden_state  # (batch_size, chunk_size, hidden_size)
            
            # 在当前chunk上进行标签注意力
            chunk_logit, chunk_text_feat = self.attention(
                chunk_hidden,
                chunk_attention_mask,
                label_embs_for_attention
            )
            
            chunk_logits.append(chunk_logit)  # (batch_size, num_labels)
            chunk_text_feats.append(chunk_text_feat)  # (batch_size, num_labels, hidden_size)
        
        # 聚合各chunk的结果
        if len(chunk_logits) == 0:
            # 所有chunk都是padding，返回零结果
            device = input_ids.device
            final_logits = torch.zeros(batch_size, self.num_labels, device=device)
            final_text_feat = torch.zeros(batch_size, self.num_labels, 
                                        self.bert_model.config.hidden_size, device=device)
        else:
            chunk_logits = torch.stack(chunk_logits, dim=1)  # (batch_size, num_valid_chunks, num_labels)
            chunk_text_feats = torch.stack(chunk_text_feats, dim=1)  # (batch_size, num_valid_chunks, num_labels, hidden_size)
            
            if self.chunk_aggregation == "mean":
                final_logits = chunk_logits.mean(dim=1)
                final_text_feat = chunk_text_feats.mean(dim=1)
            elif self.chunk_aggregation == "max":
                final_logits = chunk_logits.max(dim=1)[0]
                final_text_feat = chunk_text_feats.max(dim=1)[0]
            elif self.chunk_aggregation == "sum":
                final_logits = chunk_logits.sum(dim=1)
                final_text_feat = chunk_text_feats.sum(dim=1)
            elif self.chunk_aggregation == "weighted":
                # 基于文本特征计算权重
                chunk_weights = self.chunk_weight_net(chunk_text_feats.mean(dim=2))  # (batch_size, num_valid_chunks, 1)
                chunk_weights = F.softmax(chunk_weights, dim=1)  # 归一化
                final_logits = (chunk_logits * chunk_weights).sum(dim=1)
                final_text_feat = (chunk_text_feats * chunk_weights.unsqueeze(-1)).sum(dim=1)
            else:
                raise ValueError(f"不支持的聚合方式: {self.chunk_aggregation}")
        
        # 应用投影头
        contrastive_text_feat = self.text_projection_head(final_text_feat)
        contrastive_label_proto = self.label_projection_head(label_proto_for_contrastive)

        return final_logits, contrastive_text_feat, contrastive_label_proto

