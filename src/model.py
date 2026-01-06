import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModel
from src.module import LabelAttention, GraphAttentionLayer, LabelGNN, LabelAttentionV2, HierLabelGNN
from src.data_loader import SynonymLabelLoader
import torch.nn.functional as F


class ClinicalLongformerLabelAttention(nn.Module):
    """
    Clinical-Longformer backbone + Label Attention multi-label classification model.
    Args:
      longformer_path: Local Clinical Longformer model path
      codes_file: ICD code and description file path
      label_model_name: Pretrained model name for loading label descriptions, default Bio_ClinicalBERT
      use_gnn: Whether to use GNN to update label embeddings
      adj_matrix: Label co-occurrence adjacency matrix
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
        # Text encoder
        self.model = AutoModel.from_pretrained(longformer_path,add_pooling_layer=False)
        hidden_size = self.model.config.hidden_size

        # ---------------- Label loading ----------------
        if label_loader is None:          # If not provided externally, create internally (maintain compatibility)
            assert codes_file is not None, "codes_file must be specified when label_loader is not provided"
            label_loader = SynonymLabelLoader(
                codes_file=codes_file,
                pretrained_model_name=label_model_name,
                term_count=term_counts
            )

        self.num_labels = label_loader.num_labels
        label_embs = label_loader()       # Pre-compute label embeddings
        #self.register_buffer("label_embs", label_embs)
        self.label_embs = nn.Parameter(label_embs)
        self.term_counts = term_counts  # Save term_counts
        # ---------------- GNN module (optional) ----------------
        self.use_gnn = use_gnn
        self.hier_mode = False
        if self.use_gnn:
            assert adj_matrix is not None, "adj_matrix must be provided when use_gnn=True."
            # Support two types: co-occurrence graph (edge_index, edge_weight) or hierarchy graph {'up':(...), 'down':(...)}
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
                # Register as buffer
                self.register_buffer("edge_index", edge_index)
                self.register_buffer("edge_weight", edge_weight)
                # GNN module internally has two GCNConv layers
                self.gnn = LabelGNN(hidden_size, hidden_size // 2, hidden_size)
            
            # Can stack multiple GAT layers
            # Or use just one layer
            # self.gat_layer = GraphAttentionLayer(hidden_size, hidden_size, dropout=0.2, alpha=0.2, concat=False)


        # Label-aware attention
        self.attention = LabelAttention(
            attention_head=term_counts,
            rep_droupout_num=0.2,
            head_pooling="concat",
            att_dropout_num=0.1,
            attention_dim=hidden_size,
            num_labels=self.num_labels,
            est_cls=1
        )
        # Projection head
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
        Get label embeddings, use graph attention network for update if GNN is enabled
        
        Returns:
            label_embs_for_attention: (C*k, H) - Label embeddings for attention mechanism
            label_proto_for_contrastive: (C, H) - Label prototypes for contrastive learning
        """
        # Original label embeddings are already in (C*k, H) form
        raw_embs = self.label_embs  # (C*k, H)
        
        if self.use_gnn:
            if self.hier_mode:
                # Code-level graph: first aggregate terms to code-level, then graph update, then broadcast back to term positions
                code_pool = raw_embs.view(self.num_labels, self.term_counts, -1).max(dim=1)[0]  # (C, H)
                code_updated = self.gnn_hier(
                    code_pool,
                    self.up_edge_index, self.up_edge_weight,
                    self.down_edge_index, self.down_edge_weight
                )  # (C, H)
                label_embs_for_attention = code_updated.repeat_interleave(self.term_counts, dim=0)  # (C*k, H)
                label_proto_for_contrastive = code_updated  # (C, H)
            else:
                # Pass sparse representation (co-occurrence graph) when calling GNN
                updated_embs = self.gnn(raw_embs, self.edge_index, self.edge_weight)  # (C*k, H)
                label_embs_for_attention = updated_embs  # (C*k, H)
                label_proto_for_contrastive = updated_embs.view(
                    self.num_labels, self.term_counts, -1
                ).max(dim=1)[0]  # (C, H)
            
        else:
            # If not using GNN, directly use original embeddings
            label_embs_for_attention = raw_embs  # (C*k, H)
            
            # For contrastive learning, take max pooling over k synonyms for each label
            label_proto_for_contrastive = raw_embs.view(
                self.num_labels, self.term_counts, -1
            ).max(dim=1)[0]  # (C, H)

        return label_embs_for_attention, label_proto_for_contrastive
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> dict:
        # Text encoding
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        text_hidden = outputs.last_hidden_state  # (N, L, H)
        
        # Mean Pooling for text_feat, using attention_mask to ignore padding
        # attention_mask_expanded = attention_mask.unsqueeze(-1)
        # sum_embeddings = torch.sum(text_hidden * attention_mask_expanded, dim=1)
        # sum_mask = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9)
        # text_feat = sum_embeddings / sum_mask
        label_embs_for_attention, label_proto_for_contrastive = self.get_label_embeddings()
        # Label attention generates logits
        logits, per_label_text_feat = self.attention(text_hidden, attention_mask, label_embs_for_attention)

        # Apply projection head to generate isolated features for contrastive learning
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
    BERT-based text chunking attention model
    
    Processing flow:
    1. Text chunking: (batch_size, total_tokens) → (batch_size, num_chunks, chunk_size)
    2. BERT independent encoding: each chunk through BERT → (batch_size, num_chunks, chunk_size, hidden_size)
    3. Reorganize: concatenate embeddings from all chunks → (batch_size, total_tokens, hidden_size)
    4. Label attention: perform attention computation on the concatenated complete sequence
    
    Args:
        bert_model_path: BERT model path
        chunk_size: Number of tokens per chunk, default 128
        codes_file: ICD code and description file path
        label_model_name: Pretrained model name for label descriptions
        term_counts: Number of synonyms per label
        label_loader: Externally provided label loader
        use_gnn: Whether to use GNN to update label embeddings
        adj_matrix: Label co-occurrence adjacency matrix
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
        
        # BERT encoder
        self.bert_model = AutoModel.from_pretrained(bert_model_path,add_pooling_layer=False)
        self.chunk_size = chunk_size
        hidden_size = self.bert_model.config.hidden_size
        
        # ---------------- Label loading ----------------
        if label_loader is None:
            assert codes_file is not None, "codes_file must be specified when label_loader is not provided"
            label_loader = SynonymLabelLoader(
                codes_file=codes_file,
                pretrained_model_name=label_model_name,
                term_count=term_counts
            )

        self.num_labels = label_loader.num_labels
        label_embs = label_loader()
        self.label_embs = nn.Parameter(label_embs)
        self.term_counts = term_counts
        
        # ---------------- GNN module (optional) ----------------
        self.use_gnn = use_gnn
        self.hier_mode = False
        if self.use_gnn:
            assert adj_matrix is not None, "adj_matrix must be provided when use_gnn=True."
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

        # Label-aware attention
        self.attention = LabelAttention(
            attention_head=term_counts,
            rep_droupout_num=0.2,
            head_pooling="concat",
            att_dropout_num=0.1,
            attention_dim=hidden_size,
            num_labels=self.num_labels,
            est_cls=1
        )
        
        # Projection head
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
        Chunk text and perform BERT encoding
        
        Args:
            input_ids: (batch_size, total_tokens)
            attention_mask: (batch_size, total_tokens)
            
        Returns:
            chunked_embeddings: (batch_size, total_tokens, hidden_size)
            final_attention_mask: (batch_size, total_tokens)
        """
        batch_size, total_tokens = input_ids.shape
        
        # Calculate number of chunks needed
        num_chunks = (total_tokens + self.chunk_size - 1) // self.chunk_size
        
        # If total length is not divisible by chunk_size, need padding
        padded_length = num_chunks * self.chunk_size
        if padded_length > total_tokens:
            # Use pad_token_id for padding, usually 0
            pad_length = padded_length - total_tokens
            input_ids = F.pad(input_ids, (0, pad_length), value=0)
            attention_mask = F.pad(attention_mask, (0, pad_length), value=0)
        
        # Step 1: Chunk and reshape
        # (batch_size, total_tokens) → (batch_size, num_chunks, chunk_size)
        chunked_input_ids = input_ids.view(batch_size, num_chunks, self.chunk_size)
        chunked_attention_mask = attention_mask.view(batch_size, num_chunks, self.chunk_size)
        
        # Step 2: BERT independent encoding
        # Flatten to (batch_size * num_chunks, chunk_size)
        flat_input_ids = chunked_input_ids.view(-1, self.chunk_size)
        flat_attention_mask = chunked_attention_mask.view(-1, self.chunk_size)
        
        # Each chunk independently encoded through BERT
        with torch.cuda.device(flat_input_ids.device):
            chunk_outputs = self.bert_model(
                input_ids=flat_input_ids,
                attention_mask=flat_attention_mask
            )
        chunk_embeddings = chunk_outputs.last_hidden_state  # (batch_size * num_chunks, chunk_size, hidden_size)
        
        # Step 3: Reorganize embeddings
        # Reshape to (batch_size, num_chunks, chunk_size, hidden_size)
        chunk_embeddings = chunk_embeddings.view(batch_size, num_chunks, self.chunk_size, -1)
        # Concatenate all chunks: (batch_size, num_chunks * chunk_size, hidden_size)
        concatenated_embeddings = chunk_embeddings.view(batch_size, -1, chunk_embeddings.size(-1))
        
        # Similarly process attention_mask
        final_attention_mask = chunked_attention_mask.view(batch_size, -1)
        
        # Truncate to original length
        if padded_length > total_tokens:
            concatenated_embeddings = concatenated_embeddings[:, :total_tokens, :]
            final_attention_mask = final_attention_mask[:, :total_tokens]
        
        return concatenated_embeddings, final_attention_mask

    def get_label_embeddings(self):
        """Get label embeddings, consistent with original model"""
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
        Forward pass
        
        Args:
            input_ids: (batch_size, total_tokens) e.g. (1, 2000)
            attention_mask: (batch_size, total_tokens)
            
        Returns:
            logits: Classification logits
            contrastive_text_feat: Text features for contrastive learning
            contrastive_label_proto: Label prototypes for contrastive learning
        """
        # Text chunking and encoding
        text_hidden, processed_attention_mask = self.chunk_and_encode(input_ids, attention_mask)
        # text_hidden: (batch_size, total_tokens, hidden_size)
        # processed_attention_mask: (batch_size, total_tokens)
        
        # Get label embeddings
        label_embs_for_attention, label_proto_for_contrastive = self.get_label_embeddings()
        
        # Label attention generates logits
        logits, per_label_text_feat = self.attention(
            text_hidden, 
            processed_attention_mask, 
            label_embs_for_attention
        )
        
        # Apply projection head to generate isolated features for contrastive learning
        contrastive_text_feat = self.text_projection_head(per_label_text_feat)
        contrastive_label_proto = self.label_projection_head(label_proto_for_contrastive)

        return logits, contrastive_text_feat, contrastive_label_proto


class ClinicalBERTChunkAttentionV2(nn.Module):
    """
    BERT-based text chunking attention model - Version 2
    
    Processing flow (attention first, then concatenate):
    1. Text chunking: (batch_size, total_tokens) → (batch_size, num_chunks, chunk_size)
    2. BERT independent encoding: each chunk through BERT → (batch_size, num_chunks, chunk_size, hidden_size)
    3. Label attention within each chunk
    4. Concatenate attention results from all chunks
    
    This version performs attention computation independently within each chunk, may be more suitable for long text processing
    """
    
    def __init__(self,
                 bert_model_path: str,
                 chunk_size: int = 512,
                 overlap: int = 0,
                 codes_file: str | None = None,
                 label_model_name: str = "Bio_ClinicalBERT",
                 term_counts: int = 1,
                 label_loader: Optional[SynonymLabelLoader] = None,
                 use_gnn: bool = False,
                 adj_matrix: Optional[torch.Tensor] = None,
                 chunk_aggregation: str = "max"):  # Chunk result aggregation method
        super().__init__()
        
        # BERT encoder
        self.bert_model = AutoModel.from_pretrained(bert_model_path,add_pooling_layer=False)
        self.chunk_size = chunk_size
        self.overlap = max(0, int(overlap))
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        self.stride = self.chunk_size - self.overlap
        self.chunk_aggregation = chunk_aggregation  # "mean", "max", "sum", "weighted"
        hidden_size = self.bert_model.config.hidden_size
        
        # Label loading (same as V1)
        if label_loader is None:
            assert codes_file is not None, "codes_file must be specified when label_loader is not provided"
            label_loader = SynonymLabelLoader(
                codes_file=codes_file,
                pretrained_model_name=label_model_name,
                term_count=term_counts
            )

        self.num_labels = label_loader.num_labels
        label_embs = label_loader()
        self.label_embs = nn.Parameter(label_embs)
        self.term_counts = term_counts
        
        # GNN module (same as V1)
        self.use_gnn = use_gnn
        self.hier_mode = False
        if self.use_gnn:
            assert adj_matrix is not None, "adj_matrix must be provided when use_gnn=True."
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

        # Label-aware attention
        self.attention = LabelAttention(
            attention_head=term_counts,
            rep_droupout_num=0.1,
            head_pooling="concat",
            att_dropout_num=0.1,
            attention_dim=hidden_size,
            num_labels=self.num_labels,
            est_cls=1
        )
        
        # Chunk weight network (for weighted aggregation)
        if self.chunk_aggregation == "weighted":
            self.chunk_weight_net = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid()
            )
        
        # Projection head
        projection_dim = 256
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim)
        )
        self.text_projection_head = self.shared_head
        self.label_projection_head = self.shared_head

    def get_label_embeddings(self):
        """Get label embeddings, same as V1"""
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
        Forward pass - V2 version (attention first, then concatenate)
        """
        batch_size, total_tokens = input_ids.shape
        
        # Calculate stride and required padding to ensure sliding window covers to the end
        stride = self.stride  # = chunk_size - overlap
        if stride <= 0:
            raise ValueError("stride must be positive. Ensure overlap < chunk_size.")

        if self.overlap == 0:
            # Keep original whole chunk splitting logic
            num_chunks = (total_tokens + self.chunk_size - 1) // self.chunk_size
            padded_length = num_chunks * self.chunk_size
            if padded_length > total_tokens:
                pad_length = padded_length - total_tokens
                input_ids = F.pad(input_ids, (0, pad_length), value=0)
                attention_mask = F.pad(attention_mask, (0, pad_length), value=0)
            chunked_input_ids = input_ids.view(batch_size, num_chunks, self.chunk_size)
            chunked_attention_mask = attention_mask.view(batch_size, num_chunks, self.chunk_size)
        else:
            # Calculate padding needed to ensure last window start doesn't exceed total_tokens-1
            # Make (num_chunks-1)*stride + chunk_size >= total_tokens
            num_chunks = (max(0, total_tokens - self.chunk_size) + stride) // stride + 1
            effective_length = (num_chunks - 1) * stride + self.chunk_size
            if effective_length > total_tokens:
                pad_length = effective_length - total_tokens
                input_ids = F.pad(input_ids, (0, pad_length), value=0)
                attention_mask = F.pad(attention_mask, (0, pad_length), value=0)
            # Use unfold to generate sliding window chunks
            # (N, L) -> (N, num_chunks, chunk_size)
            chunked_input_ids = input_ids.unfold(dimension=1, size=self.chunk_size, step=stride)
            chunked_attention_mask = attention_mask.unfold(dimension=1, size=self.chunk_size, step=stride)
            # unfold returns (N, num_chunks, chunk_size)
        
        # Get label embeddings
        label_embs_for_attention, label_proto_for_contrastive = self.get_label_embeddings()
        
        # Process each chunk independently
        chunk_logits = []
        chunk_text_feats = []
        
        for i in range(num_chunks):
            # Current chunk input
            chunk_input_ids = chunked_input_ids[:, i, :]  # (batch_size, chunk_size)
            chunk_attention_mask = chunked_attention_mask[:, i, :]  # (batch_size, chunk_size)
            
            # Skip chunks that are all padding
            if chunk_attention_mask.sum() == 0:
                continue
                
            # BERT encoding
            chunk_outputs = self.bert_model(
                input_ids=chunk_input_ids,
                attention_mask=chunk_attention_mask
            )
            chunk_hidden = chunk_outputs.last_hidden_state  # (batch_size, chunk_size, hidden_size)
            
            # Perform label attention on current chunk
            chunk_logit, chunk_text_feat = self.attention(
                chunk_hidden,
                chunk_attention_mask,
                label_embs_for_attention
            )
            
            chunk_logits.append(chunk_logit)  # (batch_size, num_labels)
            chunk_text_feats.append(chunk_text_feat)  # (batch_size, num_labels, hidden_size)
        
        # Aggregate results from all chunks
        if len(chunk_logits) == 0:
            # All chunks are padding, return zero results
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
                # Compute weights based on text features
                chunk_weights = self.chunk_weight_net(chunk_text_feats.mean(dim=2))  # (batch_size, num_valid_chunks, 1)
                chunk_weights = F.softmax(chunk_weights, dim=1)  # Normalize
                final_logits = (chunk_logits * chunk_weights).sum(dim=1)
                final_text_feat = (chunk_text_feats * chunk_weights.unsqueeze(-1)).sum(dim=1)
            else:
                raise ValueError(f"Unsupported aggregation method: {self.chunk_aggregation}")
        
        # Apply projection head
        contrastive_text_feat = self.text_projection_head(final_text_feat)
        contrastive_label_proto = self.label_projection_head(label_proto_for_contrastive)

        return final_logits, contrastive_text_feat, contrastive_label_proto

