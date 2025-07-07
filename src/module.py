import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from torch_geometric.nn import GCNConv
from typing import Tuple


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.0, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.dropout = dropout
        self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(self.num_layers - 1))
        if act == "relu":
            self.act_fn = F.relu
        elif act == "gelu":
            self.act_fn = F.gelu

    def forward(self, x):
        if not hasattr(self, 'act_fn'):
            self.act_fn = F.relu
        for i, layer in enumerate(self.layers):
            x = self.act_fn(layer(x)) if i < self.num_layers - 1 else layer(x)
            if hasattr(self, 'dropouts') and i < self.num_layers - 1:
                x = self.dropouts[i](x)
        return x
class LabelAttention(nn.Module):
  def __init__(self,attention_head,rep_droupout_num,head_pooling,att_dropout_num,attention_dim,num_labels,est_cls:int=0):
      super().__init__()
      self.attention_dim = attention_dim
      self.classifier = nn.Linear(attention_dim,num_labels)
      self.attention_head = attention_head
      self.rep_dropout = nn.Dropout(rep_droupout_num)
      self.head_pooling = head_pooling
      self.est_cls = est_cls
      if self.est_cls > 0:
        self.w_linear = MLP(self.attention_dim, self.attention_dim, self.attention_dim, self.est_cls)
        self.b_linear = MLP(self.attention_dim, self.attention_dim, 1, self.est_cls)

      
      # 添加缺失的 W 线性层
      self.W = nn.Linear(attention_dim, attention_dim)
      
      if self.head_pooling == "concat":
        assert self.attention_dim % self.attention_head == 0
        self.reduce = nn.Linear(self.attention_dim,
                                  self.attention_dim // self.attention_head)
      if att_dropout_num > 0.0:
        self.att_dropout_rate = att_dropout_num
        self.att_dropout = nn.Dropout(self.att_dropout_rate)
        
  def transform_label_feats(self, label_feat):
        if not hasattr(self, 'head_pooling') or self.head_pooling == "max":
            label_count = label_feat.shape[0] // self.attention_head
            label_feat = label_feat.reshape(label_count, self.attention_head, -1).max(1)[0]
        elif self.head_pooling == "concat":
            label_count = label_feat.shape[0] // self.attention_head
            label_feat = self.reduce(label_feat) # (label * head) * (hidden // head)
            label_feat = label_feat.reshape(label_count, self.attention_head, -1) # label * head * hidden
            label_feat = label_feat.reshape(label_count, -1)
        return label_feat
        
  def forward(self,input_ids,attention_mask,label_feat):
    m = self.get_label_queried_features(input_ids , attention_mask, label_feat)
    if hasattr(self, 'w_linear'):
            label_feat = self.transform_label_feats(label_feat)
            w = self.w_linear(label_feat) # label * hidden
            b = self.b_linear(label_feat) # label * 1
            logits = self.get_logits(m, w, b)
    else:
            logits = self.get_logits(m)
    return logits, m
  
  def get_logits(self, m, w=None, b=None):
        # logits = self.final(m).squeeze(-1)
        # m: batch * label * hidden
        if w is None:
            # logits = self.final(m).squeeze(-1)
            logits = self.classifier.weight.mul(m).sum(dim=2).add(self.classifier.bias)
        else:
            logits = contract('blh,lh->bl', m, w) + b.squeeze(-1)
        return logits
    


  def get_label_queried_features(self,h,word_mask,label_feat):
    # 输入: h [batch_size, seq_length, hidden_dim]
    # 输出: z [batch_size, seq_length, attention_dim]
    z = torch.tanh(self.W(h))
    
    batch_size, seq_length, att_dim = z.size()
    # 计算标签数量
    label_count = label_feat.size(0) // self.attention_head
    
    # 输入: label_feat [label_count * attention_head, attention_dim]
    # 输出: u_reshape [label_count, attention_head, attention_dim]
    u_reshape = label_feat.reshape(label_count, self.attention_head, att_dim)
    
    # 计算注意力分数
    # 输入: z [batch_size, seq_length, attention_dim], u_reshape [label_count, attention_head, attention_dim]
    # 输出: score [batch_size, label_count, seq_length, attention_head]
    score = contract('abd,ecd->aebc', z, u_reshape)
    
    # 处理mask
    word_mask = word_mask.bool()
    # 输入: word_mask [batch_size, seq_length]
    # 输出: score [batch_size, label_count, seq_length, attention_head] (masked)
    score = score.masked_fill(mask=~word_mask[:,0:score.shape[-2]].unsqueeze(1).unsqueeze(-1).expand_as(score),
                                      value=float('-1e6'))
    
    # 输入: score [batch_size, label_count, seq_length, attention_head]
    # 输出: alpha [batch_size, label_count, seq_length, attention_head]
    alpha = F.softmax(score, dim=2)
    
    if hasattr(self,"att_dropout"):
      # 应用注意力dropout
      alpha = self.att_dropout(alpha)
      if self.training:
          # 重新归一化
          alpha_sum = torch.clamp(alpha.sum(dim=2, keepdim=True), 1e-5)
          alpha = alpha / alpha_sum
    
    # 加权聚合特征
    # 输入: h [batch_size, seq_length, hidden_dim], alpha [batch_size, label_count, seq_length, attention_head]
    # 输出: m [batch_size, label_count, attention_head, hidden_dim//attention_head]
    m = contract('abd,aebc->aedc', h, alpha)
    
    if not hasattr(self,"head_pooling") or self.head_pooling == "max":
        # max pooling: 在attention_head维度上取最大值
        # 输入: m [batch_size, label_count, attention_head, hidden_dim//attention_head]
        # 输出: m [batch_size, label_count, hidden_dim//attention_head]
        m = m.max(-1)[0]
    elif self.head_pooling == "concat":
        # concat: 拼接所有head的特征
        # 输入: m [batch_size, label_count, attention_head, hidden_dim//attention_head]
        # 输出: m [batch_size, label_count, attention_dim]
        m = self.reduce(m.permute(0,1,3,2))
        m=m.reshape(batch_size, -1, att_dim)
    
    # 应用dropout
    # 输入: m [batch_size, label_count, hidden_dim//attention_head] 或 [batch_size, label_count, attention_dim]
    # 输出: m [batch_size, label_count, hidden_dim//attention_head] 或 [batch_size, label_count, attention_dim]
    m = self.rep_dropout(m)
    return m
class LabelAttentionV2(LabelAttention):
    def __init__(self,attention_head,rep_droupout_num,head_pooling,att_dropout_num,attention_dim,num_labels,est_cls:int=1):
        super().__init__(attention_head,rep_droupout_num,head_pooling,att_dropout_num,attention_dim,num_labels,est_cls)

        # 输入: attention_dim
        # 输出: attention_dim // attention_head
        self.u_reduce = nn.Linear(self.attention_dim,
                                    self.attention_dim // self.attention_head)

    def get_label_queried_features(self,h,word_mask,label_feat):
        z = torch.tanh(self.W(h))
        batch_size, seq_length, att_dim = z.size()
        z_reshape = z.reshape(batch_size, seq_length, self.attention_head, att_dim // self.attention_head)
        label_count = label_feat.size(0) // self.attention_head
        u_reshape = self.u_reduce(label_feat.reshape(label_count, self.attention_head, att_dim))
        score = contract('abcd,ecd->aebc', z_reshape, u_reshape)
        word_mask = word_mask.bool()
        score = score.masked_fill(mask=~word_mask[:,0:score.shape[-2]].unsqueeze(1).unsqueeze(-1).expand_as(score),
                                      value=float('-1e4'))
        alpha = F.softmax(score, dim=2)
        if hasattr(self, 'att_dropout'):
            alpha = self.att_dropout(alpha)
            if self.training:
                alpha *= (1 - self.att_dropout_rate)
        m = contract('abd,aebc->aedc', h, alpha)
        if not hasattr(self, 'head_pooling') or self.head_pooling == "max":
            m = m.max(-1)[0]
        elif self.head_pooling == "concat":
            m = self.reduce(m.permute(0,1,3,2))
            m = m.reshape(batch_size, -1, att_dim)
        m = self.rep_dropout(m)
        return m

class GraphAttentionLayer(nn.Module):
  def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
      super(GraphAttentionLayer, self).__init__()
      self.in_features = in_features
      self.out_features = out_features
      self.dropout = dropout
      self.alpha = alpha
      self.concat = concat

      # Linear transformation weight
      self.W = nn.Parameter(torch.Tensor(in_features, out_features))
      nn.init.xavier_uniform_(self.W.data, gain=1.414)

      # Attention mechanism weight vector
      self.a = nn.Parameter(torch.Tensor(2 * out_features, 1))
      nn.init.xavier_uniform_(self.a.data, gain=1.414)

      self.leakyrelu = nn.LeakyReLU(self.alpha)

  def forward(self, h, adj):
      # h: [N, in_features]
      # adj: [N, N], binary adjacency matrix (0/1)
      Wh = torch.matmul(h, self.W)  # [N, out_features]
      N = Wh.size(0)

      # Prepare attention input by concatenating Wh_i || Wh_j for all i,j
      Wh_repeat_i = Wh.repeat_interleave(N, dim=0)        # [N*N, out_features]
      Wh_repeat_j = Wh.repeat(N, 1)                       # [N*N, out_features]
      a_input = torch.cat([Wh_repeat_i, Wh_repeat_j], dim=1)  # [N*N, 2*out_features]
      a_input = a_input.view(N, N, 2 * self.out_features)    # [N, N, 2*out_features]

      # Compute attention coefficients
      e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [N, N]
      # Masked attention: set entries where adj=0 to -inf
      zero_vec = -9e15 * torch.ones_like(e)
      attention = torch.where(adj > 0, e, zero_vec)

      # Softmax normalization
      attention = F.softmax(attention, dim=1)        # [N, N]
      attention = F.dropout(attention, self.dropout, training=self.training)

      # Weighted sum of neighbor features
      h_prime = torch.matmul(attention, Wh)  # [N, out_features]

      if self.concat:
          # Apply nonlinearity for intermediate layers
          return F.elu(h_prime)
      else:
          # Last layer: no nonlinearity
          return h_prime
    
class JaccardWeightedSupConLoss(nn.Module):
    """
    文本-标签 & 文本-文本 对比学习损失

    Args:
        temperature: 对比温度 τ
        eps:         防止除零
        use_text_text: 是否启用文本-文本分支
    """
    def __init__(self,
                 temperature: float = 0.07,
                 eps: float = 1e-12,
                 use_text_text: bool = False):
        super().__init__()
        self.tau = temperature
        self.eps = eps
        self.use_text_text = use_text_text

    def forward(
        self,
        text_feat: torch.Tensor,    # (B, H)
        label_proto: torch.Tensor,  # (C, H)
        targets: torch.Tensor       # (B, C) multi-hot
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H = text_feat.shape
        device = text_feat.device

        # ---- 文本-标签分支 ----
        # 1) 归一化
        t = F.normalize(text_feat, dim=1)      # (B, H)
        l = F.normalize(label_proto, dim=1)    # (C, H)
        # 2) 相似度 & exp
        sim_tl = torch.matmul(t, l.t()) / self.tau  # (B, C)
        exp_tl = torch.exp(sim_tl)

        # 3) 正样本 mask & log-prob
        pos_mask = targets.bool()                   # (B, C)
        denom_tl = exp_tl.sum(dim=1, keepdim=True) + self.eps
        logp_pos = torch.log(exp_tl + self.eps) - torch.log(denom_tl)
        # 4) 按每个样本的正标签数归一化
        pos_count = pos_mask.sum(dim=1).clamp(min=1).float()
        loss_tl = - (logp_pos * pos_mask).sum(dim=1) / pos_count
        loss_text_label = loss_tl.mean()

        # ---- 文本-文本 Jaccard-Weighted SupCon ----
        if not self.use_text_text:
            return loss_text_label, torch.tensor(0., device=device)

        # 1) 同样归一化 & 计算相似度
        sim_tt = torch.matmul(t, t.t()) / self.tau  # (B, B)
        # 2) mask 自身对角
        diag = torch.eye(B, device=device, dtype=torch.bool)
        sim_tt = sim_tt.masked_fill(diag, float('-inf'))
        # 3) 用 logsumexp 计算 log-prob
        log_denom = torch.logsumexp(sim_tt, dim=1, keepdim=True)  # (B,1)
        logp_tt = sim_tt - log_denom                              # (B, B)

        # 4) 计算 Jaccard 权重
        t_bool = targets.bool()
        # intersection & union
        inter = (t_bool.unsqueeze(1) & t_bool.unsqueeze(0)).sum(-1).float()  # (B,B)
        union = (t_bool.unsqueeze(1) | t_bool.unsqueeze(0)).sum(-1).float()  # (B,B)
        jacc = inter / (union + self.eps)
        jacc = jacc.masked_fill(diag, 0.0)

        # 5) 归一化权重 & detach
        weight = jacc / (jacc.sum(dim=1, keepdim=True) + self.eps)
        weight = weight.detach()

        # 6) 加权 loss & 取有效行平均
        loss_row = -(weight * logp_tt).sum(dim=1)  # (B,)
        valid = weight.sum(dim=1) > 0
        if valid.any():
            loss_text_text = loss_row[valid].mean()
        else:
            loss_text_text = torch.tensor(0., device=device)

        return loss_text_label, loss_text_text

class LabelWiseContrastiveLoss(nn.Module):
    """
    标签感知的对比学习损失 (Label-Wise Contrastive Loss)
    
    对于每一个正标签，将其对应的文本表征作为"锚点"，
    其自身的标签原型作为"正样本"，而该样本的所有"负标签"对应的原型作为"负样本"。
    """
    def __init__(self, temperature: float = 0.1, eps: float = 1e-12, neg_samples: int = None):
        super().__init__()
        self.tau = temperature
        self.eps = eps
        self.neg_samples = neg_samples

    def forward(self, 
                per_label_text_feat: torch.Tensor,  # (B, C, H)
                label_proto: torch.Tensor,          # (C, H)
                targets: torch.Tensor               # (B, C) multi-hot
               ) -> torch.Tensor:
        
        # 归一化特征
        per_label_text_feat = F.normalize(per_label_text_feat, dim=-1)
        label_proto = F.normalize(label_proto, dim=-1)

        # 创建正负样本的掩码
        pos_mask = targets.bool() # (B, C)
        
        # 如果批次内没有正样本，则损失为0
        if not pos_mask.any():
            return torch.tensor(0., device=per_label_text_feat.device)
            
        # 计算每个"标签定制的文本表征"与"所有标签原型"之间的相似度
        # (B, C, H) x (H, C) -> (B, C, C)
        # sim_matrix[b, i, j] = sim(text_feat_for_label_i, proto_for_label_j)
        sim_matrix = torch.matmul(per_label_text_feat, label_proto.t()) / self.tau

        # --- 核心修改：为每个 anchor 构建正确的正负样本集 ---

        # 1. 提取正样本相似度
        # 对于 anchor (b, i), 正样本是 label_proto i.
        # 相似度在 sim_matrix 的对角线上
        pos_sim = torch.diagonal(sim_matrix, offset=0, dim1=-2, dim2=-1) # (B, C)

        # 2. 提取负样本相似度
        neg_mask = ~pos_mask  # (B, C)
        if self.neg_samples is not None and self.neg_samples > 0:
            B, C, _ = sim_matrix.size()
            K = min(self.neg_samples, C - 1)
            neg_indices = []
            for b in range(B):
                neg_b = torch.where(neg_mask[b])[0]
                perm = neg_b[torch.randperm(neg_b.size(0), device=neg_b.device)]
                chosen = perm[:K]
                neg_indices.append(chosen.unsqueeze(0).repeat(C, 1))
            neg_indices = torch.stack(neg_indices, dim=0)  # (B, C, K)
            # 采样后的负相似度
            neg_sim_matrix = torch.gather(sim_matrix, 2, neg_indices)
        else:
            # 原先逻辑: 保留所有负样本
            neg_sim_matrix = sim_matrix.masked_fill(
                neg_mask.unsqueeze(1).unsqueeze(-1).expand_as(sim_matrix),
                float('-inf')
            )
        
        # 3. 计算 InfoNCE loss
        log_sum_exp_neg = torch.logsumexp(neg_sim_matrix, dim=-1) # (B, C)
        
        # 分子是正样本相似度, 分母是 (正样本 + 所有负样本)
        # log(P) = log(exp(pos) / (exp(pos) + sum(exp(neg))))
        #        = pos_sim - log(exp(pos_sim) + exp(log_sum_exp_neg))
        #        = pos_sim - logaddexp(pos_sim, log_sum_exp_neg)
        log_prob = pos_sim - torch.logaddexp(pos_sim, log_sum_exp_neg)

        # 4. 计算最终损失
        # 只对存在的正样本(anchor)计算损失
        loss = -log_prob * pos_mask
        
        # 按每个样本的正标签数量进行归一化
        pos_count = pos_mask.sum(dim=1).clamp(min=1).float()
        loss = loss.sum(dim=1) / pos_count
        
        return loss.mean()
class LabelGNN(nn.Module):
    """
    用于增强标签原型的 GCN 模块。
    将共现图信息注入到原型表示中。
    """
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor,
                edge_index: torch.LongTensor,
                edge_weight: torch.FloatTensor = None) -> torch.Tensor:
        # 第一层 GCN + 激活
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # 第二层 GCN 输出增强后的原型
        x = self.conv2(x, edge_index, edge_weight)
        return x