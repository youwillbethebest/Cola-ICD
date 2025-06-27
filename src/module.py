import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract

class LabelAttention(nn.Module):
  def __init__(self,attention_head,rep_droupout_num,head_pooling,att_dropout_num,attention_dim,num_labels):
      super().__init__()
      self.attention_dim = attention_dim
      self.classifier = nn.Linear(attention_dim,num_labels)
      self.attention_head = attention_head
      self.rep_dropout = nn.Dropout(rep_droupout_num)
      self.head_pooling = head_pooling
      
      # 添加缺失的 W 线性层
      self.W = nn.Linear(attention_dim, attention_dim)
      
      if self.head_pooling == "concat":
        assert self.attention_dim % self.attention_head == 0
        self.reduce = nn.Linear(self.attention_dim,
                                  self.attention_dim // self.attention_head)
      if att_dropout_num > 0.0:
        self.att_dropout_rate = att_dropout_num
        self.att_dropout = nn.Dropout(self.att_dropout_rate)
        
  def forward(self,input_ids,attention_mask,label_feater):
    m = self.get_label_queried_features(input_ids , attention_mask, label_feater)
    if hasattr(self, 'w_linear'):
            label_feat = self.transform_label_feats(label_feat)
            w = self.w_linear(label_feat) # label * hidden
            b = self.b_linear(label_feat) # label * 1
            logits = self.get_logits(m, w, b)
    else:
            logits = self.get_logits(m)
    return logits
  
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
    z = torch.tanh(self.W(h))
    batch_size, seq_length, att_dim = z.size()
    label_count = label_feat.size(0) // self.attention_head
    u_reshape = label_feat.reshape(label_count, self.attention_head, att_dim)
    score = contract('abd,ecd->aebc', z, u_reshape)
    word_mask = word_mask.bool()
    score = score.masked_fill(mask=~word_mask[:,0:score.shape[-2]].unsqueeze(1).unsqueeze(-1).expand_as(score),
                                      value=float('-1e6'))
    alpha = F.softmax(score, dim=2)
    if hasattr(self,"att_dropout"):
      alpha = self.att_dropout(alpha)
      if self.training:
          alpha_sum = torch.clamp(alpha.sum(dim=2, keepdim=True), 1e-5)
          alpha = alpha / alpha_sum
    m = contract('abd,aebc->aedc', h, alpha)
    if not hasattr(self,"head_pooling") or self.head_pooling == "max":
        m = m.max(-1)[0]
    elif self.head_pooling == "concat":
        m = self.reduce(m.permute(0,1,3,2))
        m=m.reshape(batch_size, -1, att_dim)
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
    def __init__(self, temperature=0.07, use_jaccard=True):
        ...
    def forward(self, text_feat, label_proto, targets):
        # text–label 对比
        # text–text Jaccard 权重对比（可选）
    