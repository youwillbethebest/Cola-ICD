import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossWithLogits(nn.Module):
    """
    Binary Focal Loss with logits for multi-label classification.
    FL = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t = sigmoid(x) if y=1, 1-sigmoid(x) if y=0.
    """

    def __init__(self, gamma: float = 0.5, alpha: float = 0.25, reduction: str = "mean"):
        """
        Args:
            gamma: focusing parameter >= 0. (gamma=0 => reduces to BCE)
            alpha: balance factor in [0,1]. Use scalar or tensor of shape [num_classes].
            reduction: "none" | "mean" | "sum"
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits: raw model outputs (before sigmoid), shape (B, C)
            targets: binary labels (0 or 1), same shape as logits
        Returns:
            focal loss (scalar if reduction != "none", else tensor of shape (B, C))
        """
        # BCE loss with logits, without reduction
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        # Convert logits to probabilities
        prob = torch.sigmoid(logits)
        # p_t: prob if label=1, 1-prob if label=0
        p_t = prob * targets + (1 - prob) * (1 - targets)
        # modulating factor
        mod_factor = (1 - p_t) ** self.gamma
        # alpha factor
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        # focal loss
        loss = alpha_factor * mod_factor * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss



class HierarchyConsistencyLoss(nn.Module):
    """
    有向 parent→child 的层级一致性约束：
      L = mean(ReLU(margin + s_child - s_parent))
    默认在 logits 空间计算（更稳）。
    """
    def __init__(self, margin: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, edges_pc: torch.LongTensor, weight: torch.Tensor = None) -> torch.Tensor:
        # logits: (B, C); edges_pc: (2, E) with parent->child
        if edges_pc is None or edges_pc.numel() == 0:
            return torch.tensor(0., device=logits.device)
        parent_idx = edges_pc[0]
        child_idx = edges_pc[1]
        s_parent = logits[:, parent_idx]  # (B, E)
        s_child = logits[:, child_idx]    # (B, E)
        viol = torch.relu(self.margin + s_child - s_parent)  # (B, E)
        if weight is not None:
            viol = viol * weight  # broadcast (E,) or (B,E)
        if self.reduction == "mean":
            return viol.mean()
        elif self.reduction == "sum":
            return viol.sum()
        else:
            return viol  # (B, E)

