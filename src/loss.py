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


