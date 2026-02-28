"""
utils/focal_loss.py — Focal Loss and compatible HuggingFace Trainer subclass.
"""

import torch
from transformers import Trainer

# ── Default hyperparameters (Optuna run-17) ───────────────────────────────────
DEFAULT_GAMMA        = 1.6919871410013687
DEFAULT_FRAUD_WEIGHT = 2.8251219104371517


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for binary classification on imbalanced datasets.

    Focal Loss down-weights easy (well-classified) examples and focuses
    training on hard examples near the decision boundary. With gamma=0
    it reduces to standard weighted cross-entropy.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", 2017.

    Args:
        alpha     : Class weight tensor of shape [num_classes]. If None,
                    all classes are weighted equally.
        gamma     : Focusing parameter. Higher values focus more on hard
                    examples. gamma=0 → weighted CE. Typical range: 1–3.
        reduction : 'mean' or 'sum'.
    """

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw model output of shape [batch_size, num_classes].
            labels: Ground truth class indices of shape [batch_size].

        Returns:
            Scalar loss tensor.
        """
        ce_loss      = torch.nn.functional.cross_entropy(
            logits, labels, weight=self.alpha, reduction="none",
        )
        pt           = torch.exp(-ce_loss)          # probability of correct class
        focal_weight = (1 - pt) ** self.gamma
        loss         = focal_weight * ce_loss
        return loss.mean() if self.reduction == "mean" else loss.sum()


class FocalLossTrainer(Trainer):
    """
    HuggingFace Trainer subclass that uses FocalLoss instead of
    standard cross-entropy.

    Reads `focal_gamma` and `fraud_class_weight` from `self.args` at
    every forward pass so Optuna can inject different values per trial
    without reinstantiating the trainer.

    Falls back to DEFAULT_GAMMA and DEFAULT_FRAUD_WEIGHT if the
    attributes are not set on self.args (e.g. standalone training).

    The legitimate class weight is read from `self.class_weights[0]`,
    which must be set on the trainer instance after construction:

        trainer = FocalLossTrainer(...)
        trainer.class_weights = class_weights  # numpy array [legit_w, fraud_w]
    """

    def compute_loss(
        self,
        model,
        inputs: dict,
        return_outputs: bool = False,
        **kwargs,
    ):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        gamma              = getattr(self.args, "focal_gamma",        DEFAULT_GAMMA)
        fraud_class_weight = getattr(self.args, "fraud_class_weight", DEFAULT_FRAUD_WEIGHT)

        # Rebuild alpha tensor on the correct device every forward pass
        alpha = torch.tensor(
            [self.class_weights[0], fraud_class_weight],
            dtype=torch.float,
        ).to(logits.device)

        loss = FocalLoss(alpha=alpha, gamma=gamma)(logits, labels)
        return (loss, outputs) if return_outputs else loss
