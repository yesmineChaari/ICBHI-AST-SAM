"""
Changes :
  - Improved classifier head: LayerNorm → Dropout(0.3) → Linear → GELU → Dropout(0.2) → Linear
  - freeze_layers=8: first 8 transformer blocks kept frozen, only top 4 fine-tuned
    (reduces overfitting risk on a small dataset, speeds up training)
  - EMA class added
"""

from copy import deepcopy
import torch
import torch.nn as nn
from transformers import ASTForAudioClassification


class CustomAST(nn.Module):
    """
    AST with an improved classification head and partial layer freezing.

    Args:
        num_classes   : 4 for ICBHI (Normal / Crackle / Wheeze / Both)
        freeze_layers : number of transformer encoder blocks to freeze from the bottom
    """

    MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"

    def __init__(self, num_classes: int = 4, freeze_layers: int = 8):
        super().__init__()

        self.ast = ASTForAudioClassification.from_pretrained(
            self.MODEL_ID,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        # Freeze the first `freeze_layers` transformer encoder blocks.
        for i, layer in enumerate(self.ast.audio_spectrogram_transformer.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False


        # GELU + two Dropout layers prevent overconfident predictions and add
        # mild regularisation .
        hidden = self.ast.config.hidden_size

        self.ast.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(p=0.3),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.ast(input_values=input_values).logits


class EMA:
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy updated as:
        shadow = decay * shadow + (1 - decay) * current_weights

    At evaluation, shadow weights are used instead of the last checkpoint.
    They generalise better because they average over many training steps
    instead of relying on a single (potentially noisy) final state.

    Zero extra compute during training — just a running mean of tensors.
    
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay   = decay
        self.shadow  = deepcopy(model.state_dict())
        self._backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v
            else:
                self.shadow[k] = v.clone()

    def apply_shadow(self, model: nn.Module):
        self._backup = deepcopy(model.state_dict())
        model.load_state_dict(self.shadow)

    def restore(self, model: nn.Module):
        model.load_state_dict(self._backup)
        self._backup = {}

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, d: dict):
        self.shadow = d["shadow"]
        self.decay  = d["decay"]