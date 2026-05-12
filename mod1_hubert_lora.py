from typing import Iterable

import torch
import torch.nn as nn

try:
    from transformers import HubertModel
except ImportError as e:
    raise ImportError(
        "Mod 1 requires `transformers`. Install with: pip install transformers"
    ) from e

try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    raise ImportError(
        "Mod 1 requires `peft`. Install with: pip install peft"
    ) from e


DEFAULT_HUBERT = "facebook/hubert-base-ls960"
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "intermediate_dense", "output_dense"]


class HubertLoRAEncoder(nn.Module):
    def __init__(
        self,
        method: str = "scl",
        num_classes: int = 2,
        model_name: str = DEFAULT_HUBERT,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.1,
        target_modules: Iterable[str] = tuple(DEFAULT_TARGET_MODULES),
        proj_dim: int = 512,
        freeze_feature_extractor: bool = True,
    ):
        super().__init__()
        self.method = method
        self.model_name = model_name
        self.proj_dim = proj_dim

        backbone = HubertModel.from_pretrained(model_name)

        if freeze_feature_extractor:
            try:
                backbone.feature_extractor._freeze_parameters()
            except AttributeError:
                pass

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=list(target_modules),
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.backbone = get_peft_model(backbone, lora_config)

        hidden_size = self._get_hidden_size()
        self.hidden_size = hidden_size

        if method in ("scl", "ssl"):
            self.lin = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, proj_dim),
            )
        else:
            self.lin = nn.Linear(hidden_size, num_classes)

    def _get_hidden_size(self) -> int:
        if hasattr(self.backbone, "config") and hasattr(self.backbone.config, "hidden_size"):
            return self.backbone.config.hidden_size
        if hasattr(self.backbone, "base_model"):
            base = self.backbone.base_model
            if hasattr(base, "config"):
                return base.config.hidden_size
        return 768

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.squeeze(1)
        outputs = self.backbone(x)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)
        out = self.lin(pooled)
        return pooled, out

    def replace_head_for_ce(self, num_classes: int = 2):
        self.lin = nn.Linear(self.hidden_size, num_classes)
        self.method = "ce"
        return self
