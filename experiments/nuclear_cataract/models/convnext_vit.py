import torch
import torch.nn as nn
import torchvision
from .base import ModelBase


class ConvNextViT(ModelBase):
    _VIT_DIM = 768
    _N_HEADS = 8
    _N_LAYERS = 4

    def __init__(self, n_classes: int, lr: float = 1e-3, weight_decay: float = 1e-4, dropout: float = 0.2):
        super().__init__(n_classes)
        self.save_hyperparameters()

        backbone = torchvision.models.convnext_base(weights='IMAGENET1K_V1')
        self.features = backbone.features  # output: [B, 1024, 7, 7]
        for p in self.features.parameters():
            p.requires_grad = False

        n_patches = 7 * 7  # 49 for 224×224 input
        self.proj = nn.Linear(1024, self._VIT_DIM)
        self.cls_token = nn.Parameter(torch.empty(1, 1, self._VIT_DIM))
        self.pos_embed = nn.Parameter(torch.empty(1, n_patches + 1, self._VIT_DIM))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._VIT_DIM, nhead=self._N_HEADS,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self._N_LAYERS)

        self.head = nn.Sequential(
            nn.LayerNorm(self._VIT_DIM),
            nn.Dropout(dropout),
            nn.Linear(self._VIT_DIM, n_classes),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        patches = self.features(x).flatten(2).transpose(1, 2)  # [B, 49, 1024]
        tokens = self.proj(patches)                             # [B, 49, 768]
        cls = self.cls_token.expand(B, -1, -1)                 # [B, 1, 768]
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed  # [B, 50, 768]
        tokens = self.transformer(tokens)
        return self.head(tokens[:, 0])                          # CLS token