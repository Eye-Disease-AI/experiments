import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .base import ModelBase


class ConvNextViT(ModelBase):
    """
    This uses a cnn as a tokenizer to a vit.
    Instead of tokenizing 16x16 pixel patches we use features extracted by cnn.
    """

    def __init__(self, n_classes: int, lr: float = 1e-3, weight_decay: float = 1e-4, dropout: float = 0.2):
        super().__init__(n_classes)
        self.save_hyperparameters()

        backbone = torchvision.models.convnext_base(weights='IMAGENET1K_V1')
        self.features = backbone.features
        convnext_channels = 1024
        convnext_patch_size = 7*7

        # Pretrained ViT encoder
        vit = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
        vit_channels = 768

        # convnext outputs 1024 
        self.proj = nn.Linear(convnext_channels, vit_channels)

        # CLS token from pretrained ViT
        self.cls_token = nn.Parameter(vit.class_token.clone())

        # Interpolate pos_embedding from 14x14 to 7x7 tokens
        # positional embeddings are passed along the tokens
        # vit was trained on 14*14 tokens (tokenizer split 224x224 images into 16x16 patches)
        # convnext outputs 7x7 features on each channel.
        # These need to be reshaped into 14x14 to use in vit

        pos         = vit.encoder.pos_embedding.data # shape: 1 x 197 x 768 (1 CLS + 14x14 patch tokens)
        pos_cls     = pos[:, :1] # first token is CLS, not spatial
        pos_patches = pos[:, 1:] # remaining 196 = 14x14 spatial grid
        pos_patches = pos_patches.reshape(1, 14, 14, vit_channels).permute(0, 3, 1, 2)
        pos_patches = F.interpolate(pos_patches, size=(7, 7), mode='bicubic', align_corners=False)
        pos_patches = pos_patches.permute(0, 2, 3, 1).reshape(1, convnext_patch_size, vit_channels)

        self.encoder = vit.encoder
        self.encoder.pos_embedding = nn.Parameter(torch.cat([pos_cls, pos_patches], dim=1))

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(vit_channels, n_classes),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def backbone_modules(self):
        return [self.features, self.encoder]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        patches = self.features(x).flatten(2).transpose(1, 2)      # 7x7 spatial grid flattened to 49 tokens
        tokens = self.proj(patches)                                # project to ViT dim
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)                  # prepend CLS: 50 tokens total
        tokens = self.encoder(tokens)                             # adds pos_embed, 12 layers, LN
        return self.head(tokens[:, 0])                             # CLS token carries global summary
