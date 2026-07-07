from dataclasses import dataclass

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModule,
)
from experiments.experiment.models.base_classifier import (
    BaseClassifierModel,
    BaseClassifierModelConfig,
)
from experiments.lib.config_serializing import OptunaOptimised


class CapturingProxy(nn.Module):
    """Wraps target_layer, runs it normally, stores its output."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.captured = torch.empty(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x)
        self.captured = out
        return out


def _install_proxy(root: nn.Module, target: nn.Module, proxy: nn.Module) -> None:
    """Find target in root's module tree and swap it for the proxy."""
    for parent in root.modules():
        for name, child in list(parent.named_children()):
            if child is target:
                setattr(parent, name, proxy)
                return
    raise ValueError("target_layer not found in model")


def compute_gradcam(
    features: torch.Tensor,  # [B, K, H, W]
    logits: torch.Tensor,  # [B, n_classes]
    target_classes: torch.Tensor,  # [B]
) -> torch.Tensor:  # [B, H, W]
    # per-class logits: s^c for every sample
    # for every image get just the target class logit
    s_c = logits.gather(1, target_classes.unsqueeze(1)).squeeze(1)  # [B]
    score = s_c.sum()  # sum over batch

    # neuron importance weights w^c_{l,k}
    # class score gradient over activation
    grads = torch.autograd.grad(
        outputs=score, inputs=features, create_graph=True, retain_graph=True
    )[0]  # [B, K, H, W]

    weights = grads.mean(dim=(2, 3))

    # Attention map A^c
    A = F.relu((weights[:, :, None, None] * features).sum(dim=1))
    return A


def soft_threshold(A: torch.Tensor, sigma: float, omega: float) -> torch.Tensor:
    return torch.sigmoid(omega * (A - sigma))


def normalize_attention(A: torch.Tensor) -> torch.Tensor:
    """Per sample min-max normalization."""
    A_flat = A.view(A.shape[0], -1)
    A_min = A_flat.min(dim=1, keepdim=True)[0]
    A_max = A_flat.max(dim=1, keepdim=True)[0]
    return ((A_flat - A_min) / (A_max - A_min).clamp(min=1e-8)).view_as(A)


def interpolate_A(image: torch.Tensor, A: torch.Tensor):
    A_upsampled = F.interpolate(
        A.unsqueeze(1), size=image.shape[-2:], mode="bilinear", align_corners=False
    ).squeeze(1)
    return A_upsampled


def gradcam_visualization(image: torch.Tensor, A_upsampled: torch.Tensor):
    a = A_upsampled.detach().cpu().numpy()
    heatmap = (matplotlib.colormaps["turbo"](a)[..., :3] * 255).astype(np.uint8)
    mean = torch.tensor(
        NuclearCataractDataModule.DATASET_MEAN, device=image.device, dtype=image.dtype
    ).view(-1, 1, 1)
    std = torch.tensor(
        NuclearCataractDataModule.DATASET_STD, device=image.device, dtype=image.dtype
    ).view(-1, 1, 1)
    denorm = image * std + mean
    img_np = (denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    alpha = (a * 0.5)[..., None]
    overlay = img_np * (1 - alpha) + heatmap * alpha
    return overlay.astype(np.uint8)


def mask_image(
    image: torch.Tensor, A_upsampled: torch.Tensor, sigma: float, omega: float
) -> torch.Tensor:
    T_A = soft_threshold(A_upsampled, sigma, omega)
    return image * (1 - T_A.unsqueeze(1))


def boxes_to_mask(
    boxes: torch.Tensor,  # [N, 4] format (x1, y1, x2, y2)
    image_size: tuple[int, int],  # (H, W)
    device: torch.device,
) -> torch.Tensor:  # [H, W] values 0-1
    """Makes an image size bit mask using the bounding box. Zeros inside box."""
    H, W = image_size
    mask = torch.zeros(H, W, device=device)
    for x1, y1, x2, y2 in boxes:
        x1, y1 = int(x1.item() * image_size[1]), int(y1.item() * image_size[0])
        x2, y2 = int(x2.item() * image_size[1]), int(y2.item() * image_size[0])
        mask[y1:y2, x1:x2] = 1.0
    return mask


def boxes_batch_to_masks(
    boxes_batch: list[torch.Tensor],  # list of B elements [H, W]
    image_size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:  # [B, H, W]
    return torch.stack([boxes_to_mask(b, image_size, device) for b in boxes_batch])


def _resolve_layer(root: nn.Module, keys):
    """Resolve a target layer by a sequence of keys (str -> getattr, int -> index)."""
    obj = root
    for k in keys:
        obj = getattr(obj, k) if isinstance(k, str) else obj[k]
    return obj


@dataclass(frozen=True, kw_only=True)
class GAINWrapperConfig(BaseClassifierModelConfig):
    target_layer: tuple[str | int, ...] = ("model", "features", -2, -1)
    am_loss_weight: float | OptunaOptimised = 1.0  # alpha
    es_loss_weight: float | OptunaOptimised = 1.0  # omega_e
    sigma_mask: float | OptunaOptimised = 0.5  # soft-threshold sigmoid center
    omega_mask: float | OptunaOptimised = 100.0  # soft-threshold sigmoid steepness
    warmup_epochs: int | OptunaOptimised = 0
    calculate_attention_mining: bool = True
    use_attention_mining: bool = False
    use_external_supervision: bool = False
    gradcam_visualization_amount: int = 3


class GAINWrapper(BaseClassifierModel):
    def __init__(self, config: GAINWrapperConfig):
        super().__init__(config)
        self.config: GAINWrapperConfig = config
        target = _resolve_layer(self, self.config.target_layer)
        _install_proxy(self, target, CapturingProxy(target))

    @property
    def _proxy(self) -> CapturingProxy:
        # Re-resolved instead of stored: registering the proxy as an attribute
        # would duplicate its parameters in state_dict.
        return _resolve_layer(self, self.config.target_layer)

    def training_step(self, batch, batch_idx):
        """Integrate all losses during training"""
        x, y, *other = batch
        boxes = other[0] if other else None
        loss_cl, features, logits = self._classification_step(x, y)
        losses = {"L_cl": loss_cl}
        total = loss_cl

        A = compute_gradcam(features, logits, y)  # [B, h, w]
        A = interpolate_A(x, A)
        A = normalize_attention(A)

        if (
            self.config.use_attention_mining
            and self.current_epoch >= self.config.warmup_epochs
        ):
            L_am = self._attention_mining_loss(
                x, A, y, log_gradcam=self.trainer.is_last_batch
            )
            L_am = L_am * self.config.am_loss_weight
            losses["L_am"] = L_am
            total = total + L_am
        elif self.config.calculate_attention_mining:
            _ = self._attention_mining_loss(
                x, A, y, log_gradcam=self.trainer.is_last_batch
            )

        if self.config.use_external_supervision:
            L_e = self._external_supervision_loss(A, boxes, x.shape[-2:])
            L_e = L_e * self.config.es_loss_weight
            losses["L_e"] = L_e
            total = total + L_e

        self.log_dict({f"train_{k}": v for k, v in losses.items()}, prog_bar=True)
        self.log("train_loss", total, prog_bar=True)
        return total

    def _classification_step(self, x, y):
        """Classification loss and extracting the features along the way"""
        proxy = self._proxy
        proxy.captured = torch.empty(0)
        logits = self(x)
        loss_cl = self.loss_fn(logits, y)
        return loss_cl, proxy.captured, logits

    def _attention_mining_loss(self, x, A, y, log_gradcam=False):
        """Second forward on masked image."""
        if log_gradcam:
            for i in range(self.config.gradcam_visualization_amount):
                idx = i % len(x)
                heatmap = gradcam_visualization(x[idx], A[idx])
                self.logger.experiment.log_image(
                    run_id=self.logger.run_id,
                    image=heatmap,
                    artifact_file=f"gradcam/step_{self.global_step}_img{i}.png",
                )

        x_star = mask_image(x, A, self.config.sigma_mask, self.config.omega_mask)
        logits_star = self(x_star)
        probs_star = F.softmax(logits_star, dim=1)
        # s^c(I*): ground-truth-class score of images with the grad-cam mask applied.
        # Higher score after masking => worse, so it acts directly as a loss.
        s_c_star = probs_star.gather(1, y.unsqueeze(1)).squeeze(1)
        return s_c_star.mean()

    def _external_supervision_loss(self, A, boxes_batch, image_size):
        """External mask and gradcam difference"""
        if boxes_batch is None or not boxes_batch:
            return 0
        H = boxes_batch_to_masks(boxes_batch, image_size, A.device)
        return F.mse_loss(A, H)
