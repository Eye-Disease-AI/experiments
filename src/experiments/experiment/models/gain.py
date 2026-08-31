from collections.abc import Callable
from dataclasses import dataclass

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.detection import MeanAveragePrecision

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
    features: list[torch.Tensor],  # L x [B, K, H, W]
    logits: torch.Tensor,  # [B, n_classes]
    target_classes: torch.Tensor,  # [B]
) -> list[torch.Tensor]:  # L x [B, H, W]
    # per-class logits: s^c for every sample
    # for every image get just the target class logit
    s_c = logits.gather(1, target_classes.unsqueeze(1)).squeeze(1)  # [B]
    score = s_c.sum()  # sum over batch

    # neuron importance weights w^c_{l,k}
    # class score gradient over activation
    grads = torch.autograd.grad(
        outputs=score, inputs=features, create_graph=True, retain_graph=True
    )  # L x [B, K, H, W]
    maps = []
    for feats, grad in zip(features, grads):
        # channel wise weight obtained by averaging the gradients in A
        weights = grad.mean(dim=(2, 3))
        # Attention map A^c
        A = F.relu((weights[:, :, None, None] * feats).sum(dim=1))
        maps.append(A)
    return maps


def compute_gradcam_pp(
    features: list[torch.Tensor],  # L x [B, K, H, W]
    logits: torch.Tensor,  # [B, n_classes]
    target_classes: torch.Tensor,  # [B]
) -> list[torch.Tensor]:  # L x [B, H, W]
    s_c = logits.gather(1, target_classes.unsqueeze(1)).squeeze(1)  # [B]
    score = s_c.sum()

    grads = torch.autograd.grad(
        outputs=score, inputs=features, create_graph=True, retain_graph=True
    )  # L x [B, K, H, W]
    maps = []
    # gradcam ++:
    # Instead of channel wise average, it uses both feature maps and gradients
    # for channel wise weights
    for feats, grad in zip(features, grads):
        g2 = grad.pow(2)
        sum_a = feats.sum(dim=(2, 3))[:, :, None, None]
        alpha = torch.where(grad != 0, g2 / (2 * g2 + sum_a * grad.pow(3)), 0)
        weights = (alpha * F.relu(grad)).sum(dim=(2, 3))  # [B,K]
        A = F.relu((weights[:, :, None, None] * feats).sum(dim=1))
        maps.append(A)
    return maps


def compute_layercam(
    features: list[torch.Tensor],  # L x [B, K, H, W]
    logits: torch.Tensor,  # [B, n_classes]
    target_classes: torch.Tensor,  # [B]
) -> list[torch.Tensor]:  # L x [B, H, W]
    # per-class logits: s^c for every sample
    # for every image get just the target class logit
    s_c = logits.gather(1, target_classes.unsqueeze(1)).squeeze(1)  # [B]
    score = s_c.sum()

    # neuron importance weights w^c_{l,k}
    # class score gradient over activation
    grads = torch.autograd.grad(
        outputs=score, inputs=features, create_graph=True, retain_graph=True
    )  # L x [B, K, H, W]
    maps = []
    # Layercam:
    # 1. crops grads to > 0 before multiplying by feats
    # 2. does not average pool, uses raw grads
    for feats, grad in zip(features, grads):
        A = F.relu((F.relu(grad) * feats).sum(dim=1))
        maps.append(A)
    return maps


# Receives: list of batched layer activations, batched output logits (1-hot), batched true classes
# returns list of batched heatmaps
HeatmapFn = Callable[
    [list[torch.Tensor], torch.Tensor, torch.Tensor], list[torch.Tensor]
]

HEATMAP_METHODS: dict[str, HeatmapFn] = {
    "gradcam": compute_gradcam,
    "layercam": compute_layercam,
    "gradcam_pp": compute_gradcam_pp,
}


def attention_map(
    method: str,
    image: torch.Tensor,
    features: torch.Tensor,
    logits: torch.Tensor,
    target_classes: torch.Tensor,
) -> torch.Tensor:  # [B, H, W], image-sized, normalized
    maps = HEATMAP_METHODS[method](features, logits, target_classes)
    s = sum([normalize_attention(interpolate_A(image, A)) for A in maps])
    return normalize_attention(s)


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


def gradcam_visualization(image: torch.Tensor, A_upsampled: torch.Tensor) -> np.ndarray:
    a = A_upsampled.detach().cpu().numpy()
    heatmap = (matplotlib.colormaps["turbo"](a)[..., :3] * 255).astype(np.uint8)
    mean = torch.tensor(
        NuclearCataractDataModule.DATASET_MEAN, device=image.device, dtype=image.dtype
    ).view(-1, 1, 1)
    std = torch.tensor(
        NuclearCataractDataModule.DATASET_STD, device=image.device, dtype=image.dtype
    ).view(-1, 1, 1)
    denorm = image.detach() * std + mean
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


def _resolve_layer(root: nn.Module, keys) -> nn.Module:
    """Resolve a target layer by a sequence of keys (str -> getattr, int -> index)."""
    obj = root
    for k in keys:
        obj = getattr(obj, k) if isinstance(k, str) else obj[k]
    return obj


def calculate_heatmaps(
    methods: list[str], amount, x, features: list[torch.Tensor], logits, y
) -> dict[str, list[np.ndarray]]:
    heatmaps = {}
    for name in methods:
        A = attention_map(name, x, features, logits, y).detach()
        heatmaps[name] = [
            gradcam_visualization(x[i % len(x)], A[i % len(x)]) for i in range(amount)
        ]
    return heatmaps


@dataclass(frozen=True, kw_only=True)
class GAINWrapperConfig(BaseClassifierModelConfig):
    target_layers: list[tuple[str | int, ...]]
    am_loss_weight: float | OptunaOptimised = 1.0  # alpha
    es_loss_weight: float | OptunaOptimised = 1.0  # omega_e
    sigma_mask: float | OptunaOptimised = 0.5  # soft-threshold sigmoid center
    omega_mask: float | OptunaOptimised = 100.0  # soft-threshold sigmoid steepness
    warmup_epochs: int | OptunaOptimised = 0  # before AM/ES are added to loss
    calculate_heatmaps: bool = True  # visualizations
    use_attention_mining: bool = False
    use_external_supervision: bool = False
    gradcam_visualization_amount: int = 3  # images per epoch
    threshold_ratio: float = 0.2  # for giving class labels to pixels for IoU/mAP
    loss_heatmap_method: str = "gradcam"  # for AM/ES loss + val metrics
    visualization_heatmap_methods: tuple[str, ...] = (
        "gradcam",
    )  # for visualization only

    def validate_config(self):
        super().validate_config()
        unknown = {
            self.loss_heatmap_method,
            *self.visualization_heatmap_methods,
        } - HEATMAP_METHODS.keys()
        if unknown:
            raise ValueError(
                f"Unknown heatmap methods {sorted(unknown)}; "
                f"available: {sorted(HEATMAP_METHODS)}"
            )


class GAINWrapper(BaseClassifierModel):
    def __init__(self, config: GAINWrapperConfig):
        super().__init__(config)
        self.config: GAINWrapperConfig = config
        for path in self.config.target_layers:
            target = _resolve_layer(self, path)
            _install_proxy(self, target, CapturingProxy(target))

        self.val_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        self.val_iou_sum = torch.zeros(())
        self.val_iou_n = 0

    @property
    def _proxies(self) -> list[CapturingProxy]:
        return [_resolve_layer(self, path) for path in self.config.target_layers]

    def training_step(self, batch, batch_idx):
        """Integrate all losses during training"""
        x, y, *other = batch
        boxes = other[0] if other else None
        # GradCAM needs grad on features and backbone freezing prevents that.
        # setting input to require grad makes intermediate gradients to be
        # calcualted without unfreezig the intermediate layers
        x = x.requires_grad_(True)
        loss_cl, features, logits = self._classification_step(x, y)
        losses = {"L_cl": loss_cl}
        total = loss_cl

        A = attention_map(self.config.loss_heatmap_method, x, features, logits, y)

        if (
            self.config.use_attention_mining
            and self.current_epoch >= self.config.warmup_epochs
        ):
            L_am = self._attention_mining_loss(x, A, y)
            L_am = L_am * self.config.am_loss_weight
            losses["L_am"] = L_am
            total = total + L_am

        if self.config.calculate_heatmaps and self.trainer.is_last_batch:
            heatmaps = calculate_heatmaps(
                self.config.visualization_heatmap_methods,
                self.config.gradcam_visualization_amount,
                x,
                features,
                logits,
                y,
            )
            for name, heatmaps_of_type in heatmaps.items():
                for i, heatmap in enumerate(heatmaps_of_type):
                    self.logger.experiment.log_image(
                        run_id=self.logger.run_id,
                        image=heatmap,
                        artifact_file=f"{name}/step_{self.global_step}_img{i}.png",
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
        proxies = self._proxies
        for p in proxies:
            # clearing proxies
            p.captured = torch.empty(0)
        logits = self(x)  # proxies capture the activations after forward pass
        loss_cl = self.loss_fn(logits, y)
        return loss_cl, [p.captured for p in proxies], logits

    def _attention_mining_loss(self, x, A, y):
        """Second forward on masked image."""
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

    def _attention_to_segmentation_mask(self, A: torch.Tensor) -> torch.Tensor:
        thresh = self.config.threshold_ratio * A.amax(dim=(1, 2), keepdim=True)
        return (A >= thresh).float()

    def mask_iou(
        self, prediction_mask: torch.Tensor, true_mask: torch.Tensor
    ) -> torch.Tensor:
        p = prediction_mask.flatten(1).bool()
        g = true_mask.flatten(1).bool()
        inter = (p & g).sum(1).float()
        union = (p | g).sum(1).float()
        # Edge case for no labels
        # if pred has a box but gt had none, completely invalid
        # if gt has no box, and pred has no box, perfect
        return torch.where(union == 0, torch.ones_like(union), inter / union)

    def mask_to_box(self, m: torch.Tensor) -> torch.Tensor | None:
        """Minimal box that covers the whole mask"""
        ys, xs = torch.nonzero(m, as_tuple=True)
        if xs.numel() == 0:
            # no box if theres no non-zero mask pixels
            return None
        return torch.stack([xs.min(), ys.min(), xs.max(), ys.max()]).float()

    def mean_average_precision(
        self,
        pred_mask: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        boxes: list[torch.Tensor],
        logits: torch.Tensor,
    ):
        dev = pred_mask.device
        probs = torch.softmax(logits, dim=1).detach()
        H, W = x.shape[-2:]
        scale = torch.tensor([W, H, W, H], device=dev)
        preds, targets = [], []
        for i in range(x.shape[0]):
            # Converting to COCO format to use in torchvision.detection.MeanAveragePrecision
            box = self.mask_to_box(pred_mask[i])
            if box is None:
                preds.append(
                    {
                        "boxes": torch.zeros(0, 4, device=dev),
                        "scores": torch.zeros(0, device=dev),
                        "labels": torch.zeros(0, dtype=torch.long, device=dev),
                    }
                )
            else:
                preds.append(
                    {
                        "boxes": box.unsqueeze(0),
                        "scores": probs[i, y[i]].unsqueeze(0),
                        "labels": y[i].unsqueeze(0),
                    }
                )
            true_boxes = boxes[i].to(dev) * scale
            targets.append(
                {"boxes": true_boxes, "labels": y[i].repeat(len(true_boxes))}
            )
        # print("=== MAP debug ===")
        # for i in range(min(3, len(preds))):
        #     print(f"[{i}] scores {preds[i]['scores']} labels {preds[i]['labels']}"
        #     print(f"    pred boxes {preds[i]['boxes'].shape}")
        #     print(f"    pred box vals {preds[i]['boxes']}")
        #     print(f"    true boxes {targets[i]['boxes'].shape} labels {targets[i]['labels']}")
        #     print(f"    true box vals {targets[i]['boxes'][:2]}")
        self.val_map.update(preds, targets)

    def validation_step(self, batch, batch_idx):
        """Adding new metrics to the base classifier metrics"""
        # Calculates original metrics
        super().validation_step(batch, batch_idx)
        x, y, *other = batch
        boxes = other[0] if other else None
        assert boxes
        with (
            torch.inference_mode(False),
            torch.enable_grad(),
        ):  # val has no grad, GradCAM needs it
            # Need to clone the tensor in order to free it up from inference mode context
            x: torch.Tensor = x.clone().requires_grad_(True)
            y: torch.Tensor = y.clone()
            _, features, logits = self._classification_step(x, y)
            A = attention_map(
                self.config.loss_heatmap_method, x, features, logits, y
            ).detach()
        mask = self._attention_to_segmentation_mask(A)
        gt_mask = boxes_batch_to_masks(boxes, x.shape[-2:], A.device)

        self.val_iou_sum += self.mask_iou(mask, gt_mask).sum().cpu()  # sum over batch
        self.val_iou_n += x.shape[0]

        self.mean_average_precision(mask, x, y, boxes, logits)

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()

        self.log("val_miou", self.val_iou_sum / max(self.val_iou_n, 1), prog_bar=True)
        self.val_iou_sum = torch.zeros(())
        self.val_iou_n = 0

        m = self.val_map.compute()
        self.log_dict({"val_map": m["map"], "val_map_50": m["map_50"]}, prog_bar=True)
        self.val_map.reset()
