from dataclasses import dataclass
from typing import Literal, override

import lightning as L
import torch
from torchmetrics import MeanMetric
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from experiments.lib.config_serializing import ClassConfig


@dataclass(frozen=True, kw_only=True)
class LPIPSModuleConfig(ClassConfig):
    n_classes: int = 2
    net_type: Literal["alex", "squeeze", "vgg"] = "alex"

    @override
    @staticmethod
    def get_configured_class():
        return LPIPSModule


class LPIPSModule(L.LightningModule):
    def __init__(self, config: LPIPSModuleConfig):
        super().__init__()
        self.config = config
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type=config.net_type, reduction="none", normalize=True
        )
        self.class_means = torch.nn.ModuleList(
            [MeanMetric() for _ in range(config.n_classes)]
        )

    def validation_step(self, batch, batch_idx):
        first, second, labels = batch
        distances = self.lpips(first, second).reshape(-1)
        for class_index, metric in enumerate(self.class_means):
            assert isinstance(metric, MeanMetric)
            mask = labels == class_index
            if mask.any():
                metric.update(distances[mask])

    def on_validation_epoch_end(self):
        metrics = {"lpips": self.lpips.compute().mean()}
        for class_index, metric in enumerate(self.class_means):
            assert isinstance(metric, MeanMetric)
            metrics[f"lpips_class_{class_index}"] = metric.compute()
        self.log_dict(metrics, prog_bar=True)
        self.lpips.reset()
        for metric in self.class_means:
            assert isinstance(metric, MeanMetric)
            metric.reset()
