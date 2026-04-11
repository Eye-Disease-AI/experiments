import torch
import torch.nn as nn
import torchvision


class Model(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = torchvision.models.convnext.convnext_base(weights='IMAGENET1K_V1')
        new_clf = list(self.model.classifier.children())[:-1]
        new_clf.append(nn.LazyLinear(n_classes))
        new_clf.append(nn.Dropout(0.2))
        new_clf = nn.Sequential(*new_clf)
        self.model.classifier = new_clf


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
