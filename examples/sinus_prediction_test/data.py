import torch

# create a "datamodule" or something

# Data
# fit sin(x) over [0, 2pi]
X = torch.linspace(0, 2 * torch.pi, 100).unsqueeze(1)
y = torch.sin(X)

