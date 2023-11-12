import torch
from torch import nn



class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()

        # Create 2 nn.Linear layers
        self.layer_1 = nn.Linear(
            in_features=2,
            out_features=10
        )
        # We upscale to 10 because the more neurons there are,
        # the more opportunities the computer has to find a pattern

        self.layer_2 = nn.Linear(
            in_features=10,
            out_features=10
        )
        # We output 1 feature because we are looking for 1 result and our
        # y data has 1 feature
        self.layer_3 = nn.Linear(
            in_features=10,
            out_features=1
        )

        self.relu = nn.ReLU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
