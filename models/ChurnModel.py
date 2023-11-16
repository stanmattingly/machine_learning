import torch
from torch import nn



class ChurnModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Create 2 nn.Linear layers
        self.layer_1 = nn.Linear(
            in_features=11,
            out_features=64
        )
        # We upscale to 10 because the more neurons there are,
        # the more opportunities the computer has to find a pattern
        self.layer_2 = nn.Linear(
            in_features=64,
            out_features=16
        )

        self.layer_3 = nn.Linear(
            in_features=16,
            out_features=8
        )
        # We output 1 feature because we are looking for 1 result and our
        # y data has 1 feature
        self.layer_4 = nn.Linear(
            in_features=8,
            out_features=1
        )
        
        # Non-linear activation
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(8)

        self.dropout = nn.Dropout(0.2)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In-between each linear function, we add non-linear activation to get
        # closer to the circle trend we see when plotting the raw data
        x = self.layer_1(x)

        x = self.bn1(x)
        x = self.relu(self.layer_2(x))

        x = self.dropout(x)

        x = self.bn2(x)
        x = self.relu(self.layer_3(x))

        x = self.dropout(x)

        x = self.bn3(x)
        x = self.layer_4(x)


        return x