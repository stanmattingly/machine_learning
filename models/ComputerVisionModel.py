import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn

import matplotlib.pyplot as plt


train_data = datasets.FashionMNIST(
    'data',
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

test_data = datasets.FashionMNIST(
    'data',
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)


BATCH_SIZE = 32

train_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


class FashionModel(nn.Module):
    def __init__(self, 
                 input_shape,
                 hidden_unit,
                 output_shape):
        
        super().__init__()

        self.layer_stack = nn.Sequential(
            # Flattens the image tensor to a vector for linear models below
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_unit),
            nn.Linear(in_features=hidden_unit, out_features=output_shape)
        )

    
    def forward(self, x):
        return self.layer_stack(x)
    
fashion_model = FashionModel(
    784,
    hidden_unit=10,
    output_shape=len(test_data.classes)
)

optimizer = torch.optim.SGD(params=FashionModel.parameters(), lr=0.01)











