import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

import matplotlib.pyplot as plt

from models.ComputerVisionModel import FashionModel, FashionModelConv
from helper_functions import accuracy_fn

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# fashion_model = FashionModel(
#     784,
#     hidden_unit=10,
#     output_shape=len(test_data.classes)
# ).to(device)

fashion_model = FashionModelConv(
    input_shape=1,
    hidden_unit=10,
    output_shape=len(test_data.classes)
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=fashion_model.parameters(), lr=0.1)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 3

from helper_functions import test_step, train_step

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}")
    train_step(
        fashion_model, 
        train_dataloader, 
        loss_fn, 
        optimizer, 
        accuracy_fn, 
        device
    )

    test_step(
        fashion_model,
        train_dataloader,
        loss_fn,
        accuracy_fn,
        device
    )
