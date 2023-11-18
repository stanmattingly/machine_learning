import torch
from torch import nn 

from torchvision import transforms

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import requests
import zipfile
from pathlib import Path

from models.ComputerVisionModel import FashionModelConv

device = "cuda" if torch.cuda.is_available() else "cpu"
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"


## Below is for downloading the data and creating files
## 
## 
# if image_path.is_dir():
#     print("already exists")
# else:
#     image_path.mkdir(parents=True, exist_ok=True)

# with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
#     request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/data/pizza_steak_sushi.zip")
#     f.write(request.content)

# with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
#     print("unzipping")
#     zip_ref.extractall(image_path)

import os

train_dir = image_path / "train"
test_dir = image_path / "test"

train_image_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

test_image_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

train_dataset = ImageFolder(train_dir, transform=train_image_transform)
test_dataset = ImageFolder(test_dir, transform=test_image_transform)

train_dataloader = DataLoader(train_dataset,
                              batch_size=1,
                              num_workers=os.cpu_count(),
                              shuffle=True)

test_dataloader = DataLoader(test_dataset,
                              batch_size=1,
                              num_workers=os.cpu_count(),
                              shuffle=False)


image_model = FashionModelConv(
    input_shape=3,
    hidden_unit=20,
    output_shape=len(test_dataset.classes)
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=image_model.parameters(), lr=0.001)

epochs = 100

from helper_functions import test_step, train_step, accuracy_fn
from tqdm import tqdm

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}")
    train_step(
        image_model, 
        train_dataloader, 
        loss_fn, 
        optimizer, 
        accuracy_fn, 
        device
    )

    test_step(
        image_model,
        train_dataloader,
        loss_fn,
        accuracy_fn,
        device
    )
