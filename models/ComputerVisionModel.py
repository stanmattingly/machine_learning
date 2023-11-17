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
    dataset=train_data,
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

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=fashion_model.parameters(), lr=0.1)

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 3

for epoch in range(epochs):
    print(f"Epoch: {epoch}")

    train_loss = 0
    for batch, (X, y) in enumerate(train_dataloader):
        fashion_model.train()

        y_pred = fashion_model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(batch, len(X))
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")
    
    train_loss /= len(train_dataloader)

    test_loss, test_acc = 0, 0
    fashion_model.eval()

    with torch.inference_mode():
        for X, y in test_dataloader:
            test_pred = fashion_model(X)

            test_loss += loss_fn(test_pred, y)

            test_acc += accuracy_fn(y, test_pred.argmax(dim=1))

        test_loss /= len(test_dataloader)

        test_acc /= len(test_dataloader)

    print(f"Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")














