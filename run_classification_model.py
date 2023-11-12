import torch
from torch import nn
import pandas as pd
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

from models.ClassificationModel import CircleModelV0
from utils import train_model

device = "cuda" if torch.cuda.is_available() else "cpu"

n_samples = 100

# Create circles
X, y = make_circles(n_samples, noise=0.03, random_state=42)

X, y = torch.from_numpy(X).to(device).type(torch.float32), torch.from_numpy(y).to(device).type(torch.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model_0 = CircleModelV0().to(device)

# Loss function & optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(
    params=model_0.parameters(),
    lr=0.05,
)

# y_logits = model_0(X_test)[:5]
# y_pred_probs = torch.round(torch.sigmoid(y_logits))

train_model(model_0, 20000, loss_fn, optimizer, X_train, y_train, X_test, y_test, torch.sigmoid, torch.round, True)



















