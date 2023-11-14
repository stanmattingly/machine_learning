import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split

from models.PressHookChurnModel import PressHookChurnModel
from utils import train_model
from torch.optim.lr_scheduler import StepLR

device = "cuda" if torch.cuda.is_available() else "cpu"


data = pd.read_csv('output.csv')

features = data.iloc[:, :-1]
labels = data.iloc[:, -1]

min_vals = features.min(axis=0)
max_vals = features.max(axis=0)
features_normalized = (features - min_vals) / (max_vals - min_vals)

X = torch.tensor(features_normalized.values, dtype=torch.float32).to(device)
y = torch.tensor(labels.values, dtype=torch.float32).to(device)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


churn_model = PressHookChurnModel().to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(
    params=churn_model.parameters(),
    lr=0.01,
)
scheduler = StepLR(optimizer, step_size=100, gamma=0.1)



train_model(churn_model, 150000, loss_fn, optimizer, X_train, y_train, X_test, y_test, torch.sigmoid, torch.round, True)

torch.save(churn_model, "churn_model.pt")