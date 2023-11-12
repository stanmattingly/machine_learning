import torch
from torch import nn
from models.LinearModel import LinearRegressionModel
from utils import train_model, plot_predictions

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create known parameters with linear regression
weight = 0.7
bias = 0.3
# y = weight*x + bias (linear formula)

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1) # Need to unsqueeze for models
y = weight * X + bias

X, y = X.to(device), y.to(device)

train_split = int(0.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

RANDOM_SEED = 42
torch.manual_seed(42)

model_0 = LinearRegressionModel().to(device)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.001)


train_model(model_0, 10000, loss_fn, optimizer, X_train, y_train, X_test, y_test)

with torch.inference_mode():
    y_preds = model_0(X_test)

plot_predictions(X_train, y_train, X_test, y_test, y_preds)