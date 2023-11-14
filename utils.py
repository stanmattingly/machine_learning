import torch
import matplotlib.pyplot as plt
from datetime import datetime


def l1_regularization(model, lambda_):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return lambda_ * l1_norm

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

def train_model(model, epochs, loss_fn, optimizer, X_train, y_train, X_test, y_test, activation_fn=None, pred_format_fn=None, should_squeeze=False, scheduler=None):
    """
    Train a PyTorch model and evaluate it on test data.

    Args:
    model (torch.nn.Module): The neural network model to train.
    epochs (int): The number of epochs to train for.
    loss_fn (callable): The loss function.
    optimizer (torch.optim.Optimizer): The optimizer.
    X_train (torch.Tensor): Training data features.
    y_train (torch.Tensor): Training data labels.
    X_test (torch.Tensor): Test data features.
    y_test (torch.Tensor): Test data labels.
    """
    
    for epoch in range(epochs):
        # Training phase
        model.train()  # sets the model to training mode
        y_pred = model(X_train)

        if should_squeeze:
            y_pred = y_pred.squeeze()

        loss = loss_fn(y_pred, y_train)

        loss += l1_regularization(model, lambda_=1e-3)

        if activation_fn:
            y_pred = activation_fn(y_pred)
        if pred_format_fn:
            y_pred = pred_format_fn(y_pred)

        acc = accuracy_fn(y_train, y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Evaluation phase
        model.eval()  # sets the model to evaluation mode
        with torch.inference_mode():  # inference mode for evaluation
            test_pred = model(X_test)

            if should_squeeze:
                test_pred = test_pred.squeeze()

            test_loss = loss_fn(test_pred, y_test)
            test_loss += l1_regularization(model, lambda_=1e-3)

            if activation_fn:
                test_pred = activation_fn(test_pred)
            if pred_format_fn:
                test_pred = pred_format_fn(test_pred)

            test_acc = accuracy_fn(y_train, y_pred)

        # (Optional) Print epoch, loss, test loss, etc.
        if epoch % 100 == 0:  # print every 100 epochs, adjust as needed
            print(f"Epoch: {epoch}, Loss: {loss.item()}, Test Loss: {test_loss.item()}, Accuracy: {acc}, Test Accuracy: {test_acc}, Current LR: {scheduler.get_last_lr()}")


def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    """Function for plotting predictions against real data, handling PyTorch tensors."""

    plt.figure(figsize=(10, 7))

    # Function to convert tensor to numpy array if necessary
    def to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().detach().numpy()  # Move to CPU and convert to numpy
        return tensor

    # Convert data to numpy arrays if they are tensors
    train_data = to_numpy(train_data)
    train_labels = to_numpy(train_labels)
    test_data = to_numpy(test_data)
    test_labels = to_numpy(test_labels)

    # Plot training data
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    # Are there predictions?
    if predictions is not None:
        predictions = to_numpy(predictions)  # Convert predictions to numpy arrays if they are tensors
        # Plot the predictions
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.savefig(f"figures/{int(datetime.now().timestamp())}-plt.png")