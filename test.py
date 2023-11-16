import torch
import pandas as pd
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the model structure
model = torch.load('churn_model.pt')

# Set the model to evaluation mode
model.eval()

data = pd.read_csv('output.csv')

features = data.iloc[:, :-1]
labels = data.iloc[:, -1]

min_vals = features.min(axis=0)
max_vals = features.max(axis=0)
features_normalized = (features - min_vals) / (max_vals - min_vals)

def predict_new_data(new_data, model, min_vals, max_vals, device):
    """
    Predict churn for new data using the trained model.

    Args:
    - new_data (list of lists): New data for which predictions are needed. Each sublist represents a single data point.
    - model (torch.nn.Module): Trained PyTorch model for predictions.
    - min_vals (pd.Series): Minimum values of the training data features for normalization.
    - max_vals (pd.Series): Maximum values of the training data features for normalization.
    - device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
    - List of predictions.
    """
    # Convert new data to DataFrame for easy normalization
    new_data_df = pd.DataFrame(new_data, columns=min_vals.index)
    
    # Normalize the new data
    new_data_normalized = (new_data_df - min_vals) / (max_vals - min_vals)
    
    # Convert normalized data to PyTorch tensor
    new_data_tensor = torch.tensor(new_data_normalized.values, dtype=torch.float32).to(device)

    # Prepare the model for evaluation
    model = model.to(device)
    model.eval()

    predictions = []

    # Make predictions
    with torch.inference_mode():
        for i in range(len(new_data_tensor)):
            inputs = new_data_tensor[i].unsqueeze(0)  # Add batch dimension
            outputs = model(inputs)
            prediction = torch.sigmoid(outputs) # Assuming binary classification
            predictions.append(prediction.cpu().numpy()[0])

    return predictions


new_inputs = [[32, 5, 3, 194, 49, 6, 245, 32, 21, 50560, 1], [8, 5, 4, 39, 4, 9, 35, 21, 149, 10695, 0], [3, 1, 0, 25, 0, 0, 3, 10, 0, 2699, 1], [3, 2, 1, 13, 2, 1, 7, 8, 3, 2100, 1], [8, 1, 1, 79, 6, 2, 9, 9, 149, 10401, 2], [3, 0, 0, 13, 0, 0, 0, 6, 0, 3162, 2], [8, 1, 1, 33, 7, 1, 16, 12, 149, 5582, 1], [34, 7, 2, 162, 18, 6, 76, 36, 17, 18227, 1], [8, 1, 0, 32, 2, 6, 26, 17, 149, 6645, 1], [8, 1, 0, 46, 2, 0, 11, 9, 149, 6781, 1], [27, 0, 0, 3, 1, 2, 11, 1, 1, 2268, 0], [17, 2, 1, 54, 8, 10, 30, 9, 18, 11675, 2], [27, 2, 1, 59, 9, 3, 17, 15, 149, 6085, 2], [23, 7, 0, 265, 21, 17, 194, 47, 41, 47323, 2], [14, 7, 5, 38, 10, 3, 23, 18, 149, 6785, 2], [13, 1, 1, 23, 6, 2, 31, 16, 149, 6127, 0], [9, 1, 0, 23, 0, 7, 42, 10, 45, 2602, 0], [15, 0, 0, 5, 0, 2, 27, 0, 45, 8369, 0], [12, 6, 2, 11, 2, 6, 6, 6, 193, 3195, 1], [12, 5, 5, 25, 0, 6, 97, 14, 52, 17219, 3], [12, 1, 0, 11, 1, 1, 1, 1, 52, 2912, 0], [13, 5, 0, 25, 18, 4, 18, 15, 52, 6342, 0], [10, 4, 0, 38, 14, 14, 14, 13, 63, 11367, 1], [0, 0, 0, 2, 0, 0, 1, 0, 215, 1866, 2], [12, 2, 0, 24, 2, 14, 6, 9, 81, 3456, 0], [12, 5, 5, 6, 0, 1, 7, 5, 11, 1997, 0], [12, 1, 0, 7, 0, 1, 0, 5, 0, 1118, 0], [3, 2, 2, 56, 4, 25, 4, 25, 193, 4694, 0], [3, 3, 3, 125, 14, 22, 8, 23, 77, 10427, 1], [2, 1, 1, 24, 4, 0, 0, 4, 3, 553, 0], [2, 1, 1, 0, 0, 0, 0, 2, 3, 556, 0], [1, 2, 2, 5, 2, 2, 0, 0, 10, 296, 0], [2, 4, 4, 4, 0, 1, 0, 16, 25, 2215, 1], [0, 0, 0, 0, 0, 0, 3, 5, 50, 984, 1]]
brand_input = [[1, 2, 2, 5, 2, 2, 0, 0, 10, 296, 0]]
predictions = predict_new_data(brand_input, model, min_vals, max_vals, device)

for idx, item in enumerate(predictions):
    print(item.item())

# X = torch.tensor(features_normalized.values, dtype=torch.float32).to(device)
# y = torch.tensor(labels.values, dtype=torch.float32).to(device)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = model.to(device)
# model.eval()  # Make sure the model is in evaluation mode

# test_predictions = []
# test_labels = []

# with torch.inference_mode():
#     for i in range(len(X_test)):
#         inputs = X_test[i].unsqueeze(0)  # Add batch dimension
#         label = y_test[i]
#         outputs = model(inputs)
#         prediction = torch.round(torch.sigmoid(outputs))  # Assuming binary classification
#         test_predictions.append(prediction.cpu().numpy())
#         test_labels.append(label.cpu().numpy())


# from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# # Convert predictions and labels to binary format if necessary
# binary_predictions = [int(p[0]) for p in test_predictions]
# binary_labels = [int(l) for l in test_labels]

# # Calculate metrics
# precision = precision_score(binary_labels, binary_predictions)
# recall = recall_score(binary_labels, binary_predictions)
# f1 = f1_score(binary_labels, binary_predictions)
# conf_matrix = confusion_matrix(binary_labels, binary_predictions)
# roc_auc = roc_auc_score(binary_labels, binary_predictions)

# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1 Score: {f1}")
# print(f"Confusion Matrix:\n {conf_matrix}")
# print(f"ROC-AUC Score: {roc_auc}")