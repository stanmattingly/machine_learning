import torch
import pandas as pd
# from models.PressHookChurnModel import PressHookChurnModel
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

X = torch.tensor(features_normalized.values, dtype=torch.float32).to(device)
y = torch.tensor(labels.values, dtype=torch.float32).to(device)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = model.to(device)
model.eval()  # Make sure the model is in evaluation mode

test_predictions = []
test_labels = []

with torch.inference_mode():
    for i in range(len(X_test)):
        inputs = X_test[i].unsqueeze(0)  # Add batch dimension
        label = y_test[i]
        outputs = model(inputs)
        prediction = torch.round(torch.sigmoid(outputs))  # Assuming binary classification
        test_predictions.append(prediction.cpu().numpy())
        test_labels.append(label.cpu().numpy())

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# Convert predictions and labels to binary format if necessary
binary_predictions = [int(p[0]) for p in test_predictions]
binary_labels = [int(l) for l in test_labels]

# Calculate metrics
precision = precision_score(binary_labels, binary_predictions)
recall = recall_score(binary_labels, binary_predictions)
f1 = f1_score(binary_labels, binary_predictions)
conf_matrix = confusion_matrix(binary_labels, binary_predictions)
roc_auc = roc_auc_score(binary_labels, binary_predictions)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix:\n {conf_matrix}")
print(f"ROC-AUC Score: {roc_auc}")