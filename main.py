import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import numpy as np
import matplotlib.pyplot as plt

client_amount = 5
training_rounds = 10
clients = []
vectorizer = TfidfVectorizer()

# Load the dataset
df = pd.read_csv("data/spam.csv", header=0, names=["v1", "v2"])

# Preprocess the data
df["v1"] = df["v1"].astype(str).str.strip()
df["message"] = df["v2"].astype(str).str.strip()
df["label"] = df["v1"].map({"spam": 1, "ham": 0})

# Split the data into features and labels
X = df["message"]
y = df["label"]

# Split the data into training and separate test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training data into federated training and validation sets
X_train_fed, X_val, y_train_fed, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train_fed)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

X_train_split = np.array_split(X_train_tfidf.toarray(), client_amount)
y_train_split = np.array_split(y_train_fed, client_amount)

for i in range(client_amount):
    client = {
        "data": (X_train_split[i], y_train_split[i]),
        "weight": np.zeros((1, X_train_tfidf.shape[1])),  # Adjust the shape according to the feature size
        "intercept": np.zeros(1),
    }
    clients.append(client)

def train_local_model(X_local, y_local, weight=None, intercept=None):
    model = SGDClassifier(loss='log_loss', max_iter=1, tol=None, warm_start=True)
    if weight is not None and intercept is not None:
        model.coef_ = weight
        model.intercept_ = intercept

    model.fit(X_local, y_local)
    return model.coef_, model.intercept_

global_model = SGDClassifier(loss='log_loss', max_iter=1, tol=None, warm_start=True)
global_model.classes_ = np.unique(y_train_fed)

federated_accuracies = []

# Train local models and perform federated averaging

for i in range(training_rounds):
    # print(f"Epoch {i+1}")
    for client in clients:
        coef, intercept = train_local_model(
            client["data"][0],
            client["data"][1],
            getattr(global_model, "coef_", None),
            getattr(global_model, "intercept_", None),
        )
        client["weight"] = coef
        client["intercept"] = intercept

    # Federated Averaging: Combine model weights from all clients
    global_model.coef_ = np.mean([client["weight"] for client in clients], axis=0)
    global_model.intercept_ = np.mean([client["intercept"] for client in clients], axis=0)

    y_pred_val = global_model.predict(X_val_tfidf)
    accuracy = accuracy_score(y_val, y_pred_val)
    federated_accuracies.append(accuracy)
    # print(f"Accuracy after epoch {i+1}: {accuracy:.2f}")

    # Update every client's model with global weights
    for client in clients:
        client["weight"] = global_model.coef_
        client["intercept"] = global_model.intercept_

# Evaluate the federated model on the separate test set
y_pred_test = global_model.predict(X_test_tfidf)
federated_accuracy = accuracy_score(y_test, y_pred_test)
federated_precision = precision_score(y_test, y_pred_test)
federated_recall = recall_score(y_test, y_pred_test)

# Train a central model on the entire training set
central_model = SGDClassifier(loss='log_loss', max_iter=1, tol=None)
central_model.fit(vectorizer.transform(X_train), y_train)
central_y_pred_test = central_model.predict(X_test_tfidf)
central_accuracy = accuracy_score(y_test, central_y_pred_test)
central_precision = precision_score(y_test, central_y_pred_test)
central_recall = recall_score(y_test, central_y_pred_test)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, training_rounds + 1),
    federated_accuracies,
    label="Federated Learning Model (Validation)",
)
plt.axhline(y=central_accuracy, color='r', linestyle='--', label='Central Model (Test)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.legend()
plt.show()

# Print final results
print(f"Final Federated Learning Model Accuracy on Test Set: {federated_accuracy:.2f}")
print(f"Final Federated Learning Model Precision on Test Set: {federated_precision:.2f}")
print(f"Final Federated Learning Model Recall on Test Set: {federated_recall:.2f}")
print(f"Central Model Accuracy on Test Set: {central_accuracy:.2f}")
print(f"Central Model Precision on Test Set: {central_precision:.2f}")
print(f"Central Model Recall on Test Set: {central_recall:.2f}")
print("Classification Report for Federated Learning Model on Test Set:")
print(classification_report(y_test, y_pred_test))
print("Classification Report for Central Model on Test Set:")
print(classification_report(y_test, central_y_pred_test))