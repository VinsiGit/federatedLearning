import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

amount = 5
clients = {
    "clients_data": np.empty(amount, dtype=object),
    "client_weights": np.empty(amount, dtype=object),
    "client_intercepts": np.empty(amount, dtype=object),
}
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

skf = StratifiedKFold(n_splits=amount, shuffle=True, random_state=42)


for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    clients["clients_data"][i]= {
            "training": (X_train, y_train),
            "testing": (X_test, y_test),
        }
    

    print(f"Client {i+1}:")
    print("Train labels distribution:", y_train.value_counts())
    print("Test labels distribution:", y_test.value_counts())
    print()

# Ensure every client gets a train and test set
for client in clients["clients_data"]:
    X_train, y_train = client["training"]
    X_test, y_test = client["testing"]
    print("Client training set size:", X_train.shape[0])
    print("Client test set size:", X_test.shape[0])


def train_local_model(X_train, y_train, weight=None):
    X_train_tfidf = vectorizer.fit_transform(X_train)
    model = LogisticRegression()
    model.coef_ = weight
    model.fit(X_train_tfidf, y_train)
    return model.coef_, model.intercept_, vectorizer


# Function for federated averaging
def federated_averaging(weights_list, intercepts_list):
    avg_weights = np.mean(weights_list[0], axis=0)
    avg_intercepts = np.mean(intercepts_list, axis=0)
    return avg_weights, avg_intercepts

vectorizers = np.empty(amount, dtype=object)

global_weight = None
global_intercept = None

# Train local models and perform federated averaging
for i in range(5):
    for j in range(amount):
        X_train, y_train = clients["clients_data"][j]["training"]
        coef, intercept, vectorizer = train_local_model(
            X_train,
            y_train,
            global_weight,
        )
        clients["client_weights"][j] = coef
        clients["client_intercepts"][j] = intercept
        vectorizers[j] = vectorizer
    global_weight, global_intercept = federated_averaging(
        clients["client_weights"], clients["client_intercepts"]
    )

    # Update every client's model with global weights
    for j in range(amount):
        clients["client_weights"][j] = global_weight
        clients["client_intercepts"][j] = global_intercept

# Test the federated model on each client's test set and print results
for i, client in enumerate(clients["clients_data"]):
    X_test, y_test = client["testing"]
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.coef_ = global_weight
    model.intercept_ = global_intercept

    print(f"global_weight shape: {model.coef_.shape}")
    print(f"global_intercept shape: {model.intercept_.shape}")
    print(f"X_test_tfidf shape: {X_test_tfidf.shape}")

    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Client {i+1} Test Results:")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    print()
