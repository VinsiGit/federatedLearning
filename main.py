import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

amount = 5
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

X_train_split = np.array_split(X_train_tfidf.toarray(), amount)
y_train_split = np.array_split(y_train, amount)

for i in range(amount):
    client = {
    "data": (X_train_split[i], y_train_split[i]),
    "weight": np.zeros((1,1)),  # Adjust the shape according to the feature size
    "intercept": np.zeros(1),
}
    clients.append(client)

def train_local_model(X_local, y_local, weight=None):
    model = LogisticRegression()
    if weight is not None:
        model.coef_ = weight
    model.fit(X_local, y_local)
    return model.coef_, model.intercept_

global_model = LogisticRegression()

# Train local models and perform federated averaging
for i in range(5):
    print(f"Epoch {i+1}")
    for client in clients:
        coef, intercept = train_local_model(
            client["data"][0],
            client["data"][1],
            getattr(global_model, 'coef_', None),
        )
        client["weight"] = coef
        client["intercept"] = intercept
    
    global_model.coef_ = np.mean([client["weight"] for client in clients], axis=0)
    global_model.intercept_ = np.mean([client["intercept"] for client in clients], axis=0)
    global_model.classes_ = np.unique(y_train)
    print(global_model.coef_)

    
    # Update every client's model with global weights
    for client in clients:
        client["weight"] = global_model.coef_
        client["intercept"] = global_model.intercept_

y_pred = global_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Test Results:")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)
print()