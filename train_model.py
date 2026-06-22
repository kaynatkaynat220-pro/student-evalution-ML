import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("dataset/student-mat.csv", sep=";")

# Features
X = df[["studytime", "failures", "absences"]]

# Target
y = df["G3"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Models
# -----------------------------
rf = RandomForestRegressor(random_state=42)

svm = SVR()

nn = MLPRegressor(
    hidden_layer_sizes=(50,50),
    max_iter=500,
    random_state=42
)

# -----------------------------
# Training
# -----------------------------
rf.fit(X_train, y_train)

svm.fit(X_train_scaled, y_train)

nn.fit(X_train_scaled, y_train)

# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_model(name, y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_true, y_pred)

    print("\n==============================")

    print(name)

    print("==============================")

    print("MAE :", mae)

    print("MSE :", mse)

    print("RMSE:", rmse)

    print("R2  :", r2)

# -----------------------------
# Predictions
# -----------------------------
rf_pred = rf.predict(X_test)

svm_pred = svm.predict(X_test_scaled)

nn_pred = nn.predict(X_test_scaled)

# -----------------------------
# Metrics
# -----------------------------
evaluate_model(
    "Random Forest",
    y_test,
    rf_pred
)

evaluate_model(
    "SVM",
    y_test,
    svm_pred
)

evaluate_model(
    "Neural Network",
    y_test,
    nn_pred
)

# -----------------------------
# Save Model
# -----------------------------
pickle.dump(
    rf,
    open("model/model.pkl", "wb")
)

pickle.dump(
    scaler,
    open("model/scaler.pkl", "wb")
)

print("\nModel Saved Successfully!")
