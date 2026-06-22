import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("dataset/student-mat.csv", sep=";")

# Features
X = df[["studytime", "failures", "absences"]]
y = df["G3"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling (important for SVM + ANN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------
# Model 1: Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Model 2: SVM
svm = SVR()
svm.fit(X_train_scaled, y_train)

# Model 3: Neural Network
nn = MLPRegressor(hidden_layer_sizes=(50,50), max_iter=500)
nn.fit(X_train_scaled, y_train)

# Evaluation
models = {
    "Random Forest": rf,
    "SVM": svm,
    "Neural Network": nn
}

for name, model in models.items():
    if name == "Random Forest":
        pred = model.predict(X_test)
    else:
        pred = model.predict(X_test_scaled)

    print(name, "MAE:", mean_absolute_error(y_test, pred))

# Save best model (Random Forest)
pickle.dump(rf, open("model/model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

print("Training completed!")
##metrices
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name} Performance:")
    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R2  :", r2)
    ##apply metrices on all models
    # Random Forest
rf_pred = rf.predict(X_test)
evaluate_model("Random Forest", y_test, rf_pred)

# SVM
svm_pred = svm.predict(X_test_scaled)
evaluate_model("SVM", y_test, svm_pred)

# Neural Network
nn_pred = nn.predict(X_test_scaled)
evaluate_model("Neural Network", y_test, nn_pred)