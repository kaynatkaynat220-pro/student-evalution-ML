## back end
import os
import numpy as np
import pickle

from flask import Flask, render_template, request

# 🔥 matplotlib imports
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

print("Current Folder:", os.getcwd())
print("Templates Folder Exists:", os.path.exists("templates"))

# Load model + scaler
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict_page():
    return render_template("predict.html")

@app.route("/result", methods=["POST"])
def result():

    # ---------------- INPUT ----------------
    studytime = int(request.form["studytime"])
    failures = int(request.form["failures"])
    absences = int(request.form["absences"])

    input_data = np.array([[studytime, failures, absences]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    # ---------------- PASS / FAIL ----------------
    if prediction >= 10:
        status = "PASS 🎉"
    else:
        status = "FAIL ❌"

    # ---------------- GRAPH ----------------
    fig, ax = plt.subplots()

    features = ["Study Time", "Failures", "Absences"]
    values = [studytime, failures, absences]

    ax.bar(features, values, color=["green", "red", "orange"])
    ax.set_title("Student Input Analysis")

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)

    graph_url = base64.b64encode(img.getvalue()).decode()

    # ---------------- RETURN ----------------
    return render_template(
        "result.html",
        prediction=round(prediction, 2),
        status=status,
        graph=graph_url
    )

if __name__ == "__main__":
    app.run(debug=True)