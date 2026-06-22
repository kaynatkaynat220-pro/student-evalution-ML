import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Performance Evaluation",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# -----------------------------
# Title
# -----------------------------
st.title("🎓 Student Performance Evaluation System")

st.write(
    """
This application predicts a student's final performance based on:

- Study Time
- Number of Failures
- Absences

The prediction is performed using a Machine Learning model.
"""
)

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Enter Student Details")

studytime = st.sidebar.slider(
    "Study Time (1-4)",
    min_value=1,
    max_value=4,
    value=2
)

failures = st.sidebar.slider(
    "Previous Failures",
    min_value=0,
    max_value=4,
    value=0
)

absences = st.sidebar.slider(
    "Absences",
    min_value=0,
    max_value=100,
    value=5
)

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("Predict Performance"):

    input_data = np.array([[studytime, failures, absences]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    if prediction >= 10:
        status = "PASS 🎉"
        st.success(f"Predicted Grade: {prediction:.2f}")
        st.success(status)
    else:
        status = "FAIL ❌"
        st.error(f"Predicted Grade: {prediction:.2f}")
        st.error(status)

    st.subheader("Student Input Analysis")

    fig, ax = plt.subplots(figsize=(6,4))

    features = ["Study Time", "Failures", "Absences"]
    values = [studytime, failures, absences]

    colors = ["green", "red", "orange"]

    ax.bar(features, values, color=colors)
    ax.set_ylabel("Value")
    ax.set_title("Student Performance Factors")

    st.pyplot(fig)

st.divider()

st.markdown(
"""
### Project Information

**Machine Learning Algorithm:** Random Forest Regression

**Input Features**

- Study Time
- Failures
- Absences

**Output**

- Predicted Final Grade (G3)
- PASS / FAIL Status
"""
)
