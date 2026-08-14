import streamlit as st
import joblib
import pandas as pd

# Load saved model and feature list
model = joblib.load("thyroid_lightgbm_model.pkl")
features = joblib.load("thyroid_features.pkl")

# App title
st.title("Thyroid Disease Classification")

st.write(
    "This app predicts whether a patient is Hyperthyroid, "
    "Hypothyroid, or has Normal thyroid function."
)

st.subheader("Patient Information")

age = st.number_input(
    "Age (normalized value)",
    min_value=0.0,
    max_value=1.0,
    step=0.01
)

sex = st.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

st.write("Age:", age)
st.write("Sex:", sex)

st.subheader("Clinical Information")

on_thyroxine = st.selectbox("On Thyroxine", [0, 1])
query_on_thyroxine = st.selectbox("Query On Thyroxine", [0, 1])
on_antithyroid_medication = st.selectbox(
    "On Antithyroid Medication", [0, 1]
)
sick = st.selectbox("Sick", [0, 1])
pregnant = st.selectbox("Pregnant", [0, 1])
thyroid_surgery = st.selectbox("Thyroid Surgery", [0, 1])
I131_treatment = st.selectbox("I131 Treatment", [0, 1])
query_hypothyroid = st.selectbox("Query Hypothyroid", [0, 1])
query_hyperthyroid = st.selectbox("Query Hyperthyroid", [0, 1])
lithium = st.selectbox("Lithium", [0, 1])
goitre = st.selectbox("Goitre", [0, 1])
tumor = st.selectbox("Tumor", [0, 1])
hypopituitary = st.selectbox("Hypopituitary", [0, 1])
psych = st.selectbox("Psych", [0, 1])

st.subheader("Laboratory Measurements")

TSH = st.number_input(
    "TSH",
    min_value=0.0,
    step=0.001,
    format="%.4f"
)

T3 = st.number_input(
    "T3",
    min_value=0.0,
    step=0.001,
    format="%.4f"
)

TT4 = st.number_input(
    "TT4",
    min_value=0.0,
    step=0.001,
    format="%.4f"
)

T4U = st.number_input(
    "T4U",
    min_value=0.0,
    step=0.001,
    format="%.4f"
)

FTI = st.number_input(
    "FTI",
    min_value=0.0,
    step=0.001,
    format="%.4f"
)

# Combine all user inputs into one patient record
patient_data = {
    "age": age,
    "sex": sex,
    "on_thyroxine": on_thyroxine,
    "query_on_thyroxine": query_on_thyroxine,
    "on_antithyroid_medication": on_antithyroid_medication,
    "sick": sick,
    "pregnant": pregnant,
    "thyroid_surgery": thyroid_surgery,
    "I131_treatment": I131_treatment,
    "query_hypothyroid": query_hypothyroid,
    "query_hyperthyroid": query_hyperthyroid,
    "lithium": lithium,
    "goitre": goitre,
    "tumor": tumor,
    "hypopituitary": hypopituitary,
    "psych": psych,
    "TSH": TSH,
    "T3": T3,
    "TT4": TT4,
    "T4U": T4U,
    "FTI": FTI
}

# Convert to DataFrame
input_df = pd.DataFrame([patient_data])

# Keep the exact feature order used during training
input_df = input_df[features]

st.write("Patient input prepared successfully.")

if st.button("Predict Thyroid Class"):

    prediction = model.predict(input_df)[0]

    class_labels = {
        1: "Hyperthyroid",
        2: "Hypothyroid",
        3: "Normal"
    }

    result = class_labels[prediction]

    st.success(f"Predicted Thyroid Class: {result}")