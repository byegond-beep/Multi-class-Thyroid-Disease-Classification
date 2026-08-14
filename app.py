import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------
st.set_page_config(
    page_title="Thyroid Disease Classifier",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    h1, h2, h3 {
        color: #1f5f6b;
    }

    .stButton > button {
        background-color: #2a788e;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #22a884;
        color: white;
    }

    .result-box {
        padding: 1.3rem;
        border-radius: 12px;
        background-color: #e8f5f2;
        border-left: 6px solid #22a884;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }

    .subtitle {
        color: #52757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .info-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f2f8f7;
        margin-top: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = joblib.load("thyroid_lightgbm_model.pkl")
features = joblib.load("thyroid_features.pkl")

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🩺 Thyroid Disease Classification")

st.markdown(
    """
    <div class="subtitle">
    Enter patient information below to generate a prediction for
    Hyperthyroidism, Hypothyroidism, or Normal thyroid function.
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# PATIENT INFORMATION
# --------------------------------------------------
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age (normalized value)",
        min_value=0.0,
        max_value=1.0,
        step=0.01
    )

with col2:
    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )

# --------------------------------------------------
# CLINICAL INFORMATION
# --------------------------------------------------
st.subheader("Clinical Information")

c1, c2, c3 = st.columns(3)

with c1:
    on_thyroxine = st.selectbox("On Thyroxine", [0, 1])
    sick = st.selectbox("Sick", [0, 1])
    thyroid_surgery = st.selectbox("Thyroid Surgery", [0, 1])
    query_hypothyroid = st.selectbox("Query Hypothyroid", [0, 1])
    lithium = st.selectbox("Lithium", [0, 1])

with c2:
    query_on_thyroxine = st.selectbox(
        "Query On Thyroxine", [0, 1]
    )

    pregnant = st.selectbox("Pregnant", [0, 1])

    I131_treatment = st.selectbox(
        "I131 Treatment", [0, 1]
    )

    query_hyperthyroid = st.selectbox(
        "Query Hyperthyroid", [0, 1]
    )

    goitre = st.selectbox("Goitre", [0, 1])

with c3:
    on_antithyroid_medication = st.selectbox(
        "On Antithyroid Medication", [0, 1]
    )

    tumor = st.selectbox("Tumor", [0, 1])

    hypopituitary = st.selectbox(
        "Hypopituitary", [0, 1]
    )

    psych = st.selectbox("Psych", [0, 1])

# --------------------------------------------------
# LABORATORY MEASUREMENTS
# --------------------------------------------------
st.subheader("Laboratory Measurements")

l1, l2, l3, l4, l5 = st.columns(5)

with l1:
    TSH = st.number_input(
        "TSH",
        min_value=0.0,
        step=0.001,
        format="%.4f"
    )

with l2:
    T3 = st.number_input(
        "T3",
        min_value=0.0,
        step=0.001,
        format="%.4f"
    )

with l3:
    TT4 = st.number_input(
        "TT4",
        min_value=0.0,
        step=0.001,
        format="%.4f"
    )

with l4:
    T4U = st.number_input(
        "T4U",
        min_value=0.0,
        step=0.001,
        format="%.4f"
    )

with l5:
    FTI = st.number_input(
        "FTI",
        min_value=0.0,
        step=0.001,
        format="%.4f"
    )

# --------------------------------------------------
# PREPARE PATIENT DATA
# --------------------------------------------------
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

input_df = pd.DataFrame([patient_data])
input_df = input_df[features]

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
st.divider()

if st.button("Predict Thyroid Class"):

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    class_labels = {
        1: "Hyperthyroid",
        2: "Hypothyroid",
        3: "Normal"
    }

    result = class_labels[prediction]

    # Prediction result
    st.markdown(
        f"""
        <div class="result-box">
            <h3>Prediction: {result}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------
    # PREDICTION CONFIDENCE
    # --------------------------------------------------
    st.subheader("Prediction Confidence")

    probability_data = []

    for model_class, probability in zip(
        model.classes_,
        probabilities
    ):
        probability_data.append({
            "Thyroid Class": class_labels[model_class],
            "Probability (%)": round(probability * 100, 2)
        })

    probability_df = pd.DataFrame(probability_data)

    st.dataframe(
        probability_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # MODEL EXPLAINABILITY
    # --------------------------------------------------
    st.subheader("Model Explainability")

    feature_importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    feature_importance = (
        feature_importance
        .sort_values("Importance", ascending=False)
        .head(10)
    )

    st.write(
        "The chart below shows the features that contribute most "
        "strongly to the model overall."
    )

    st.bar_chart(
        feature_importance.set_index("Feature")
    )

    top_features = feature_importance["Feature"].head(5).tolist()

    st.markdown(
        f"""
        <div class="info-box">
        The model relies most strongly on
        <b>{", ".join(top_features)}</b>.
        These values represent global model importance and do not
        indicate that a feature directly caused this individual prediction.
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------
st.divider()

st.caption(
    "For educational and research purposes only. "
    "This application is not a substitute for professional "
    "medical diagnosis."
)