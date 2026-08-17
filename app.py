import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from openai import OpenAI

# --------------------------------------------------
# OPENAI CLIENT
# --------------------------------------------------
client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

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

st.caption(
    "ℹ️ Age is represented in the standardized numerical format used by the model, "
    "so the value shown on the screen may look different from age in years. "
    "This is expected."
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age (standardized value)",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        help=(
            "The model was trained using standardized age values "
            "from the ANN-Thyroid dataset."
        )
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

st.caption(
    "Select Yes or No for each item. The app automatically converts "
    "your selections into the numerical format required by the model."
)

def yes_no(label, help_text=None):
    return st.selectbox(
        label,
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help=help_text
    )

c1, c2, c3 = st.columns(3)

with c1:
    on_thyroxine = yes_no(
        "Currently taking thyroxine?"
    )

    sick = yes_no(
        "Currently unwell?"
    )

    thyroid_surgery = yes_no(
        "Previous thyroid surgery?"
    )

    query_hypothyroid = yes_no(
        "Hypothyroidism suspected?",
        "Select Yes if hypothyroidism is being considered or investigated."
    )

    lithium = yes_no(
        "Currently taking lithium?"
    )

with c2:
    query_on_thyroxine = yes_no(
        "Thyroxine use uncertain?"
    )

    pregnant = yes_no(
        "Currently pregnant?"
    )

    I131_treatment = yes_no(
        "Previous radioactive iodine (I-131) treatment?"
    )

    query_hyperthyroid = yes_no(
        "Hyperthyroidism suspected?",
        "Select Yes if hyperthyroidism is being considered or investigated."
    )

    goitre = yes_no(
        "Goitre present?"
    )

with c3:
    on_antithyroid_medication = yes_no(
        "Currently taking antithyroid medication?"
    )

    tumor = yes_no(
        "Tumor present?"
    )

    hypopituitary = yes_no(
        "Hypopituitarism present?"
    )

    psych = yes_no(
        "Psychiatric condition recorded?"
    )
# --------------------------------------------------
# LABORATORY MEASUREMENTS
# --------------------------------------------------
st.subheader("Laboratory Measurements")

st.caption(
    "ℹ️ Laboratory values are represented in the standardized numerical "
    "format used by the model, so the values shown on the screen may look "
    "different from those on a lab report. This is expected."
)

l1, l2, l3, l4, l5 = st.columns(5)

with l1:
    TSH = st.number_input(
        "TSH",
        min_value=0.0,
        step=0.001,
        format="%.4f",
        help="Standardized TSH value used by the ANN-Thyroid model."
    )

with l2:
    T3 = st.number_input(
        "T3",
        min_value=0.0,
        step=0.001,
        format="%.4f",
        help="Standardized T3 value used by the ANN-Thyroid model."
    )

with l3:
    TT4 = st.number_input(
        "TT4",
        min_value=0.0,
        step=0.001,
        format="%.4f",
        help="Standardized TT4 value used by the ANN-Thyroid model."
    )

with l4:
    T4U = st.number_input(
        "T4U",
        min_value=0.0,
        step=0.001,
        format="%.4f",
        help="Standardized T4U value used by the ANN-Thyroid model."
    )

with l5:
    FTI = st.number_input(
        "FTI",
        min_value=0.0,
        step=0.001,
        format="%.4f",
        help="Standardized FTI value used by the ANN-Thyroid model."
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

# --------------------------------------------------
# THYROID AI ASSISTANT
# --------------------------------------------------

st.divider()

st.subheader("💬 Thyroid AI Assistant")

st.caption(
    "Ask me about thyroid conditions, laboratory terms, model predictions, "
    "or how to understand this app."
)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Show previous messages
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_question = st.chat_input(
    "Ask the Thyroid AI Assistant..."
)

if user_question:

    # Display user's question
    st.session_state.chat_messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    try:
        response = client.responses.create(
            model="gpt-5-mini",
            instructions="""
            You are the AI assistant for a machine-learning thyroid
            disease classification application.

            Explain thyroid-related concepts and the application's
            machine-learning results in simple, accessible language.

            You may explain:
            - hypothyroidism
            - hyperthyroidism
            - normal thyroid function
            - TSH, T3, TT4, T4U and FTI
            - prediction confidence
            - feature importance
            - how the machine-learning application works
            - limitations of the model

            Do not claim that the model provides a medical diagnosis.
            Do not prescribe medication or treatment.
            Clearly distinguish a model prediction from a clinical diagnosis.

            When a question requires medical evaluation, encourage the
            user to discuss their results with a qualified healthcare
            professional.

            Keep answers concise, friendly and understandable to
            someone without a medical or technical background.
            """,
            input=user_question
        )

        assistant_reply = response.output_text

    except Exception:
        assistant_reply = (
            "I'm sorry, I couldn't connect to the AI assistant right now. "
            "Please try again shortly."
        )

    st.session_state.chat_messages.append(
        {"role": "assistant", "content": assistant_reply}
    )

    with st.chat_message("assistant"):
        st.markdown(assistant_reply)