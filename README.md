# Multiclass Thyroid Disease Classification

## **Project Overview**

This project develops a machine-learning model to classify thyroid function into three categories:

- Hyperthyroid
- Hypothyroid
- Normal

The project uses demographic, clinical, treatment, and laboratory variables from the ANN-Thyroid dataset. Multiple supervised machine-learning models were evaluated, with Class-Weighted LightGBM selected as the final model.

## **Dataset**

The project uses the ANN-Thyroid dataset from the UCI Machine Learning Repository.

The dataset contains:

- 7,200 observations
- 22 variables
- Demographic, clinical, treatment, and laboratory features
- Three target classes: Hyperthyroid, Hypothyroid, and Normal

The dataset is highly imbalanced, with Normal cases forming the majority class.

## **Project Workflow**

Data Cleaning → Exploratory Data Analysis → Preprocessing → Class Imbalance Handling → Model Training → Model Comparison → Hyperparameter Tuning → Explainability → Deployment

Models evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors
- XGBoost
- LightGBM

## **Final Model**

Class-Weighted LightGBM was selected as the final model.

| Metric | Result |
|---|---:|
| Accuracy | 99.86% |
| Macro F1-score | 0.9953 |
| Balanced Accuracy | 0.9995 |

The most influential features were FTI, TSH, T3, TT4, and T4U.

## **Project Structure**

```text
Multi-class-Thyroid-Disease-Classification/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── .gitignore
├── app.py
├── LICENSE
├── README.md
└── requirements.txt
```

- `data/raw/` – original dataset
- `data/processed/` – cleaned dataset
- `models/` – saved model and feature files
- `notebooks/` – analysis and model-development notebook
- `app.py` – Streamlit application
- `requirements.txt` – project dependencies

## **Running the Project**

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## **Limitations**

The model is limited to the three thyroid-function classes represented in the ANN-Thyroid dataset. Subclinical thyroid dysfunction, imaging findings, structural thyroid disease, malignancy, and disease aetiology are outside the scope of the model.

Age and laboratory variables are provided in standardized numerical form rather than conventional clinical units. External validation is required before broader clinical application.

## **Data Source**

UCI Machine Learning Repository – Thyroid Disease.

## **Disclaimer**

This project is intended for machine-learning research and educational purposes. Model predictions should not replace established clinical diagnostic protocols or professional clinical judgement.