**Multiclass Thyroid Disease Classification**

Machine-learning project for classifying thyroid function into three ANN-Thyroid dataset categories:

Hyperthyroid

Hypothyroid

Normal

The project compares supervised learning models, addresses class imbalance, evaluates model performance, and deploys the selected model through Streamlit.

**Dataset**

The project uses the UCI ANN-Thyroid dataset with 7,200 observations and 22 variables, including demographic, clinical, treatment, and thyroid laboratory features.

The dataset is highly imbalanced, with Normal cases forming the majority class. It does not separately classify subclinical thyroid dysfunction and does not contain imaging, cytology, or histopathology data.

**Workflow**

Data cleaning → Exploratory data analysis → Preprocessing → Class-imbalance handling → Model training → Model comparison → Hyperparameter tuning → Explainability → Deployment

Models evaluated include Logistic Regression, Decision Tree, Random Forest, SVM, KNN, XGBoost, and LightGBM.

**Final Model**

Class-Weighted LightGBM was selected as the final model.

Metric

Result

Accuracy

99.86%

Macro F1-score

0.9953

Balanced Accuracy

0.9995

The most influential features were FTI, TSH, T3, TT4, and T4U.

**Project Structure**

Multi-class-Thyroid-Disease-Classification/
│
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Cleaned dataset
├── models/                  # Saved model and feature files
├── notebooks/               # Analysis/model-development notebook
├── .gitignore
├── app.py                   # Streamlit application
├── LICENSE
├── README.md
└── requirements.txt

**Run the Project**

Install dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py
**
**Limitations**

The model is limited to the three classes represented in the ANN-Thyroid dataset. Subclinical thyroid dysfunction, imaging-based assessment, structural thyroid disease, malignancy, and disease aetiology are outside its scope. The continuous age and laboratory variables are also provided in standardized rather than ordinary clinical units.

External validation is required before broader clinical application.

**Disclaimer**

This project is intended for machine-learning research and educational use. Model predictions should not replace established clinical diagnostic protocols or professional clinical judgement.