Multiclass Thyroid Disease Classification Using Machine Learning

Project Overview

This project develops and evaluates supervised machine-learning models for multiclass thyroid-function classification using the UCI ANN-Thyroid dataset. The prediction task is limited to the three outcome classes represented in the dataset:

Class 1: Hyperthyroidism

Class 2: Hypothyroidism

Class 3: Normal thyroid function

The project follows an end-to-end machine-learning workflow covering data understanding, exploratory data analysis, preprocessing, class-imbalance handling, model comparison, hyperparameter tuning, model explainability, and deployment.

The final selected model is a class-weighted LightGBM classifier. The deployed prototype is intended as a machine-learning classification and educational tool and is not a replacement for established clinical diagnostic protocols or professional clinical judgement.

Clinical Scope

Thyroid disorders are diagnosed using established clinical processes that combine clinical assessment and thyroid-function testing, with imaging and other investigations used when clinically indicated. This project investigates how machine learning can complement that process by identifying patterns across the demographic, clinical, treatment, and laboratory variables available in the ANN-Thyroid dataset.

The dataset imposes important clinical boundaries. It does not provide separate target classes for subclinical hyperthyroidism or subclinical hypothyroidism and does not contain the imaging, cytology, or histopathology information required to assess thyroid nodules, TI-RADS categories, structural thyroid disease, malignancy, or disease aetiology. The model should therefore be interpreted strictly as a classifier for the three dataset-defined thyroid-function categories.

Key Questions

Can machine-learning models accurately classify patients into Hyperthyroid, Hypothyroid, and Normal thyroid-function categories using the demographic, clinical, and laboratory variables available in the ANN-Thyroid dataset?

Which clinical and laboratory features contribute most strongly to distinguishing the three thyroid-function classes?

Which supervised machine-learning algorithm provides the strongest and most balanced multiclass classification performance?

Success Metrics

Because the dataset is highly imbalanced, model selection emphasizes balanced multiclass performance rather than overall accuracy alone.

Macro F1-score ≥ 0.90

Balanced Accuracy ≥ 0.90

Recall ≥ 0.85 for both Hyperthyroid and Hypothyroid classes

Dataset

The project uses the ANN-Thyroid data associated with the UCI Thyroid Disease collection. The original combined dataset contains 7,200 observations and 22 variables, including the target.

The predictors include:

Demographic variables: age and sex

Clinical/treatment indicators: on thyroxine, query on thyroxine, antithyroid medication, sick, pregnant, thyroid surgery, I131 treatment, query hypothyroid, query hyperthyroid, lithium, goitre, tumor, hypopituitary, and psych

Laboratory variables: TSH, T3, TT4, T4U, and FTI

Target: class

The continuous variables in the ANN-Thyroid data are supplied in a standardized numerical representation rather than ordinary raw clinical units. They should therefore not be interpreted directly as age in years or conventional laboratory-report values.

Data Quality

Initial assessment found:

7,200 records

No missing values

71 duplicate records

7,129 records after duplicate removal

Severe class imbalance, with Normal cases forming the large majority of observations

The cleaned data are used for subsequent preprocessing and model development.

Exploratory Data Analysis

EDA includes:

Target class distribution

Numerical-variable distributions

Binary clinical-variable distributions

Numerical variables versus target class

Kruskal-Wallis statistical testing

Binary variables versus target class

Chi-square testing

Correlation analysis

Assessment of class imbalance and potentially informative predictors

The analysis identified thyroid laboratory measurements, particularly TSH, FTI, TT4, and T3, as important variables for distinguishing the three classes.

Data Preprocessing

The preprocessing workflow includes:

Duplicate removal

Predictor and target separation

Stratified 80:20 train-test split

Evaluation of the original imbalanced training data

Class weighting

SMOTE oversampling

Standardization for scale-sensitive algorithms

Scaling is applied to models such as Logistic Regression, SVM, and KNN. Tree-based models are trained on unscaled features because their threshold-based splitting does not require feature standardization.

Models Evaluated

The following supervised classification algorithms are evaluated:

Logistic Regression

Decision Tree

Random Forest

Support Vector Machine

K-Nearest Neighbors

XGBoost

LightGBM

Models are compared across the original data, class-weighted approaches, and SMOTE-based training where appropriate.

Model Evaluation

Given the class imbalance, the project prioritizes:

Macro F1-score

Balanced Accuracy

Class-specific precision and recall

Confusion matrices

Cross-validation performance

The strongest candidate models are further assessed using cross-validation before final model selection and hyperparameter tuning.

Final Model

Class-Weighted LightGBM was selected as the final model because of its strong test-set performance and stable cross-validation results.

Final tuned LightGBM performance:

Metric

Result

Accuracy

99.86%

Macro F1-score

0.9953

Balanced Accuracy

0.9995

The final model correctly classified all Hyperthyroid and Hypothyroid test cases, with only two Normal cases misclassified as Hypothyroid.

These near-perfect results should be interpreted cautiously because they may reflect the relatively clean ANN-Thyroid dataset, strong class-separating laboratory patterns, feature redundancy, or similarity between observations in the train and test partitions. External validation is required before broader clinical interpretation.

Model Explainability

Global LightGBM feature importance is used to examine which predictors contribute most strongly to model classification.

The most influential features include:

FTI

TSH

T3

TT4

T4U

Feature importance reflects predictive contribution within the model and should not be interpreted as evidence of causality or as a patient-specific clinical explanation.

Deployment

The final LightGBM classifier is serialized for use in the Streamlit application.

Saved model artifacts:

thyroid_lightgbm_model.pkl — trained final LightGBM classifier

thyroid_features.pkl — feature names/order expected by the model

The Streamlit interface provides a user-facing prediction workflow and model information. Because the ANN-Thyroid continuous variables are already standardized, the current prototype must preserve the same numerical representation expected by the trained model.

A future deployment-oriented version should be trained using raw clinical measurements, or use a fully documented preprocessing pipeline, so that users can enter age in years and laboratory values in familiar clinical units.

Project Structure

Multi-class-Thyroid-Disease-Classification/
│
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Cleaned dataset
│
├── models/                  # Saved model and feature files
├── notebooks/               # Analysis and model-development notebook
│
├── .gitignore
├── app.py                   # Streamlit application
├── LICENSE
├── README.md
└── requirements.txt

The structure separates raw and cleaned data, trained model artifacts, notebook analysis, and deployment files. Keep API keys and .streamlit/secrets.toml out of GitHub.

Running the Project

1. Clone the repository

git clone <your-repository-url>
cd Multi-class-Thyroid-Disease-Classification

2. Create and activate a virtual environment

python -m venv venv

Windows:

venv\Scripts\activate

macOS/Linux:

source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Run the notebook

Open notebooks/thyroid_disease_classification.ipynb in Jupyter Notebook, JupyterLab, or VS Code and run the cells in sequence.

5. Run the Streamlit application

streamlit run app.py

Main Python Libraries

The project uses libraries including:

pandas

NumPy

Matplotlib

Seaborn

SciPy

scikit-learn

imbalanced-learn

XGBoost

LightGBM

joblib

Streamlit

Refer to requirements.txt for the deployment environment's exact package requirements.

Limitations

Hyperthyroid and Hypothyroid observations are substantially less frequent than Normal observations.

The model has been developed and evaluated on a single dataset and has not been externally validated.

SMOTE-generated observations may not fully represent real clinical patient profiles.

The unusually high test performance may partly reflect characteristics of the ANN-Thyroid dataset.

Age and laboratory variables are supplied in standardized numerical form rather than ordinary clinical units.

Subclinical hyperthyroidism and subclinical hypothyroidism are not separately represented as target classes.

Imaging, cytology, histopathology, and TI-RADS information are not available, preventing assessment of structural thyroid disease and malignancy.

Global feature importance does not establish causality or provide patient-level clinical explanations.

The model is intended to support classification within the dataset-defined categories and should not replace established diagnostic protocols or clinical judgement.

Future Improvements

Future work could include:

External validation using independent and more diverse patient populations

Development using raw age and laboratory measurements in clinically familiar units

A reproducible end-to-end preprocessing and prediction pipeline

Evaluation of probability calibration

Patient-level explainability methods

Expansion to clinically relevant thyroid states if suitable datasets become available

Integration of imaging or pathology information where appropriate and supported by the data

References

American Thyroid Association. Thyroid function tests [Internet]. Falls Church (VA): American Thyroid Association. Available from: https://www.thyroid.org/thyroid-function-tests/

Van Uytfanghe K, Ehrenkranz J, Halsall D, Hoff K, Loh TP, Spencer CA, et al. Thyroid stimulating hormone and thyroid hormones (triiodothyronine and thyroxine): an American Thyroid Association-commissioned review of current clinical and laboratory status. Thyroid. 2023;33(9).

Biondi B, Cooper DS. The clinical significance of subclinical thyroid dysfunction. Endocr Rev. 2008;29(1):76-131.

Tessler FN, Middleton WD, Grant EG, Hoang JK, Berland LL, Teefey SA, et al. ACR Thyroid Imaging, Reporting and Data System (TI-RADS): white paper of the ACR TI-RADS Committee. J Am Coll Radiol. 2017;14(5):587-595.

Liu J, Wang Y, Da D, Zheng M. Hyperfunctioning thyroid carcinoma: a systematic review. Mol Clin Oncol. 2019;11(6):535-550.

UCI Machine Learning Repository. Thyroid Disease [Internet]. Irvine (CA): University of California, Irvine. Available from: https://archive.ics.uci.edu/dataset/102/thyroid+disease

Disclaimer

This project is intended for machine-learning research and educational demonstration. Predictions from the model should not be interpreted as a comprehensive thyroid diagnosis or used as a substitute for professional medical evaluation