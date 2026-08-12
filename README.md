# Multi-class-Thyroid-Disease-Classification
A multiclass machine learning project using the UCI ANN-Thyroid dataset to classify patients as Hyperthyroid, Hypothyroid, or Normal. The project compares multiple classification algorithms, evaluates their predictive performance, and identifies the clinical and laboratory features most associated with thyroid disease.

The objective is to classify patients into three thyroid status categories:

* Class 1 — Hyperthyroidism
* Class 2 — Hypothyroidism
* Class 3 — Normal thyroid function

A key challenge addressed in the project is severe class imbalance, with Normal cases accounting for approximately 92.5% of observations. The analysis therefore prioritizes evaluation metrics that measure performance across all three classes rather than relying on overall accuracy alone.

## Project Objectives

The project aims to:

* Explore the structure and quality of the thyroid dataset.
* Examine relationships between clinical and laboratory variables and thyroid diagnosis.
* Assess the impact of class imbalance on model performance.
* Compare multiple machine learning classification algorithms.
* Evaluate class weighting and SMOTE as imbalance-handling strategies.
* Identify the strongest-performing model.
* Optimize the selected model through hyperparameter tuning.
* Identify the features that contribute most strongly to thyroid classification.

## Dataset

The project uses a thyroid disease dataset containing:

* 7,200 observations
* 21 predictor variables
* 1 multiclass target variable

The predictors include demographic information, clinical indicators, treatment history, and thyroid laboratory measurements such as:

* TSH
* T3
* TT4
* T4U
* FTI
* Age
* Sex
* Thyroxine treatment status
* Thyroid surgery history
* I131 treatment
* Thyroid-related query indicators
* Other clinical binary variables

### Target Distribution

The dataset is highly imbalanced:

| Class | Diagnosis       | Approximate Proportion |
| ----- | --------------- | ---------------------: |
| 1     | Hyperthyroidism |                   2.3% |
| 2     | Hypothyroidism  |                   5.1% |
| 3     | Normal          |                  92.6% |

Because of this imbalance, Macro F1-score and Balanced Accuracy are emphasized during model evaluation.

## Project Workflow

The notebook follows the machine learning workflow below:

1. Data Understanding
2. Data Quality Assessment
3. Exploratory Data Analysis
4. Data Preparation
5. Model Development and Evaluation
6. Model Explainability
7. Final Findings and Conclusion

### Exploratory Data Analysis

EDA includes:

* Target class distribution
* Numerical feature distributions
* Binary feature distributions
* Outlier and skewness assessment
* Numerical features versus thyroid diagnosis
* Binary features versus thyroid diagnosis
* Kruskal-Wallis statistical testing
* Chi-square tests of association
* Multivariate correlation analysis

The analysis identified substantial differences in several thyroid laboratory measurements across diagnostic classes.

## Data Preparation

The preprocessing workflow includes:

* Removal of duplicate observations
* Stratified train-test splitting
* Preservation of the original three-class target
* Feature scaling for scale-sensitive algorithms
* Class weighting
* SMOTE oversampling

SMOTE is applied only to the training data to prevent information leakage into the test set.

## Machine Learning Models

Seven classification algorithms are evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* XGBoost
* LightGBM

Where supported, each algorithm is evaluated under three strategies:

1. Original imbalanced training data
2. Class weighting
3. SMOTE

KNN is evaluated using the Original and SMOTE strategies because it does not directly support class weighting.

## Evaluation Metrics

Given the severe target imbalance, model evaluation emphasizes performance across all three diagnostic classes.

The following metrics are used:

* Accuracy
* Balanced Accuracy
* Macro Precision
* Macro Recall
* Macro F1-score
* Confusion Matrix

Macro F1-score is used as the primary model-comparison metric.

## Model Comparison

Tree-based models substantially outperform the linear and distance-based approaches.

The strongest baseline configurations are:

| Model         | Strategy       | Accuracy | Balanced Accuracy | Macro F1 |
| ------------- | -------------- | -------: | ----------------: | -------: |
| XGBoost       | Class Weighted |   0.9986 |            0.9995 |   0.9953 |
| LightGBM      | Class Weighted |   0.9986 |            0.9995 |   0.9953 |
| Decision Tree | Class Weighted |   0.9993 |            0.9899 |   0.9947 |

Class-Weighted XGBoost and LightGBM achieve identical test-set performance and therefore undergo additional cross-validation.

## Final Model Selection

Five-fold stratified cross-validation is used to compare the two leading models.

| Model                     | Mean CV Macro F1 | Standard Deviation |
| ------------------------- | ---------------: | -----------------: |
| XGBoost — Class Weighted  |           0.9724 |             0.0101 |
| LightGBM — Class Weighted |           0.9751 |             0.0095 |

Class-Weighted LightGBM is selected because it achieves a slightly higher mean cross-validation Macro F1-score with slightly lower variability.

## Hyperparameter Tuning

The selected LightGBM model is optimized using RandomizedSearchCV with stratified five-fold cross-validation.

The selected hyperparameters include:

| Hyperparameter    | Selected Value |
| ----------------- | -------------: |
| n_estimators      |            300 |
| learning_rate     |           0.05 |
| num_leaves        |             15 |
| min_child_samples |             30 |
| max_depth         |             -1 |
| subsample         |           0.70 |
| colsample_bytree  |           0.70 |

Hyperparameter tuning improves the cross-validation Macro F1-score from:

`0.9751 → 0.9793`

## Final Model Performance

The tuned Class-Weighted LightGBM model achieves:

| Metric            |  Score |
| ----------------- | -----: |
| Accuracy          | 0.9986 |
| Balanced Accuracy | 0.9995 |
| Macro Precision   | 0.9912 |
| Macro Recall      | 0.9995 |
| Macro F1          | 0.9953 |

The final confusion matrix shows that the model correctly classifies:

* 33 of 33 Hyperthyroid cases
* 74 of 74 Hypothyroid cases
* 1,317 of 1,319 Normal cases

Only two Normal observations are misclassified as Hypothyroid.

## Model Explainability

LightGBM feature importance is used to examine the predictors contributing most strongly to model decisions.

The most influential features include:

1. FTI
2. TSH
3. T3
4. TT4
5. T4U
6. Age

Overall, thyroid-related laboratory measurements contribute substantially more to classification than most binary clinical variables.

## Key Insights

* Severe class imbalance can make overall accuracy misleading.
* Macro F1 and Balanced Accuracy provide a more informative assessment of multiclass performance.
* Class imbalance handling has different effects depending on the algorithm.
* SMOTE substantially improves minority-class detection for Logistic Regression and SVM.
* Class weighting performs particularly well with the strongest tree-based models.
* Tree-based models outperform Logistic Regression, SVM, and KNN in this dataset.
* Hyperparameter tuning improves LightGBM cross-validation performance, although its already strong test performance remains unchanged.
* Thyroid laboratory measurements are the dominant predictors of diagnostic class.

## Technologies and Libraries

The project is implemented in Python using:

* pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* scikit-learn
* imbalanced-learn
* XGBoost
* LightGBM

## Running the Notebook

Clone or download the project and ensure the required Python packages are installed.

Example:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn xgboost lightgbm
```

Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

Then open the project notebook and execute the cells sequentially from top to bottom.

## Limitations

* Hyperthyroid and Hypothyroid observations are substantially less frequent than Normal cases.
* The model has been developed and evaluated using a single dataset.
* External validation on an independent patient population has not been conducted.
* Synthetic observations generated through SMOTE may not fully represent real clinical patient profiles.
* Near-perfect test performance should be interpreted cautiously until independently validated.
* Feature importance represents predictive contribution and should not be interpreted as evidence of causality.

## Conclusion

The project demonstrates that machine learning can effectively distinguish Hyperthyroid, Hypothyroid, and Normal thyroid cases when class imbalance is appropriately addressed.

Among the evaluated algorithms, Class-Weighted LightGBM provides the strongest and most stable overall performance. Following hyperparameter tuning, the final model achieves a Macro F1-score of 0.9953 and Balanced Accuracy of 0.9995 on the test set.

FTI, TSH, T3, TT4, and T4U emerge as the most influential predictors, highlighting the importance of thyroid-related laboratory measurements in classification.

Further validation using independent and more diverse datasets is recommended before considering broader practical or clinical application.

## Disclaimer

This project is intended for educational and research purposes only. The developed machine learning model is not a medical diagnostic tool and should not replace professional clinical assessment or medical judgment.

