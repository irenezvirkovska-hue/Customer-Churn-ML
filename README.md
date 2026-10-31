# Customer Churn ML

A machine learning course project by **Iryna Zvirkovska** for predicting customer churn from anonymized telecom data. The notebook explores preprocessing, model comparison, class imbalance, hyperparameter tuning, feature engineering and decision thresholds, using **Balanced Accuracy** as the main evaluation metric.

## Notebook

- [View the notebook](customer_churn_prediction.ipynb)
- [Open the original in Google Colab](https://colab.research.google.com/drive/1sqivMZRNqCGqA9uaxvY6KuytZkqfgmtx)

## Tools

Python, pandas, NumPy, scikit-learn, LightGBM, XGBoost, CatBoost, imbalanced-learn, Optuna, Matplotlib, Seaborn and Google Colab.

## Project workflow

- Explore missing values, feature types, correlations and the imbalanced target.
- Prepare numerical and categorical features using imputation, one-hot encoding and preprocessing pipelines.
- Compare logistic regression and ensemble classifiers.
- Evaluate class weighting and SMOTE, and tune models with RandomizedSearchCV and Optuna.
- Investigate correlated features, missing-value indicators, feature importance and classification errors.
- Evaluate quartile-based features and probability thresholds; generate and check submission files.

## Data

The notebook loads the training data, test data and submission template from the [GoIT Academy course repository](https://github.com/goitacademy/MACHINE-LEARNING-NEO/tree/main/competition). The recorded shapes are 10,000 training rows with 231 columns, including the target, and 2,500 test rows with 230 columns. Dataset files are not duplicated in this repository.

## Recorded results

The notebook identifies **D5** as the selected submission: balanced LightGBM with 212 non-empty input columns, 12 quartile flags and a classification threshold of **0.30**. It reports a Kaggle Public Score of **0.8977** for `submission_final_d5_threshold_030.csv`.

This public score is reported by the notebook; it was not independently checked against the live Kaggle leaderboard during publication. Local validation and threshold-selected out-of-fold scores are development estimates because the data informed multiple modeling decisions. In the matched D7 comparison, adding quartile flags did not improve the pooled out-of-fold score at the fixed threshold. A matching submission fingerprint checks ordered predictions, not predictive quality.

## How to run

1. Open the notebook in Google Colab.
2. Use a runtime with the required packages; the opening cells install several dependencies and record package versions.
3. Restart the session and run the cells in order from top to bottom.
4. Review the generated metrics, figures and submission checks. Model searches and cross-validation can take time.

The notebook contains Google Colab-specific download calls. Its recorded environment includes Python 3.13.15, scikit-learn 1.6.1 and LightGBM 4.6.0. Different environments may produce different predictions or fail strict fingerprint assertions.

## Publication note

The notebook is preserved as downloaded from the supplied Colab link, including saved outputs. It has not been rerun or modified for this publication. Saved results do not establish a fresh end-to-end execution.
