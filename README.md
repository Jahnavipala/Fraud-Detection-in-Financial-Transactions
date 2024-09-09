# Fraud-Detection-in-Financial-Transactions
## Project Overview
- This project aims to develop and evaluate models for detecting fraudulent financial transactions using a dataset of over 6 million entries. The dataset includes various transaction types, customer details, and transaction outcomes, with the target variable indicating whether a transaction is fraudulent or not.

## Table of Contents
- Project Overview
- Dataset Description
- Installation
- Data Understanding
- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Modeling and Evaluation
- Key Findings
- Conclusion
- Future Work
- References
  
## Dataset Description
- The dataset consists of 6,362,620 rows and 10 columns representing financial transactions. The key features include:

- **step:** Maps a unit of time in the real world.
- **type:** Type of transaction (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER).
- **amount:** Amount of the transaction in local currency.
- **nameOrig:** Customer who started the transaction.
- **oldbalanceOrg:** Initial balance before the transaction.
- **newbalanceOrig:** New balance after the transaction.
- **nameDest:** Customer who is the recipient of the transaction.
- **oldbalanceDest:** Initial balance recipient before the transaction.
- **newbalanceDest:** New balance recipient after the transaction.
- **isFraud:** Target variable indicating if the transaction is fraudulent (1) or not (0).
- **isFlaggedFraud:** Flags illegal attempts based on transfer amount.

## Installation
## Requirements
- Python 3.6+
- **Libraries:** pandas, numpy, seaborn, matplotlib, sklearn, joblib, xgboost, statsmodels

## Setup
- 1. Install the required packages
- 2. **Load the dataset:** Ensure the dataset Fraud.csv is placed in the appropriate directory. Update the file path in the script if needed.

## Data Understanding
- The dataset contains 6,362,620 rows and 10 columns.
- There are no missing values, but duplicates were checked.
- The dataset includes both numerical and categorical variables.
  
## Data Cleaning and Preprocessing
- **Outlier Detection and Removal:** Outliers in numeric columns were detected using the IQR method, which significantly reduced the dataset size to 4,393,187 rows.
- **Handling Multicollinearity:** Variance Inflation Factor (VIF) was calculated, revealing high multicollinearity among certain variables.
- **Feature Engineering:** New features were created to capture balance changes, and categorical variables were encoded using one-hot encoding.

## Exploratory Data Analysis (EDA)
- **Transaction Type Distribution:** Most transactions are of type PAYMENT, followed by CASH-OUT and TRANSFER.
- **Fraud Distribution:** The dataset is highly imbalanced, with only a small fraction of fraudulent transactions.
- **Correlation Analysis:** A heatmap was generated to visualize correlations between numerical features.

## Modeling and Evaluation
- Three models were trained and evaluated:

- 1. **Logistic Regression**
    - **Precision:** 0.89
    - **Recall:** 0.32
    - **ROC-AUC:** 0.991
    - **F1-Score:** Low, indicating a trade-off between precision and recall.

- 2. **Random Forest**
    - **Precision:** 0.96
    - **Recall:** 0.63
    - **ROC-AUC:** 0.982
    - **F1-Score:** 0.76, showing better balance between precision and recall.

- 3. **XGBoost**
    - **Precision:** 0.93
    - **Recall:** 0.74
    - **ROC-AUC:** 0.999
    - **F1-Score:** 0.83, the highest among the three models.

## Model Comparison
- XGBoost outperforms Logistic Regression and Random Forest in recall and F1-score, making it the best model for detecting fraud cases in this dataset.

## Key Findings
- **Top Features:**
  - **Random Forest:** Amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest.
  - **Logistic Regression:** Amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest.
  - **XGBoost:** Amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest.
- **Fraud Detection:** XGBoost showed the best performance in identifying fraudulent transactions, with the highest recall and F1-score.

## Conclusion
- This project successfully developed and evaluated models to detect fraudulent financial transactions, with XGBoost being the most effective model. The analysis of feature importance provided insights into key factors driving fraud detection.

## Future Work
- **Model Improvement:** Further tuning of models and exploration of ensemble techniques.
- **Real-time Fraud Detection:** Implementing real-time fraud detection using the trained model.
- **Deployment:** Extending the deployment to cloud platforms for scalable use.
