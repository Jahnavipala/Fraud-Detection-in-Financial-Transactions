# **Fraud Detection in Financial Transactions**

# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the dataset
df = pd.read_csv(r"C:\Users\jahna\OneDrive\Desktop\Accredian\Fraud.csv")


# **Data Understanding**

print("First few rows of the dataset:")
df.head()

print("\nLast few rows of the dataset:")
df.tail()

print("\nShape of the dataset:")
print(df.shape)

print("\nData types and non-null counts:")
df.info()

print("\nUnique values in each column:")
df.nunique()

# Exploring the Variables
print("\nTransaction Types Count:")
print(df.type.value_counts())
print("--------------------------------------------")
print("\nFraud Count:")
print(df.isFraud.value_counts())
print("--------------------------------------------")

# Calculate the Percentage Distribution of transaction_type
transaction_type_percentage = df['type'].value_counts(normalize = True) * 100
transaction_type_percentage

# Plotting the Percentage Distribution of transaction_type
plt.figure(figsize = (5, 5))
plt.pie(transaction_type_percentage, labels = transaction_type_percentage.index, autopct = '%1.1f%%', startangle = 140)
plt.title('Percentage Distribution of Transaction Type', fontsize = 15, fontweight = 'bold')
plt.axis('equal')
plt.show()

# Data Types in the Dataset
# Count of types of data type
df.dtypes.value_counts()

#Plotting Data type of the Dataset
df.dtypes.value_counts().plot.pie(explode = [0.1, 0.1, 0.1], autopct = '%1.1f%%', shadow = True)
plt.title('Data Type of the Dataset', fontsize = 15, fontweight = 'bold')


# **Data Cleaning**

# Check For Missing Values
missing_values = df.isnull().sum()
print('Missing Values in each column:')
print(missing_values)

# Checking for Duplicates
df.duplicated().sum()

# Numerical Columns
numerical_cols = [f for f in df.columns if df.dtypes[f] != 'object']
print("The numirical values in the dataset are given below:- \n",numerical_cols)

# Categorical Columns
cat_columns = [f for f in df.columns if df.dtypes[f] == 'object']
cat_columns.remove('nameOrig')
cat_columns.remove('nameDest')
print("The Categorical values in the dataset are given below:- \n",cat_columns)

# Handling Outliers: Using IQR to detect outliers for numeric columns
numeric_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

outliers = {col: detect_outliers(df, col) for col in numeric_columns}
print("Outliers detected in numeric columns.")
print(f"Original data size: {df.shape}")

# Removing outliers 
for col in numeric_columns:
     df = df[~df.index.isin(outliers[col].index)]
print(f"Data size after outlier removal: {df.shape}")

# Set the style for the plots
sns.set(style="whitegrid")

# Plotting box plots for each numeric column
plt.figure(figsize=(15, 8))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot of {col}')
    plt.ylabel(col)

plt.tight_layout()
plt.show()

# Set the style for the plots
sns.set(style="whitegrid")

# Plotting box plots for each numeric column after outlier removal
plt.figure(figsize=(15, 8))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot of {col} (After Outlier Removal)')
    plt.ylabel(col)

plt.tight_layout()
plt.show()

# The original data had 6,362,620 entries, and after outlier removal, 4,393,187 entries remained. This suggests that a significant number of outliers were removed, which could affect model performance.

# **Multicollinearity**

# Handling Multicollinearity: Calculating Variance Inflation Factor (VIF)
X = df[numeric_columns]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print("Variance Inflation Factor (VIF):\n", vif_data)


# **Exploratory Data Analysis (EDA)**

# Basic Statistics 
df.describe(include = 'all')


# **Fraud and Non-Fraud Transaction Across Different Transaction Types**

# Analyzing the distribution of transaction types
sns.countplot(x='type', data=df)
plt.title('Distribution of Transaction Types')
plt.show()

# Analyzing fraud cases by transaction type
fraud_by_type = df.groupby('type')['isFraud'].sum().reset_index()
sns.barplot(x='type', y='isFraud', data=fraud_by_type)
plt.title('Fraud Cases by Transaction Type')
plt.show()


# **Correlation Heatmap**

# Analyzing the correlation between numeric features
corr_matrix = df[numeric_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Checking the balance of the target variable
sns.countplot(x='isFraud', data=df)
plt.title('Fraud vs Non-Fraud Cases')
plt.show()


# **Feature Engineering: Creating new features based on existing data**

# Creating a feature to capture the difference between old and new balance
df['sender_balance_change'] = df['newbalanceOrig'] - df['oldbalanceOrg']
df['receiver_balance_change'] = df['newbalanceDest'] - df['oldbalanceDest']

# Encoding categorical variables: Transaction type
df = pd.get_dummies(df, columns=['type'], drop_first=True)

# Dropping irrelevant columns for model training
df.drop(columns=['nameOrig', 'nameDest'], inplace=True)

# Defining features (X) and target (y)
X = df.drop(columns=['isFraud', 'isFlaggedFraud'])
y = df['isFraud']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler and feature names
joblib.dump(scaler, 'scaler.pkl')
feature_names = scaler.feature_names_in_
joblib.dump(feature_names, 'feature_names.pkl')

# **Logistic Regression Model**

# Training the Logistic Regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Predictions
y_pred_lr = log_reg.predict(X_test)
y_proba_lr = log_reg.predict_proba(X_test)[:, 1]

# Performance Evaluation
print("Logistic Regression Performance:")
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba_lr))

# Plotting ROC Curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression (AUC = {:.2f})'.format(roc_auc_score(y_test, y_proba_lr)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Save Logistic Regression Model
joblib.dump(log_reg, 'Log_reg_model.pkl')

# **Analysis:** 
# > The logistic regression model shows a high precision (0.89) but relatively low recall (0.32), meaning it identifies a smaller portion of actual fraud cases but is very accurate when it does.
# 
# > The ROC-AUC score of 0.991 is very high, indicating that the model is effective at distinguishing between fraudulent and non-fraudulent transactions.
# 
# > The F1-score is low, reflecting the trade-off between precision and recall.
# 

# **Random Forest model**

# Training the Random Forest model
rf_model = RandomForestClassifier(n_estimators=30, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Performance Evaluation
print("Random Forest Performance:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba_rf))

# Plotting ROC Curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
plt.plot(fpr_rf, tpr_rf, label='Random Forest (AUC = {:.2f})'.format(roc_auc_score(y_test, y_proba_rf)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Save Random Forest Model
joblib.dump(rf_model, 'rf_model.pkl')

# **Analysis:**
# > The Random Forest model improves recall significantly (0.63) compared to Logistic Regression, meaning it identifies more actual fraud cases.
# 
# > Precision is also high (0.96), meaning few non-fraudulent cases are misclassified as fraud.
# 
# > The ROC-AUC score is slightly lower than Logistic Regression, but still very strong at 0.982.
# 
# >The F1-score of 0.76 indicates a better balance between precision and recall.

# **Extreme Gradient Boosting model**

# Training the XGBoost model
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Performance Evaluation
print("XGBoost Performance:")
print(classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba_xgb))

# Plotting ROC Curve
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost (AUC = {:.2f})'.format(roc_auc_score(y_test, y_proba_xgb)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Save XGBoost Model
joblib.dump(xgb_model, 'xgb_model.pkl')

# **Analysis:**
# > XGBoost outperforms the other models in terms of recall (0.74), meaning it correctly identifies more fraud cases.
# 
# > Precision is still high at 0.93, though slightly lower than Random Forest.
# 
# > The ROC-AUC score is near perfect at 0.999, indicating exceptional performance in distinguishing between classes.
# 
# > The F1-score of 0.83 is the highest among the three models, showing the best balance between precision and recall.

# **Interpretation and Key Factors**

# Analyze and list the important features (e.g., feature importance from Random Forest)
important_features_rf = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 5 Important Features in Random Forest Model:")
print(important_features_rf.head())

# Coefficients from Logistic Regression (absolute values for importance)
important_features_lr = pd.Series(np.abs(log_reg.coef_[0]), index=X.columns).sort_values(ascending=False)
print("\nTop 5 Important Features in Logistic Regression Model:")
print(important_features_lr.head())

# Feature Importance from XGBoost
important_features_xgb = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 5 Important Features in XGBoost Model:")
print(important_features_xgb.head())







