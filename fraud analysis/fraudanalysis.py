import pandas as pd 
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
df= pd.read_csv("C:/Users/DELL/OneDrive\Desktop/fraud analysis/Fraud.csv")
# preprocessing of data

df.head()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df.info()
# no missing values
df.isnull().sum()
df.describe()
df['step'].value_counts()
print(df['isFraud'].value_counts())
df.columns
df['type'].value_counts()
#CASH_OUT and TRANSFER transactions are particularly relevant for fraud detection, as fraudulent activities often involve emptying accounts
# or transferring money to other accounts.
df['isFlaggedFraud'].value_counts()
#visualisation
plt.figure(figsize=(10, 6))
sns.histplot(df['step'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of Transactions Over Time (Step) with KDE')
plt.xlabel('Step (Hour)')
plt.ylabel('Number of Transactions')
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='type', palette='pastel')
plt.title('Transaction Types Distribution')
plt.xlabel('Transaction Type')
plt.ylabel('Number of Transactions')
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(df['amount'], bins=50, color='lightgreen', log_scale=(False, True))
plt.title('Distribution of Transaction Amounts (Log Scale)')
plt.xlabel('Amount')
plt.ylabel('Frequency (Log Scale)')
plt.show()

# Seaborn bar plot for 'isFraud'
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='isFraud', palette='muted')
plt.title('Fraudulent vs Non-Fraudulent Transactions')
plt.xlabel('Is Fraud')
plt.ylabel('Number of Transactions')
plt.xticks(ticks=[0, 1], labels=['Non-Fraud', 'Fraud'])
plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='isFlaggedFraud', palette='coolwarm')
plt.title('Flagged Fraudulent Transactions')
plt.xlabel('Is Flagged Fraud')
plt.ylabel('Number of Transactions')
plt.xticks(ticks=[0, 1], labels=['Not Flagged', 'Flagged'])
plt.show()


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['oldbalanceOrg'], bins=50, color='lightcoral', log_scale=(False, True))
plt.title('Distribution of Old Balance (Origin)')
plt.xlabel('Old Balance')
plt.ylabel('Frequency (Log Scale)')

plt.subplot(1, 2, 2)
sns.histplot(df['newbalanceOrig'], bins=50, color='lightblue', log_scale=(False, True))
plt.title('Distribution of New Balance (Origin)')
plt.xlabel('New Balance')
plt.ylabel('Frequency (Log Scale)')


# Seaborn histograms for 'oldbalanceDest' and 'newbalanceDest'
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['oldbalanceDest'], bins=50, color='lightgreen', log_scale=(False, True))
plt.title('Distribution of Old Balance (Destination)')
plt.xlabel('Old Balance')
plt.ylabel('Frequency (Log Scale)')

plt.subplot(1, 2, 2)
sns.histplot(df['newbalanceDest'], bins=50, color='lightpink', log_scale=(False, True))
plt.title('Distribution of New Balance (Destination)')
plt.xlabel('New Balance')
plt.ylabel('Frequency (Log Scale)')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# skewness 
#Highly positively skewed, indicating that most
# transactions are of smaller amounts, with a small number of very high-value transactions.
df['amount'].skew()

df['oldbalanceOrg'].skew()
df['newbalanceOrig'].skew()
df['oldbalanceDest'].skew()
df['newbalanceDest'].skew()



# feature engineer 
# Balance changes for origin and destination
df['balanceOrigDiff'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['balanceDestDiff'] = df['oldbalanceDest'] - df['newbalanceDest']

# Flags for zero balances
df['origOldZero'] = df['oldbalanceOrg'] == 0
df['origNewZero'] = df['newbalanceOrig'] == 0
df['destOldZero'] = df['oldbalanceDest'] == 0
df['destNewZero'] = df['newbalanceDest'] == 0

df.head()
# Visualizing balance differences


# Count the number of zero balance flags
zero_balance_counts = {
    'Orig Old Zero': df['origOldZero'].sum(),
    'Orig New Zero': df['origNewZero'].sum(),
    'Dest Old Zero': df['destOldZero'].sum(),
    'Dest New Zero': df['destNewZero'].sum(),
}

print("Counts of Zero Balances:")
for key, value in zero_balance_counts.items():
    print(f"{key}: {value}")

correlation_matrix = df.corr()

# Extract the correlations with the target variable 'isFraud'
correlation_with_target = correlation_matrix['isFraud']

# Display the correlation with the target variable
print(correlation_with_target)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='balanceOrigDiff', y='isFraud', data=df, alpha=0.5)
plt.title('Scatter Plot of balanceOrigDiff vs. isFraud')
plt.xlabel('balanceOrigDiff')
plt.ylabel('isFraud')
plt.grid()
plt.show()



plt.figure(figsize=(10, 6))
sns.boxplot(x='isFraud', y='balanceOrigDiff', data=df)
plt.title('Box Plot of balanceOrigDiff by isFraud')
plt.xlabel('isFraud')
plt.ylabel('balanceOrigDiff')
plt.grid()
plt.show()



df.drop(columns=['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'], inplace=True)
df.columns
df.head()
from sklearn.preprocessing import LabelEncoder


label_encoder = LabelEncoder()

cols_to_encode = ['origOldZero', 'origNewZero', 'destOldZero', 'destNewZero']

for col in cols_to_encode:
    df[col] = label_encoder.fit_transform(df[col])

print(df[cols_to_encode].head())

df_encoded = pd.get_dummies(df, columns=['type'], drop_first=True)
print(df_encoded.head())

df.shape

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

X = df_encoded.drop(columns=['isFraud', 'nameOrig', 'nameDest'])  # Drop target and identifiers
y = df_encoded['isFraud']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

importances = rf_model.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Plot feature importance

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance in Random Forest')
plt.show()
 
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Non-Fraudulent", "Fraudulent"], 
            yticklabels=["Non-Fraudulent", "Fraudulent"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Average Cross-Validation Score:", cv_scores.mean())















