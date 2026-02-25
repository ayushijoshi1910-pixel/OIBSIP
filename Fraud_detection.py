# ==========================================
# Credit Card Fraud Detection Project
# ==========================================

# 1 Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")
# 2 Load Dataset
df = pd.read_csv("D:\Oasis internship\Project 3 of level 2\creditcard.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# 3 Check Class Distribution (Imbalanced Data)
print("\nClass Distribution:")
print(df['Class'].value_counts())

# 4 Feature & Target Selection
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale Amount column
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])

# 5 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 6 Logistic Regression
# ==========================================
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\n===== Logistic Regression =====")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# ==========================================
# 7 Decision Tree
# ==========================================
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("\n===== Decision Tree =====")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

# ==========================================
# 8 Confusion Matrix (Logistic Regression)
# ==========================================
plt.figure()
sns.heatmap(confusion_matrix(y_test, lr_pred), 
            annot=True, fmt='d')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
