# ==========================================
# Wine Quality Prediction using ML Models
# Random Forest, SGD, SVC
# ==========================================

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2️⃣ Load Dataset (USE sep=';')
df = pd.read_csv(r"D:\Oasis internship\Project 2 for level 2\WineQuality.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# 3️⃣ Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=False)
plt.title("Correlation Heatmap")
plt.show()

# 4️⃣ Feature & Target Selection
X = df.drop("quality", axis=1)
y = df["quality"]

# Convert into Binary Classification (Good=1, Bad=0)
y = y.apply(lambda x: 1 if x >= 6 else 0)

# 5️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6️⃣ Feature Scaling (Important for SGD & SVC)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# 7️⃣ Random Forest Classifier
# ==========================================
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\n===== Random Forest =====")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# ==========================================
# 8️⃣ SGD Classifier
# ==========================================
sgd = SGDClassifier(random_state=42)
sgd.fit(X_train, y_train)
sgd_pred = sgd.predict(X_test)

print("\n===== SGD Classifier =====")
print("Accuracy:", accuracy_score(y_test, sgd_pred))
print(classification_report(y_test, sgd_pred))

# ==========================================
# 9️⃣ Support Vector Classifier (SVC)
# ==========================================
svc = SVC()
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)

print("\n===== Support Vector Classifier =====")
print("Accuracy:", accuracy_score(y_test, svc_pred))
print(classification_report(y_test, svc_pred))

# ==========================================
# 🔟 Confusion Matrix (Random Forest)
# ==========================================
plt.figure()
sns.heatmap(confusion_matrix(y_test, rf_pred), 
            annot=True, fmt='d')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
