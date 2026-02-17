# ==========================================
# CUSTOMER SEGMENTATION PROJECT
# ==========================================

# -----------------------------
# 1. Import Libraries
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")


# -----------------------------
# 2. Load Dataset
# -----------------------------

df = pd.read_csv("D:\Oasis internship\customer_segmentation.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# -----------------------------
# 3. Data Cleaning
# -----------------------------

# Convert dates (DD-MM-YYYY format)
df["Order_Date"] = pd.to_datetime(df["Order_Date"], dayfirst=True, errors="coerce")
df["Signup_Date"] = pd.to_datetime(df["Signup_Date"], dayfirst=True, errors="coerce")

# Remove duplicates
df.drop_duplicates(inplace=True)

# Drop rows where date conversion failed
df.dropna(subset=["Order_Date"], inplace=True)

print("\nCleaned Data Shape:", df.shape)


# -----------------------------
# 4. Descriptive Statistics
# -----------------------------

print("\nBasic Statistics:")
print(df.describe())

print("\nAverage Order Value:", df["Order_Value_INR"].mean())
print("Total Revenue:", df["Order_Value_INR"].sum())
print("Total Unique Customers:", df["Customer_ID"].nunique())

# Frequency per customer
frequency_avg = df.groupby("Customer_ID")["Transaction_ID"].count().mean()
print("Average Purchase Frequency per Customer:", frequency_avg)


# -----------------------------
# 5. RFM Feature Engineering
# -----------------------------

reference_date = df["Order_Date"].max()

rfm = df.groupby("Customer_ID").agg({
    "Order_Date": lambda x: (reference_date - x.max()).days,
    "Transaction_ID": "count",
    "Order_Value_INR": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

print("\nRFM Table:")
print(rfm.head())


# -----------------------------
# 6. Visualization - RFM Distribution
# -----------------------------

plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
sns.histplot(rfm["Recency"], kde=True)
plt.title("Recency Distribution")

plt.subplot(1,3,2)
sns.histplot(rfm["Frequency"], kde=True)
plt.title("Frequency Distribution")

plt.subplot(1,3,3)
sns.histplot(rfm["Monetary"], kde=True)
plt.title("Monetary Distribution")

plt.tight_layout()
plt.show()


# -----------------------------
# 7. Scaling
# -----------------------------

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

rfm_scaled = pd.DataFrame(rfm_scaled, columns=rfm.columns)


# -----------------------------
# 8. Elbow Method
# -----------------------------

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


# -----------------------------
# 9. Apply K-Means
# -----------------------------

kmeans = KMeans(n_clusters=4, random_state=42)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

print("\nCluster Distribution:")
print(rfm["Cluster"].value_counts())


# -----------------------------
# 10. Model Evaluation
# -----------------------------

sil_score = silhouette_score(rfm_scaled, rfm["Cluster"])
print("\nSilhouette Score:", sil_score)


# -----------------------------
# 11. Cluster Analysis
# -----------------------------

cluster_summary = rfm.groupby("Cluster").mean()
print("\nCluster Summary:")
print(cluster_summary)


# -----------------------------
# 12. Visualization of Clusters
# -----------------------------

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=rfm["Frequency"],
    y=rfm["Monetary"],
    hue=rfm["Cluster"],
    palette="Set2"
)
plt.title("Customer Segments")
plt.show()


# Bar chart - Average Spending per Cluster
rfm.groupby("Cluster")["Monetary"].mean().plot(kind="bar")
plt.title("Average Spending per Cluster")
plt.ylabel("Average Monetary Value")
plt.show()


# ==========================================
# 13. Business Insights (Print Interpretation)
# ==========================================

print("\n--- BUSINESS INSIGHTS ---")

for cluster in cluster_summary.index:
    print(f"\nCluster {cluster}:")
    print(cluster_summary.loc[cluster])

print("\nInterpret clusters based on:")
print("Low Recency = Active Customers")
print("High Frequency = Loyal Customers")
print("High Monetary = High Value Customers")
