# ==========================================
# Airbnb NYC 2019 - Data Cleaning Project
# ==========================================

# 1️Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2️Load Dataset
original_df = pd.read_csv("D:\Oasis internship\Project 3 of Level1\AB_NYC_2019.csv")

# Create working copy
df = original_df.copy()

# Store original row count
before_rows = df.shape[0]

print("Original Dataset Shape:", df.shape)


# ==========================================
# 3️DATA CLEANING PROCESS
# ==========================================

# 🔹 Convert 'last_review' to datetime first
df['last_review'] = pd.to_datetime(df['last_review'])

# 🔹 Handle Missing Values (Professional way)
df.fillna({
    'name': "Unknown",
    'host_name': "Unknown",
    'reviews_per_month': 0
}, inplace=True)

# Handle date column separately
df['last_review'] = df['last_review'].fillna(pd.Timestamp("1900-01-01"))

# 🔹 Remove Duplicates
df = df.drop_duplicates()

# 🔹 Remove Invalid Values
df = df[df['price'] > 0]
df = df[df['minimum_nights'] > 0]

# 🔹 Standardize Column Names
df.columns = df.columns.str.lower().str.replace(" ", "_")

# 🔹 Remove Outliers (IQR Method on Price)
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

# 🔹 Convert Data Types
df['neighbourhood_group'] = df['neighbourhood_group'].astype('category')
df['room_type'] = df['room_type'].astype('category')


# ==========================================
# 4️BEFORE vs AFTER COMPARISON
# ==========================================

after_rows = df.shape[0]
rows_removed = before_rows - after_rows
reduction_percent = (rows_removed / before_rows) * 100

print("\nCleaned Dataset Shape:", df.shape)
print("Rows Removed:", rows_removed)
print("Percentage Reduction: {:.2f}%".format(reduction_percent))


# ==========================================
# 5️VISUALIZATION - BEFORE vs AFTER
# ==========================================

plt.figure(figsize=(6,5))
plt.bar(['Before Cleaning', 'After Cleaning'], [before_rows, after_rows])
plt.title("Dataset Size: Before vs After Cleaning")
plt.ylabel("Number of Rows")
plt.xlabel("Dataset Stage")
plt.show()


# ==========================================
# 6️ Save Cleaned Dataset (Optional)
# ==========================================

df.to_csv("D:\Oasis internship\Project 3 of Level1\AB_NYC_2019_Cleaned.csv", index=False)

print("\nData cleaning completed successfully ✅")
