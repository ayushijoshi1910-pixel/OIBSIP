# ==========================================
# 1️⃣ Import Required Libraries
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 2️⃣ Load Datasets
# ==========================================
apps = pd.read_csv(r"D:\Oasis internship\Project 4 of level 2\apps.csv")
reviews = pd.read_csv(r"D:\Oasis internship\Project 4 of level 2\user_reviews.csv")

# ==========================================
# 3️⃣ Merge Datasets
# ==========================================
apps.drop_duplicates(inplace=True)
reviews.drop_duplicates(inplace=True)

apps['App'] = apps['App'].str.strip()
reviews['App'] = reviews['App'].str.strip()

merged_df = pd.merge(apps, reviews, on='App', how='inner')

# ==========================================
# 4️⃣ Data Cleaning & Type Correction
# ==========================================

# Clean Installs
merged_df['Installs'] = merged_df['Installs'].str.replace('+','')
merged_df['Installs'] = merged_df['Installs'].str.replace(',','')
merged_df['Installs'] = pd.to_numeric(merged_df['Installs'], errors='coerce')

# Clean Price
merged_df['Price'] = merged_df['Price'].str.replace('$','')
merged_df['Price'] = pd.to_numeric(merged_df['Price'], errors='coerce')

# Clean Reviews
merged_df['Reviews'] = pd.to_numeric(merged_df['Reviews'], errors='coerce')

# Clean Rating
merged_df['Rating'] = pd.to_numeric(merged_df['Rating'], errors='coerce')

# Clean Size
def convert_size(size):
    if 'M' in str(size):
        return float(size.replace('M',''))
    elif 'k' in str(size):
        return float(size.replace('k',''))/1024
    else:
        return np.nan

merged_df['Size'] = merged_df['Size'].apply(convert_size)

# Remove Missing Values
merged_df.dropna(subset=['Rating','Installs'], inplace=True)

# ==========================================
# 5️⃣ Category Exploration
# ==========================================

category_count = merged_df['Category'].value_counts()

plt.figure(figsize=(12,6))
sns.barplot(x=category_count.index[:10], y=category_count.values[:10])
plt.xticks(rotation=90)
plt.title("Top 10 App Categories")
plt.show()

# ==========================================
# 6️⃣ Metrics Analysis
# ==========================================

# Rating Distribution
plt.figure(figsize=(8,5))
sns.histplot(merged_df['Rating'], bins=20)
plt.title("Rating Distribution")
plt.show()

# Rating vs Installs
plt.figure(figsize=(8,6))
sns.scatterplot(x='Rating', y='Installs', data=merged_df)
plt.title("Rating vs Installs")
plt.show()

# Price Trend
plt.figure(figsize=(8,6))
sns.boxplot(x='Type', y='Price', data=merged_df)
plt.title("Price Distribution (Free vs Paid)")
plt.show()

# ==========================================
# 7️⃣ Sentiment Analysis
# ==========================================

sentiment_count = merged_df['Sentiment'].value_counts()

plt.figure(figsize=(6,6))
plt.pie(sentiment_count, labels=sentiment_count.index, autopct='%1.1f%%')
plt.title("Sentiment Distribution")
plt.show()

# Average Rating by Sentiment
avg_rating_sentiment = merged_df.groupby('Sentiment')['Rating'].mean()
print("Average Rating by Sentiment:")
print(avg_rating_sentiment)

# ==========================================
# 8️⃣ Correlation Analysis
# ==========================================

plt.figure(figsize=(8,6))
sns.heatmap(merged_df[['Rating','Reviews','Installs','Price']].corr(),
            annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# ==========================================
# 9️⃣ Save Cleaned Dataset
# ==========================================
merged_df.to_csv("cleaned_google_play_store.csv", index=False)

print("Cleaned dataset saved successfully!")
