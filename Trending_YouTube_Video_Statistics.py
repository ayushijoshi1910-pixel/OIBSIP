# ==========================================================
# PROFESSIONAL MERGE + CLEAN PIPELINE
# YouTube Multi-Country Trending Dataset
# ==========================================================

import pandas as pd
import numpy as np

# ==========================================================
# 1️⃣ Load All Country Files
# ==========================================================

file_paths = {
    "CA": r"D:\Oasis internship\project 3 of level 1 dataset 2\CAvideos.csv",
    "DE": r"D:\Oasis internship\project 3 of level 1 dataset 2\DEvideos.csv",
    "FR": r"D:\Oasis internship\project 3 of level 1 dataset 2\FRvideos.csv",
    "GB": r"D:\Oasis internship\project 3 of level 1 dataset 2\GBvideos.csv",
    "IN": r"D:\Oasis internship\project 3 of level 1 dataset 2\INvideos.csv",
    "JP": r"D:\Oasis internship\project 3 of level 1 dataset 2\JPvideos.csv",
    "KR": r"D:\Oasis internship\project 3 of level 1 dataset 2\KRvideos.csv",
    "MX": r"D:\Oasis internship\project 3 of level 1 dataset 2\MXvideos.csv",
    "RU": r"D:\Oasis internship\project 3 of level 1 dataset 2\RUvideos.csv",
    "US": r"D:\Oasis internship\project 3 of level 1 dataset 2\USvideos.csv"
}



dataframes = []

for country, path in file_paths.items():
    try:
        df = pd.read_csv(path)
        df["country"] = country
        dataframes.append(df)
        print(f"{country} loaded successfully.")
    except Exception as e:
        print(f"Error loading {country}: {e}")

# ==========================================================
# 2️⃣ Merge All Files
# ==========================================================

merged_df = pd.concat(dataframes, ignore_index=True)
print("\nMerged Dataset Shape:", merged_df.shape)

# ==========================================================
# 3️⃣ Basic Cleaning
# ==========================================================

# Remove completely empty rows
merged_df.dropna(how="all", inplace=True)

# Remove duplicates properly (important improvement)
merged_df.drop_duplicates(
    subset=["video_id", "trending_date", "country"],
    inplace=True
)

print("After duplicate removal:", merged_df.shape)

# ==========================================================
# 4️⃣ Handle Missing Values Carefully
# ==========================================================

# Drop columns with >50% missing
merged_df = merged_df.loc[:, merged_df.isnull().mean() < 0.5]

# Numeric columns
num_cols = merged_df.select_dtypes(include=["int64", "float64"]).columns

# Replace negative values (invalid for YouTube stats)
for col in ["views", "likes", "dislikes", "comment_count"]:
    if col in merged_df.columns:
        merged_df = merged_df[merged_df[col] >= 0]

# Fill small missing numeric values with median
for col in num_cols:
    merged_df[col] = merged_df[col].fillna(merged_df[col].median())

# Categorical columns
cat_cols = merged_df.select_dtypes(include=["object"]).columns

for col in cat_cols:
    merged_df[col] = merged_df[col].fillna("unknown")


print("Missing values handled.")

# ==========================================================
# 5️⃣ Standardize Text Columns
# ==========================================================

text_columns = ["title", "channel_title", "tags"]

for col in text_columns:
    if col in merged_df.columns:
        merged_df[col] = (
            merged_df[col]
            .astype(str)
            .str.lower()
            .str.strip()
        )

print("Text standardization done.")

# ==========================================================
# 6️⃣ Convert Date Columns Properly
# ==========================================================

if "trending_date" in merged_df.columns:
    merged_df["trending_date"] = pd.to_datetime(
        merged_df["trending_date"],
        format="%y.%d.%m",
        errors="coerce"
    )

if "publish_time" in merged_df.columns:
    merged_df["publish_time"] = pd.to_datetime(
        merged_df["publish_time"],
        format="%Y-%m-%dT%H:%M:%S.%fZ",
        errors="coerce"
    )

print("Date conversion completed.")

# ==========================================================
# 7️⃣ Final Validation
# ==========================================================

print("\nFinal Dataset Info:")
print(merged_df.info())

print("\nFinal Shape:", merged_df.shape)

# ==========================================================
# 8️⃣ Save Final Cleaned CSV
# ==========================================================

merged_df.to_csv("Merged_Cleaned_YouTube_Data.csv", index=False)

print("\n✅ Final cleaned dataset saved as:")
print("Merged_Cleaned_YouTube_Data.csv")
