# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1 Load Dataset
df = pd.read_csv(r"D:\Oasis internship\Project 1 of level 2\train (1).csv")

# Display first 5 rows
print(df.head())

# 2 Data Exploration
print("\nDataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum().sort_values(ascending=False).head())

# 3 Select Important Numerical Features
features = ['OverallQual', 'GrLivArea', 'GarageArea', 
            'TotalBsmtSF', 'YearBuilt']

# Fill missing values with median
df[features] = df[features].fillna(df[features].median())

X = df[features]
y = df['SalePrice']

# 4 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5 Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# 6 Predictions
y_pred = model.predict(X_test)

# 7 Model Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# 8 Visualization – Predicted vs Actual
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# 9 Regression Coefficients
coeff_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})
print("\nFeature Importance:")
print(coeff_df)
