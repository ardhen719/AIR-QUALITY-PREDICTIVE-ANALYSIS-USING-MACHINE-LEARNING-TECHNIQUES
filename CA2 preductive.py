# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 14:40:09 2025

@author: ardhe
"""

# -------------------------------------------------------------
# Air Quality Predictive Analysis Project
# Student-level, Spyder compatible
# -------------------------------------------------------------

# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------------------
# Load dataset
data = pd.read_csv(r"C:\Users\ardhe\OneDrive\Air_Quality.csv")

# Basic dataset analysis
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

# -------------------------------------------------------------
# Data selection and cleaning
data = data[['Name', 'Start_Date', 'Data Value']]
data = data.dropna()

# Filter a single pollutant for prediction
data = data[data['Name'] == 'Nitrogen dioxide (NO2)']

# Convert date to year
data['Start_Date'] = pd.to_datetime(data['Start_Date'])
data['Year'] = data['Start_Date'].dt.year

# -------------------------------------------------------------
# -------------------- DATA VISUALIZATION ---------------------
# -------------------------------------------------------------

# 1. Histogram
plt.figure(figsize=(6,4))
sb.histplot(data['Data Value'], kde=True, bins=15)
plt.title("Distribution of NO2 Levels")
plt.xlabel("NO2 Level")
plt.ylabel("Frequency")
plt.show()

# 2. Density Plot
plt.figure(figsize=(6,4))
sb.kdeplot(data['Data Value'], fill=True)
plt.title("Density Plot of NO2 Levels")
plt.xlabel("NO2 Level")
plt.show()

# 3. Scatter Plot
plt.figure(figsize=(6,4))
sb.scatterplot(x=data['Year'], y=data['Data Value'])
plt.title("NO2 Levels Over Years")
plt.xlabel("Year")
plt.ylabel("NO2 Level")
plt.show()

# 4. Line Chart – Yearly Average
yearly_avg = data.groupby('Year')['Data Value'].mean()
plt.figure(figsize=(7,4))
plt.plot(yearly_avg.index, yearly_avg.values, marker='o')
plt.title("Average NO2 Level per Year")
plt.xlabel("Year")
plt.ylabel("Average NO2 Level")
plt.grid(True)
plt.show()

# 5. Box Plot
plt.figure(figsize=(10,5))
sb.boxplot(x=data['Year'], y=data['Data Value'])
plt.xticks(rotation=90)
plt.title("Year-wise NO2 Distribution")
plt.xlabel("Year")
plt.ylabel("NO2 Level")
plt.show()

# 6. Correlation Heatmap
plt.figure(figsize=(6,4))
sb.heatmap(data[['Year','Data Value']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------------------------------------
# -------------------- MODEL PREPARATION ----------------------
# -------------------------------------------------------------

# Independent and dependent variables
X = data[['Year']]
y = data['Data Value']

# Normalization
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------------------------------------
# -------------------- MODEL TRAINING -------------------------
# -------------------------------------------------------------

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# 2. Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# 3. Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# -------------------------------------------------------------
# -------------------- MODEL EVALUATION -----------------------
# -------------------------------------------------------------

print("\nMODEL PERFORMANCE\n")

print("Linear Regression")
print("MSE:", mean_squared_error(y_test, lr_pred))
print("R2 Score:", r2_score(y_test, lr_pred))

print("\nDecision Tree")
print("MSE:", mean_squared_error(y_test, dt_pred))
print("R2 Score:", r2_score(y_test, dt_pred))

print("\nRandom Forest")
print("MSE:", mean_squared_error(y_test, rf_pred))
print("R2 Score:", r2_score(y_test, rf_pred))

# -------------------------------------------------------------
# -------------------- RESULT VISUALIZATION -------------------
# -------------------------------------------------------------

# 7. Actual vs Predicted
plt.figure(figsize=(6,4))
plt.scatter(X_test, y_test, label="Actual Values")
plt.scatter(X_test, rf_pred, label="Predicted Values")
plt.xlabel("Year (Scaled)")
plt.ylabel("NO2 Level")
plt.title("Actual vs Predicted NO2 Levels")
plt.legend()
plt.show()

# 8. Residual Plot
residuals = y_test - rf_pred
plt.figure(figsize=(6,4))
sb.scatterplot(x=rf_pred, y=residuals)
plt.axhline(0, linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# 9. Error Distribution
plt.figure(figsize=(6,4))
sb.histplot(residuals, kde=True, bins=15)
plt.title("Distribution of Prediction Errors")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()

# 10. Prediction Trend Line
sorted_idx = np.argsort(X_test.flatten())
plt.figure(figsize=(6,4))
plt.plot(X_test.flatten()[sorted_idx], y_test.values[sorted_idx], label="Actual")
plt.plot(X_test.flatten()[sorted_idx], rf_pred[sorted_idx], label="Predicted")
plt.xlabel("Year (Scaled)")
plt.ylabel("NO2 Level")
plt.title("Prediction Trend Line")
plt.legend()
plt.show()

# 11. Model Comparison Bar Chart
models = ['Linear Regression', 'Decision Tree', 'Random Forest']
r2_scores = [
    r2_score(y_test, lr_pred),
    r2_score(y_test, dt_pred),
    r2_score(y_test, rf_pred)
]

plt.figure(figsize=(6,4))
sb.barplot(x=models, y=r2_scores)
plt.ylabel("R2 Score")
plt.title("Model Performance Comparison")
plt.show()
