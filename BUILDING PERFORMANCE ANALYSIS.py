import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Sample synthetic data based on your plots (replace with real data if available)
dates = pd.date_range(start="2025-04-05", periods=100, freq='D')
actual_energy = np.random.normal(180, 30, size=len(dates))
predicted_energy = actual_energy + np.random.normal(0, 10, size=len(dates))

# DataFrame
df = pd.DataFrame({
    "Date": dates,
    "Actual": actual_energy,
    "Predicted": predicted_energy
})

# Plot 1: Actual vs Predicted Energy Usage
plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["Actual"], label='Actual')
plt.plot(df["Date"], df["Predicted"], label='Predicted', linestyle='--')
plt.title("Energy Usage vs Predicted (First 100 Points)")
plt.xlabel("Timestamp")
plt.ylabel("Energy (kWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Energy Usage Distribution
plt.figure(figsize=(7, 4))
plt.hist(df["Actual"], bins=20, alpha=0.7, label='Actual')
plt.title("Energy Usage Distribution")
plt.xlabel("Energy (kWh)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# Model Evaluation
mse = mean_squared_error(df["Actual"], df["Predicted"])
r2 = r2_score(df["Actual"], df["Predicted"])

print("Building Performance Analysis")
print("Model Evaluation Metrics:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Regression Line (Optional)
X = np.array(df["Actual"]).reshape(-1, 1)
y = df["Predicted"]
reg = LinearRegression().fit(X, y)
y_pred_line = reg.predict(X)

plt.figure(figsize=(6, 4))
plt.scatter(df["Actual"], df["Predicted"], alpha=0.5, label='Data Points')
plt.plot(df["Actual"], y_pred_line, color='red', label='Regression Line')
plt.title("Regression: Actual vs Predicted")
plt.xlabel("Actual Energy (kWh)")
plt.ylabel("Predicted Energy (kWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
