import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load the public_cases.json
with open("public_cases.json", "r") as f:
    data = json.load(f)

# Extract input/output
inputs = [case["input"] for case in data]
outputs = [case["expected_output"] for case in data]

# Build DataFrame
df = pd.DataFrame(inputs)
df["expected"] = outputs

# Derived features
df["miles_per_day"] = df["miles_traveled"] / df["trip_duration_days"]
df["receipts_per_day"] = df["total_receipts_amount"] / df["trip_duration_days"]
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# Base model to calculate estimated reimbursement
def base_estimate(row):
    reimbursement = (
        50 * row["trip_duration_days"] +
        0.50 * row["miles_traveled"] +
        0.40 * row["total_receipts_amount"] +
        200
    )

    if row["trip_duration_days"] == 5:
        reimbursement += 50

    miles_per_day = row["miles_traveled"] / row["trip_duration_days"]
    if 150 <= miles_per_day <= 220:
        reimbursement *= 1.10
    elif miles_per_day > 300:
        reimbursement *= 0.95

    receipt_str = f"{row['total_receipts_amount']:.2f}"
    if receipt_str.endswith("49") or receipt_str.endswith("99"):
        reimbursement += 10

    if row["total_receipts_amount"] > 1500:
        reimbursement -= 0.05 * (row["total_receipts_amount"] - 1500)

    return min(reimbursement, 2000)

# Compute residuals
df["base"] = df.apply(base_estimate, axis=1)
df["residual"] = df["expected"] - df["base"]

# Features to use
features = [
    "trip_duration_days",
    "miles_traveled",
    "total_receipts_amount",
    "miles_per_day",
    "receipts_per_day"
]
X = df[features]
y = df["residual"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train neural net
model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"Validation MAE: {mae:.2f}")

# Save model
joblib.dump(model, "residual_nn_model.pkl")
