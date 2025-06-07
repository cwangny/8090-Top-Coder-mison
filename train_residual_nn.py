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
df["miles_times_receipts"] = df["miles_traveled"] * df["total_receipts_amount"]
df["days_squared"] = df["trip_duration_days"] ** 2
df["efficiency_score"] = df["miles_per_day"] / (df["receipts_per_day"] + 1)
df["is_5_day_bonus"] = (df["trip_duration_days"] == 5).astype(int)
df["receipt_magic"] = df["total_receipts_amount"].apply(
    lambda x: str(f"{x:.2f}").endswith("49") or str(f"{x:.2f}").endswith("99")
).astype(int)
df["overspend_penalty"] = (df["total_receipts_amount"] > 1500).astype(int)

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

    miles_per_day = row["miles_per_day"]
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
    "receipts_per_day",
    "miles_times_receipts",
    "days_squared",
    "efficiency_score",
    "is_5_day_bonus",
    "receipt_magic",
    "overspend_penalty"
]
X = df[features]
y = df["residual"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train neural net with early stopping
model = MLPRegressor(hidden_layer_sizes=(64, 32, 16),
                     max_iter=2000,
                     early_stopping=True,
                     validation_fraction=0.1,
                     n_iter_no_change=20,
                     random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"Validation MAE: {mae:.2f}")

# Save model
joblib.dump(model, "residual_nn_model.pkl")
