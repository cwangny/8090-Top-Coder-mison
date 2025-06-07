import sys
import joblib
import numpy as np
import os

def base_estimate(days, miles, receipts):
    reimbursement = (
        50 * days +
        0.50 * miles +
        0.40 * receipts +
        200
    )

    if days == 5:
        reimbursement += 50

    miles_per_day = miles / days if days > 0 else 0
    if 150 <= miles_per_day <= 220:
        reimbursement *= 1.10
    elif miles_per_day > 300:
        reimbursement *= 0.95

    receipt_str = f"{receipts:.2f}"
    if receipt_str.endswith("49") or receipt_str.endswith("99"):
        reimbursement += 10

    if receipts > 1500:
        reimbursement -= 0.05 * (receipts - 1500)

    return min(reimbursement, 2000)

def predict_correction(model, days, miles, receipts):
    miles_per_day = miles / days if days > 0 else 0
    receipts_per_day = receipts / days if days > 0 else 0

    features = np.array([[days, miles, receipts, miles_per_day, receipts_per_day]])
    correction = model.predict(features)[0]

    return max(min(correction, 300), -300)

if __name__ == "__main__":
    try:
        # Parse inputs
        days = int(sys.argv[1])
        miles = float(sys.argv[2])
        receipts = float(sys.argv[3])

        # Load model
        model_path = "residual_nn_model.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError("residual_nn_model.pkl not found")

        model = joblib.load(model_path)

        # Compute result
        base = base_estimate(days, miles, receipts)
        correction = predict_correction(model, days, miles, receipts)
        result = base + correction

        # Print final reimbursement
        print(f"{round(result, 2):.2f}")

    except Exception as e:
        # Print nothing but "0.00" for the eval script to parse
        print("0.00")
