import json
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

# Load test cases
with open("public_cases.json", "r") as f:
    cases = json.load(f)

# Load model
model = joblib.load("residual_nn_model.pkl")

# Base estimate logic
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

# Evaluation loop
successful = 0
exact = 0
close = 0
total_error = 0
max_error = 0
results = []

print("📊 Running evaluation against 1,000 test cases...\n")

for idx, case in enumerate(tqdm(cases, desc="Progress")):
    days = case["input"]["trip_duration_days"]
    miles = case["input"]["miles_traveled"]
    receipts = case["input"]["total_receipts_amount"]
    expected = case["expected_output"]

    try:
        base = base_estimate(days, miles, receipts)
        miles_per_day = miles / days if days > 0 else 0
        receipts_per_day = receipts / days if days > 0 else 0

        # Construct feature set
        X = pd.DataFrame([{
            "trip_duration_days": days,
            "miles_traveled": miles,
            "total_receipts_amount": receipts,
            "miles_per_day": miles_per_day,
            "receipts_per_day": receipts_per_day,
            "miles_times_receipts": miles * receipts,
            "days_squared": days ** 2,
            "efficiency_score": miles_per_day / (receipts_per_day + 1),
            "is_5_day_bonus": int(days == 5),
            "receipt_magic": int(str(f"{receipts:.2f}").endswith("49") or str(f"{receipts:.2f}").endswith("99")),
            "overspend_penalty": int(receipts > 1500)
        }])

        correction = model.predict(X)[0]
        predicted = round(base + correction, 2)

        error = abs(predicted - expected)
        total_error += error
        successful += 1
        if error < 0.01:
            exact += 1
        if error < 1.0:
            close += 1

        results.append((error, idx + 1, predicted, expected, days, miles, receipts))

        if error > max_error:
            max_error = error

    except Exception:
        results.append(("ERROR", idx + 1, "-", expected, days, miles, receipts))

# Summary
avg_error = total_error / successful if successful else 0
score = round(avg_error * 100 + (len(cases) - exact) * 0.1, 2)
exact_pct = round(100 * exact / successful, 1) if successful else 0
close_pct = round(100 * close / successful, 1) if successful else 0

print("\n✅ Evaluation Complete!\n")
print("📈 Results Summary:")
print(f"  Total test cases: {len(cases)}")
print(f"  Successful runs: {successful}")
print(f"  Exact matches (±$0.01): {exact} ({exact_pct}%)")
print(f"  Close matches (±$1.00): {close} ({close_pct}%)")
print(f"  Average error: ${avg_error:.2f}")
print(f"  Maximum error: ${max_error:.2f}\n")
print(f"🎯 Your Score: {score} (lower is better)\n")

# High-error cases
print("💡 Tips for improvement:")
print("  Check these high-error cases:")
for e, i, pred, exp, d, m, r in sorted(results, reverse=True)[:5]:
    if e == "ERROR":
        continue
    print(f"    Case {i}: {d} days, {m} miles, ${r} receipts")
    print(f"      Expected: ${exp:.2f}, Got: ${pred:.2f}, Error: ${e:.2f}")

# Show any failed cases
error_cases = [r for r in results if r[0] == "ERROR"]
if error_cases:
    print("\n⚠️  Errors encountered:")
    for r in error_cases[:10]:
        print(f"  Case {r[1]}: Invalid output")
    if len(error_cases) > 10:
        print(f"  ... and {len(error_cases) - 10} more errors")

print("\n📝 Next steps:")
print("  1. Fix any script errors shown above")
print("  2. Ensure your run.sh outputs only a number")
print("  3. Analyze the patterns in the interviews and public cases")
print("  4. Test edge cases around trip length and receipt amounts")
print("  5. Submit your solution via the Google Form when ready!")
