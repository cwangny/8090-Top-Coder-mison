import sys

def hybrid_reimbursement(days, miles, receipts):
    # === Base Linear Formula ===
    reimbursement = (
        50 * days +
        0.50 * miles +
        0.40 * receipts +
        200
    )

    # === Rule-based Corrections ===

    # 1. 5-day trip bonus
    if days == 5:
        reimbursement += 50

    # 2. Efficiency bonus: sweet spot of miles per day
    miles_per_day = miles / days if days > 0 else 0
    if 150 <= miles_per_day <= 220:
        reimbursement *= 1.10
    elif miles_per_day > 300:
        reimbursement *= 0.95  # Penalty for unrealistic driving

    # 3. Receipt rounding bug: add bonus for .49 or .99
    receipt_str = f"{receipts:.2f}"
    if receipt_str.endswith("49") or receipt_str.endswith("99"):
        reimbursement += 10

    # 4. Penalty for excessive spending
    if receipts > 1500:
        reimbursement -= 0.05 * (receipts - 1500)

    # 5. Soft cap for maximum reimbursement
    reimbursement = min(reimbursement, 2000)

    return round(reimbursement, 2)

if __name__ == "__main__":
    try:
        if len(sys.argv) != 4:
            raise ValueError("Expected 3 arguments")

        days = int(sys.argv[1])
        miles = float(sys.argv[2])
        receipts = float(sys.argv[3])

        result = hybrid_reimbursement(days, miles, receipts)
        print(f"{result:.2f}")

    except Exception as e:
        print("ERROR", file=sys.stderr)
        print(f"Exception: {e}", file=sys.stderr)
        print("0.00")
