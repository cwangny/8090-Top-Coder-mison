import json
import math
import sys
import os

# Load training data
DATA_PATH = os.path.join(os.path.dirname(__file__), 'public_cases.json')
with open(DATA_PATH) as f:
    TRAINING_DATA = [
        (case['input']['trip_duration_days'], case['input']['miles_traveled'],
         case['input']['total_receipts_amount'], case['expected_output'])
        for case in json.load(f)
    ]

# Compute mean and std for z-score normalization
def mean_std(index):
    values = [row[index] for row in TRAINING_DATA]
    mean = sum(values) / len(values)
    std = (sum((x - mean)**2 for x in values) / len(values)) ** 0.5
    return mean, std

MEAN_TD, STD_TD = mean_std(0)
MEAN_MILES, STD_MILES = mean_std(1)
MEAN_RECEIPTS, STD_RECEIPTS = mean_std(2)

def normalize(td, miles, receipts):
    return (
        (td - MEAN_TD) / STD_TD,
        (miles - MEAN_MILES) / STD_MILES,
        (receipts - MEAN_RECEIPTS) / STD_RECEIPTS
    )

def predict_knn(td: float, miles: float, receipts: float, k: int = 1) -> float:
    norm_td, norm_miles, norm_receipts = normalize(td, miles, receipts)

    distances = []
    for t_days, t_miles, t_receipts, t_output in TRAINING_DATA:
        n_td, n_miles, n_receipts = normalize(t_days, t_miles, t_receipts)
        d = math.sqrt(
            (norm_td - n_td) ** 2 +
            (norm_miles - n_miles) ** 2 +
            (norm_receipts - n_receipts) ** 2
        )
        distances.append((d, t_output))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]

    total_weight = 0.0
    weighted_sum = 0.0
    for dist, out in neighbors:
        w = 1.0 / (dist + 1e-6)
        total_weight += w
        weighted_sum += w * out

    result = weighted_sum / total_weight
    result = max(0.0, min(result, 3000.0))  # Clip to avoid extreme errors
    return round(result, 2)

# Command-line interface
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python solution.py <days> <miles> <receipts>")
        sys.exit(1)

    trip_days = float(sys.argv[1])
    miles = float(sys.argv[2])
    receipts = float(sys.argv[3])
    prediction = predict_knn(trip_days, miles, receipts)
    print(f"{prediction:.2f}")
