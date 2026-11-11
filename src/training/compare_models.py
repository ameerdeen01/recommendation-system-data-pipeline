
import pandas as pd
import os

# Model evaluation results
results = pd.DataFrame([
    {
        "Model": "Popularity",
        "Evaluated_Interactions": 20000,
        "MAE": 0.8439,
        "RMSE": 1.0392
    },
    {
        "Model": "Collaborative Filtering",
        "Evaluated_Interactions": 20000,
        "MAE": 0.9202,
        "RMSE": 1.1017
    }
])

# Sort models by RMSE (lower is better)
results = results.sort_values(by="RMSE")

# Create reports folder if it doesn't exist
os.makedirs("reports", exist_ok=True)

# Save comparison report
results.to_csv("reports/model_comparison.csv", index=False)

# Display results
print("\nMODEL COMPARISON REPORT")
print("=" * 50)

print(results.to_string(index=False))

print("\nBEST PERFORMING MODEL")
print("=" * 50)

print(results.iloc[0]["Model"])

print("\nReport saved successfully:")
print("reports/model_comparison.csv")
