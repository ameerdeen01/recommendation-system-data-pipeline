
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_popularity_model():
    print("\nPOPULARITY MODEL EVALUATION")
    print("=" * 40)

    test = pd.read_csv("data/processed/test.csv")
    popularity = pd.read_csv("models/popularity_model.csv")

    # Calculate fallback prediction from training ratings
    train = pd.read_csv("data/processed/train.csv")
    global_mean = train["rating"].mean()

    # Keep all test interactions
    evaluation = test.merge(
        popularity[["movie_id", "weighted_rating"]],
        on="movie_id",
        how="left"
    )

    # Use training mean when a movie has no popularity score
    evaluation["weighted_rating"] = (
        evaluation["weighted_rating"].fillna(global_mean)
    )

    y_true = evaluation["rating"]
    y_pred = evaluation["weighted_rating"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"Evaluated interactions: {len(evaluation)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")


if __name__ == "__main__":
    evaluate_popularity_model()
