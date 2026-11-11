
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_collaborative_filtering():
    print("\nCOLLABORATIVE FILTERING EVALUATION")
    print("=" * 45)

    # Load test data and trained model
    test = pd.read_csv("data/processed/test.csv")
    model = joblib.load(
        "models/collaborative_filtering_model.pkl"
    )

    user_item_matrix = model["user_item_matrix"]
    similarity_matrix = model["similarity_matrix"]

    global_mean = test["rating"].mean()
    predictions = []

    for row in test.itertuples(index=False):
        user_id = row.user_id
        movie_id = row.movie_id

        # Fallback for users or movies absent from training
        if (
            user_id not in user_item_matrix.index
            or movie_id not in user_item_matrix.columns
        ):
            predictions.append(global_mean)
            continue

        # Similarity scores for the target user
        similarities = similarity_matrix.loc[user_id]

        # Ratings given to this movie by other users
        movie_ratings = user_item_matrix[movie_id]

        # Exclude the target user and users who did not rate this movie
        neighbors = movie_ratings[
            (movie_ratings > 0) &
            (movie_ratings.index != user_id)
        ]

        if neighbors.empty:
            predictions.append(global_mean)
            continue

        neighbor_similarities = similarities.loc[neighbors.index]

        # Select the 20 most similar users
        top_neighbors = neighbor_similarities.nlargest(20)

        ratings = movie_ratings.loc[top_neighbors.index]
        weights = top_neighbors.values

        if np.abs(weights).sum() == 0:
            prediction = global_mean
        else:
            prediction = np.dot(ratings.values, weights) / np.abs(weights).sum()

        predictions.append(np.clip(prediction, 1, 5))

    # Calculate evaluation metrics
    y_true = test["rating"]
    y_pred = predictions

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"Evaluated interactions: {len(test)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")


if __name__ == "__main__":
    evaluate_collaborative_filtering()
