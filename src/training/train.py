
import pandas as pd
from pathlib import Path


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed"
FEATURE_DIR = BASE_DIR / "data" / "features"


# ==========================================
# 1. LOAD TRAINING DATASETS
# ==========================================

def load_training_data():

    print("\n===================================")
    print("      PREPARING TRAINING DATA")
    print("===================================")

    # Load training interactions
    train = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    # Load testing interactions
    test = pd.read_csv(
        DATA_DIR / "test.csv"
    )

    # Load user features
    users = pd.read_csv(
        FEATURE_DIR / "user_features.csv"
    )

    # Load movie features
    movies = pd.read_csv(
        FEATURE_DIR / "movie_features.csv"
    )

    # Verify required columns
    required_columns = [
        "user_id",
        "movie_id",
        "rating"
    ]

    for column in required_columns:

        if column not in train.columns:
            raise ValueError(
                f"Missing column in training data: {column}"
            )

        if column not in test.columns:
            raise ValueError(
                f"Missing column in testing data: {column}"
            )

    # Display dataset information
    print("\nTRAINING DATA")
    print("Total Records:", len(train))
    print(train.head())

    print("\nTESTING DATA")
    print("Total Records:", len(test))
    print(test.head())

    print("\nUSER FEATURES")
    print("Total Users:", len(users))

    print("\nMOVIE FEATURES")
    print("Total Movies:", len(movies))

    print("\nTraining datasets loaded successfully!")


# ==========================================
# 2. POPULARITY-BASED RECOMMENDATION MODEL
# ==========================================

def train_popularity_model():

    print("\n===================================")
    print("  POPULARITY-BASED MODEL TRAINING")
    print("===================================")

    # Load training interactions
    train = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    # Load movie metadata
    movies = pd.read_csv(
        FEATURE_DIR / "movie_features.csv"
    )

    # Calculate movie rating statistics
    movie_stats = train.groupby("movie_id").agg(
        rating_count=("rating", "count"),
        average_rating=("rating", "mean")
    ).reset_index()

    # Calculate global average rating
    C = train["rating"].mean()

    # Minimum number of ratings (60th percentile)
    m = movie_stats["rating_count"].quantile(0.60)

    # Calculate weighted rating
    movie_stats["weighted_rating"] = (
        (movie_stats["rating_count"] /
         (movie_stats["rating_count"] + m))
        * movie_stats["average_rating"]
        +
        (m /
         (movie_stats["rating_count"] + m))
        * C
    )

    # Merge movie titles
    popularity_model = movie_stats.merge(
        movies[["movie_id", "title"]],
        on="movie_id",
        how="left"
    )

    # Sort movies by weighted rating
    popularity_model = popularity_model.sort_values(
        "weighted_rating",
        ascending=False
    )

    # Create model output directory
    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save trained model results
    output_file = MODEL_DIR / "popularity_model.csv"

    popularity_model.to_csv(
        output_file,
        index=False
    )

    # Display top recommendations
    print("\nGlobal Average Rating:", round(C, 3))
    print("Minimum Rating Threshold:", round(m, 2))

    print("\nTOP 10 RECOMMENDED MOVIES")

    print(
        popularity_model[
            ["movie_id", "title", "rating_count",
             "average_rating", "weighted_rating"]
        ].head(10).to_string(index=False)
    )

    print("\nPopularity Model Trained Successfully!")

    print("Saved Location:", output_file)


# ==========================================
# 3. USER-BASED COLLABORATIVE FILTERING
# ==========================================

def train_collaborative_filtering():

    print("\n===================================")
    print(" USER-BASED COLLABORATIVE FILTERING")
    print("===================================")

    from sklearn.metrics.pairwise import cosine_similarity
    import joblib

    # Load training interactions
    train = pd.read_csv(DATA_DIR / "train.csv")

    # Create user-item matrix using training data only
    user_item_matrix = train.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
        fill_value=0
    )

    # Calculate user similarity
    similarity_matrix = cosine_similarity(user_item_matrix)

    # Create similarity DataFrame
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

    # Save model
    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_data = {
        "user_item_matrix": user_item_matrix,
        "similarity_matrix": similarity_df
    }

    joblib.dump(
        model_data,
        MODEL_DIR / "collaborative_filtering_model.pkl"
    )

    print("\nTotal Users:", user_item_matrix.shape[0])
    print("Total Movies:", user_item_matrix.shape[1])

    print("\nSimilarity Matrix Shape:")
    print(similarity_df.shape)

    print("\nSample User Similarities:")
    print(similarity_df.iloc[:5, :5])

    print("\nCollaborative Filtering Model Trained Successfully!")

    print(
        "Saved Location:",
        MODEL_DIR / "collaborative_filtering_model.pkl"
    )


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    load_training_data()

    train_popularity_model()

    train_collaborative_filtering()
