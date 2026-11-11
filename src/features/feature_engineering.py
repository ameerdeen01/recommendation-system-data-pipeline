
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler 
import joblib


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed"
FEATURE_DIR = BASE_DIR / "data" / "features"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. USER FEATURE ENGINEERING
# ==========================================

def create_user_features():

    print("\n===================================")
    print("       USER FEATURE ENGINEERING")
    print("===================================")

    # Load user demographic data
    users = pd.read_csv(
        DATA_DIR / "users_standardized.csv"
    )

    # Load user activity data
    activity = pd.read_csv(
        DATA_DIR / "user_activity.csv"
    )

    # Merge user and activity information
    user_features = users.merge(
        activity,
        on="user_id",
        how="left"
    )

    # Handle users without training interactions
    user_features["activity_count"] = (
        user_features["activity_count"].fillna(0).astype(int)
    )

    user_features["average_rating"] = (
        user_features["average_rating"].fillna(0)
    )

    # Save user features
    output_file = FEATURE_DIR / "user_features.csv"

    user_features.to_csv(
        output_file,
        index=False
    )

    # Display summary
    print("\nTotal Users:", len(user_features))

    print("\nUser Feature Columns:")
    print(user_features.columns.tolist())

    print("\nSample User Features:")
    print(user_features.head())

    print("\nUser Features Generated Successfully!")

    print("Saved Location:", output_file)


# ==========================================
# 2. MOVIE FEATURE ENGINEERING
# ==========================================

def create_movie_features():

    print("\n===================================")
    print("       MOVIE FEATURE ENGINEERING")
    print("===================================")

    # Load movie dataset
    movies = pd.read_csv(
        DATA_DIR / "movies_standardized.csv"
    )

    # Load movie popularity dataset
    popularity = pd.read_csv(
        DATA_DIR / "item_popularity.csv"
    )

    # Extract release year
    movies["release_year"] = pd.to_datetime(
        movies["release_date"],
        errors="coerce"
    ).dt.year

    # Merge popularity information
    movie_features = movies.merge(
        popularity,
        on="movie_id",
        how="left"
    )

    # Fill missing popularity values
    movie_features["interaction_count"] = (
        movie_features["interaction_count"].fillna(0).astype(int)
    )

    movie_features["average_rating"] = (
        movie_features["average_rating"].fillna(0)
    )

    # Save movie features
    output_file = FEATURE_DIR / "movie_features.csv"

    movie_features.to_csv(
        output_file,
        index=False
    )

    print("\nTotal Movies:", len(movie_features))

    print("\nMovie Feature Columns:")
    print(movie_features.columns.tolist())

    print("\nSample Movie Features:")
    print(movie_features.head())

    print("\nMovie Features Generated Successfully!")

    print("Saved Location:", output_file)


# ==========================================
# 3. INTERACTION FEATURE ENGINEERING
# ==========================================

def create_interaction_features():

    print("\n===================================")
    print("    INTERACTION FEATURE ENGINEERING")
    print("===================================")

    # Load training interactions only
    interactions = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    # Convert timestamp
    interactions["timestamp"] = pd.to_datetime(
        interactions["timestamp"],
        utc=True
    )

    # Create rating categories
    interactions["rating_category"] = pd.cut(
        interactions["rating"],
        bins=[0, 2, 3, 5],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )

    # Extract date features
    interactions["rating_year"] = (
        interactions["timestamp"].dt.year
    )

    interactions["rating_month"] = (
        interactions["timestamp"].dt.month
    )

    interactions["rating_day"] = (
        interactions["timestamp"].dt.day
    )

    interactions["rating_dayofweek"] = (
        interactions["timestamp"].dt.dayofweek
    )

    # Save interaction features
    output_file = (
        FEATURE_DIR / "interaction_features.csv"
    )

    interactions.to_csv(
        output_file,
        index=False
    )

    print("\nTotal Interactions:", len(interactions))

    print("\nInteraction Feature Columns:")
    print(interactions.columns.tolist())

    print("\nSample Interaction Features:")
    print(interactions.head())

    print("\nRating Category Distribution:")
    print(interactions["rating_category"].value_counts())

    print("\nInteraction Features Generated Successfully!")

    print("Saved Location:", output_file)


# ==========================================
# 4. FEATURE TRANSFORMATION AND SCALING
# ==========================================

def transform_features():

    print("\n===================================")
    print("    FEATURE TRANSFORMATION")
    print("===================================")

    SCALER_DIR = FEATURE_DIR / "scalers"
    SCALER_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------
    # USER FEATURE SCALING
    # --------------------------------------

    users = pd.read_csv(
        FEATURE_DIR / "user_features.csv"
    )

    user_columns = [
        "age",
        "activity_count",
        "average_rating"
    ]

    user_scaler = StandardScaler()

    user_scaled = user_scaler.fit_transform(
        users[user_columns]
    )

    for index, column in enumerate(user_columns):
        users[f"{column}_scaled"] = user_scaled[:, index]

    users.to_csv(
        FEATURE_DIR / "user_features_scaled.csv",
        index=False
    )

    joblib.dump(
        user_scaler,
        SCALER_DIR / "user_scaler.pkl"
    )

    # --------------------------------------
    # MOVIE FEATURE SCALING
    # --------------------------------------

    movies = pd.read_csv(
        FEATURE_DIR / "movie_features.csv"
    )

    movie_columns = [
        "release_year",
        "interaction_count",
        "average_rating"
    ]

    # Replace missing release years with median
    movies[movie_columns] = movies[movie_columns].fillna(
        movies[movie_columns].median()
    )

    movie_scaler = StandardScaler()

    movie_scaled = movie_scaler.fit_transform(
        movies[movie_columns]
    )

    for index, column in enumerate(movie_columns):
        movies[f"{column}_scaled"] = movie_scaled[:, index]

    movies.to_csv(
        FEATURE_DIR / "movie_features_scaled.csv",
        index=False
    )

    joblib.dump(
        movie_scaler,
        SCALER_DIR / "movie_scaler.pkl"
    )

    # --------------------------------------
    # SUMMARY
    # --------------------------------------

    print("\nUSER FEATURES SCALED")
    print(users[
        [f"{column}_scaled" for column in user_columns]
    ].head())

    print("\nMOVIE FEATURES SCALED")
    print(movies[
        [f"{column}_scaled" for column in movie_columns]
    ].head())

    print("\nFeature transformation completed!")

    print("\nGenerated files:")
    print("user_features_scaled.csv")
    print("movie_features_scaled.csv")
    print("user_scaler.pkl")
    print("movie_scaler.pkl")

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    create_user_features()

    create_movie_features()

    create_interaction_features()

    transform_features()
