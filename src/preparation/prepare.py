
import pandas as pd
from pathlib import Path


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. HANDLE MISSING VALUES
# ==========================================

def handle_missing_values():

    print("\n===================================")
    print("      HANDLING MISSING VALUES")
    print("===================================")

    # Load ingested datasets
    users = pd.read_csv(
        DATA_DIR / "users_ingested.csv"
    )

    movies = pd.read_csv(
        DATA_DIR / "movies_ingested.csv"
    )

    ratings = pd.read_csv(
        DATA_DIR / "ratings_ingested.csv"
    )

    # --------------------------------------
    # USERS DATASET
    # --------------------------------------

    print("\nUSERS DATASET")

    print("Missing Before:", users.isnull().sum().sum())

    # No missing values found
    users_clean = users.copy()

    print("Missing After:", users_clean.isnull().sum().sum())

    # --------------------------------------
    # MOVIES DATASET
    # --------------------------------------

    print("\nMOVIES DATASET")

    print("Missing Before:", movies.isnull().sum().sum())

    # Remove completely empty column
    if "video_release_date" in movies.columns:
        movies = movies.drop(
            columns=["video_release_date"]
        )
    # Replace missing release dates
    movies["release_date"] = (
        movies["release_date"].fillna("Unknown")
    )

    # Replace missing IMDb URLs
    movies["imdb_url"] = (
        movies["imdb_url"].fillna("")
    )

    movies_clean = movies.copy()

    print("Missing After:", movies_clean.isnull().sum().sum())

    # --------------------------------------
    # RATINGS DATASET
    # --------------------------------------

    print("\nRATINGS DATASET")

    print("Missing Before:", ratings.isnull().sum().sum())

    # No missing values found
    ratings_clean = ratings.copy()

    print("Missing After:", ratings_clean.isnull().sum().sum())

    # --------------------------------------
    # SAVE CLEANED DATASETS
    # --------------------------------------

    users_clean.to_csv(
        OUTPUT_DIR / "users_clean.csv",
        index=False
    )

    movies_clean.to_csv(
        OUTPUT_DIR / "movies_clean.csv",
        index=False
    )

    ratings_clean.to_csv(
        OUTPUT_DIR / "ratings_clean.csv",
        index=False
    )

    # --------------------------------------
    # SUMMARY
    # --------------------------------------

    print("\n===================================")
    print("       DATA CLEANING SUMMARY")
    print("===================================")

    print("Users Records:", len(users_clean))
    print("Movies Records:", len(movies_clean))
    print("Ratings Records:", len(ratings_clean))

    print("\nAll cleaned datasets saved successfully!")

    print("\nOutput Location:", OUTPUT_DIR)



# ==========================================
# 2. DUPLICATE RECORD REMOVAL
# ==========================================

def remove_duplicates():

    print("\n===================================")
    print("       DUPLICATE RECORD REMOVAL")
    print("===================================")

    datasets = {
        "Users": (
            "users_clean.csv",
            ["user_id"]
        ),
        "Movies": (
            "movies_clean.csv",
            ["movie_id"]
        ),
        "Ratings": (
            "ratings_clean.csv",
            ["user_id", "movie_id"]
        )
    }

    for name, (filename, keys) in datasets.items():

        df = pd.read_csv(DATA_DIR / filename)

        before = len(df)

        # Remove duplicate records
        df_clean = df.drop_duplicates(
            subset=keys,
            keep="first"
        )

        after = len(df_clean)

        removed = before - after

        # Save deduplicated dataset
        output_file = OUTPUT_DIR / f"{name.lower()}_deduplicated.csv"

        df_clean.to_csv(output_file, index=False)

        print(f"\n{name} Dataset")
        print("Records Before:", before)
        print("Records After:", after)
        print("Duplicates Removed:", removed)

        print("Saved:", output_file.name)

    print("\nDuplicate Removal Completed!")


# ==========================================
# 3. DATA STANDARDIZATION
# ==========================================

def standardize_data():

    print("\n===================================")
    print("        DATA STANDARDIZATION")
    print("===================================")

    # Load deduplicated datasets
    users = pd.read_csv(
        DATA_DIR / "users_deduplicated.csv"
    )

    movies = pd.read_csv(
        DATA_DIR / "movies_deduplicated.csv"
    )

    ratings = pd.read_csv(
        DATA_DIR / "ratings_deduplicated.csv"
    )

    # --------------------------------------
    # USERS DATASET
    # --------------------------------------

    users.columns = users.columns.str.strip().str.lower()

    users["gender"] = (
        users["gender"].astype("string").str.strip().str.upper()
    )

    users["occupation"] = (
        users["occupation"].astype("string").str.strip().str.lower()
    )

    users["zip_code"] = (
        users["zip_code"].astype("string").str.strip()
    )

    # --------------------------------------
    # MOVIES DATASET
    # --------------------------------------

    movies.columns = movies.columns.str.strip().str.lower()

    movies["title"] = (
        movies["title"].astype("string").str.strip()
    )

    movies["release_date"] = (
        movies["release_date"].astype("string").str.strip()
    )

    movies["imdb_url"] = (
        movies["imdb_url"].astype("string").str.strip()
    )

    # --------------------------------------
    # RATINGS DATASET
    # --------------------------------------

    ratings.columns = ratings.columns.str.strip().str.lower()

    ratings["user_id"] = ratings["user_id"].astype("int64")
    ratings["movie_id"] = ratings["movie_id"].astype("int64")
    ratings["rating"] = ratings["rating"].astype("int64")

    ratings["timestamp"] = pd.to_datetime(
        ratings["timestamp"],
        unit="s",
        utc=True
    )

    # --------------------------------------
    # SAVE STANDARDIZED DATASETS
    # --------------------------------------

    users.to_csv(
        OUTPUT_DIR / "users_standardized.csv",
        index=False
    )

    movies.to_csv(
        OUTPUT_DIR / "movies_standardized.csv",
        index=False
    )

    ratings.to_csv(
        OUTPUT_DIR / "ratings_standardized.csv",
        index=False
    )

    print("\nUsers Records:", len(users))
    print("Movies Records:", len(movies))
    print("Ratings Records:", len(ratings))

    print("\nStandardized Timestamp Example:")
    print(ratings["timestamp"].head())

    print("\nAll standardized datasets saved successfully!")


# ==========================================
# 4. TIME-BASED TRAIN TEST SPLIT
# ==========================================

def create_train_test_split():

    print("\n===================================")
    print("       TIME-BASED DATA SPLIT")
    print("===================================")

    ratings = pd.read_csv(
        DATA_DIR / "ratings_standardized.csv"
    )

    # Ensure timestamp is datetime
    ratings["timestamp"] = pd.to_datetime(
        ratings["timestamp"],
        utc=True
    )

    # Sort interactions chronologically
    ratings = ratings.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # 80% training and 20% testing
    split_index = int(len(ratings) * 0.80)

    train = ratings.iloc[:split_index].copy()

    test = ratings.iloc[split_index:].copy()

    # Save datasets
    train.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    test.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    # Display summary
    print("\nTotal Records:", len(ratings))

    print("Training Records:", len(train))

    print("Testing Records:", len(test))

    print("\nTraining Period:")
    print(train["timestamp"].min())
    print("To:", train["timestamp"].max())

    print("\nTesting Period:")
    print(test["timestamp"].min())
    print("To:", test["timestamp"].max())

    print("\nTrain/Test Split Completed Successfully!")


# ==========================================
# 5. CREATE DERIVED DATASETS
# ==========================================

def create_derived_datasets():

    print("\n===================================")
    print("       DERIVED DATASETS")
    print("===================================")

    train = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    # --------------------------------------
    # USER ACTIVITY SUMMARY
    # --------------------------------------

    user_activity = train.groupby("user_id").agg(
        activity_count=("movie_id", "count"),
        average_rating=("rating", "mean"),
        last_interaction=("timestamp", "max")
    ).reset_index()

    user_activity.to_csv(
        OUTPUT_DIR / "user_activity.csv",
        index=False
    )

    # --------------------------------------
    # ITEM POPULARITY
    # --------------------------------------

    item_popularity = train.groupby("movie_id").agg(
        interaction_count=("user_id", "count"),
        average_rating=("rating", "mean")
    ).reset_index()

    item_popularity.to_csv(
        OUTPUT_DIR / "item_popularity.csv",
        index=False
    )

    # --------------------------------------
    # SUMMARY
    # --------------------------------------

    print("\nUSER ACTIVITY SUMMARY")
    print("Total Users:", len(user_activity))
    print(user_activity.head())

    print("\nITEM POPULARITY SUMMARY")
    print("Total Movies:", len(item_popularity))
    print(item_popularity.head())

    print("\nDerived datasets created successfully!")


# ==========================================
# 6. USER-ITEM INTERACTION MATRIX
# ==========================================

def create_interaction_matrix():

    print("\n===================================")
    print("     USER-ITEM INTERACTION MATRIX")
    print("===================================")

    train = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    # Create user-item matrix
    interaction_matrix = train.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
        fill_value=0
    )

    # Save matrix
    output_file = (
        OUTPUT_DIR / "interaction_matrix.csv"
    )

    interaction_matrix.to_csv(output_file)

    # Display summary
    print("\nMatrix Shape:")
    print(interaction_matrix.shape)

    print("\nTotal Users:", interaction_matrix.shape[0])

    print("Total Movies:", interaction_matrix.shape[1])

    print("\nSample Interaction Matrix:")
    print(interaction_matrix.iloc[:5, :5])

    print("\nInteraction Matrix Saved Successfully!")

    print("Location:", output_file)


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    handle_missing_values()

    remove_duplicates()

    standardize_data()

    create_train_test_split()

    create_derived_datasets()

    create_interaction_matrix()
