
import pandas as pd
from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Dataset directory
RAW_DIR = BASE_DIR / "data" / "raw" / "ml-100k"

# Output directory
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ingest_data():

    print("\n===================================")
    print("   MOVIELENS DATA INGESTION")
    print("===================================\n")

    # 1. Load Users
    users = pd.read_csv(
        RAW_DIR / "u.user",
        sep="|",
        header=None,
        names=[
            "user_id",
            "age",
            "gender",
            "occupation",
            "zip_code"
        ]
    )

    # 2. Load Movies
    genre_columns = [
        "unknown", "action", "adventure", "animation",
        "children", "comedy", "crime", "documentary",
        "drama", "fantasy", "film_noir", "horror",
        "musical", "mystery", "romance", "sci_fi",
        "thriller", "war", "western"
    ]

    movie_columns = [
        "movie_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url"
    ] + genre_columns

    movies = pd.read_csv(
        RAW_DIR / "u.item",
        sep="|",
        header=None,
        names=movie_columns,
        encoding="latin-1"
    )

    # 3. Load Ratings / Interactions
    ratings = pd.read_csv(
        RAW_DIR / "u.data",
        sep="\t",
        header=None,
        names=[
            "user_id",
            "movie_id",
            "rating",
            "timestamp"
        ]
    )

    # 4. Save ingested datasets
    users.to_csv(
        OUTPUT_DIR / "users_ingested.csv",
        index=False
    )

    movies.to_csv(
        OUTPUT_DIR / "movies_ingested.csv",
        index=False
    )

    ratings.to_csv(
        OUTPUT_DIR / "ratings_ingested.csv",
        index=False
    )

    # 5. Display ingestion summary
    print("USER DATA")
    print("Records:", len(users))
    print(users.head())

    print("\nMOVIE DATA")
    print("Records:", len(movies))
    print(movies[["movie_id", "title"]].head())

    print("\nRATINGS DATA")
    print("Records:", len(ratings))
    print(ratings.head())

    print("\n===================================")
    print("INGESTION SUMMARY")
    print("===================================")

    print("Total Users:", len(users))
    print("Total Movies:", len(movies))
    print("Total Ratings:", len(ratings))

    print("\nAll datasets ingested successfully!")


if __name__ == "__main__":
    ingest_data()
