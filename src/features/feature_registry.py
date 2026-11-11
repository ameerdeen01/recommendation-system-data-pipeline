
import json
from pathlib import Path


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

FEATURE_DIR = BASE_DIR / "data" / "features"


# ==========================================
# FEATURE METADATA REGISTRY
# ==========================================

def create_feature_registry():

    print("\n===================================")
    print("       FEATURE METADATA REGISTRY")
    print("===================================")

    feature_metadata = {

        "user_features": {
            "file": "user_features_scaled.csv",
            "description": "User demographic and activity features",
            "features": [
                "user_id",
                "age",
                "gender",
                "occupation",
                "zip_code",
                "activity_count",
                "average_rating",
                "age_scaled",
                "activity_count_scaled",
                "average_rating_scaled"
            ]
        },

        "movie_features": {
            "file": "movie_features_scaled.csv",
            "description": "Movie metadata, genre and popularity features",
            "features": [
                "movie_id",
                "title",
                "release_year",
                "interaction_count",
                "average_rating",
                "release_year_scaled",
                "interaction_count_scaled",
                "average_rating_scaled"
            ]
        },

        "interaction_features": {
            "file": "interaction_features.csv",
            "description": "User movie interaction and rating features",
            "features": [
                "user_id",
                "movie_id",
                "rating",
                "timestamp",
                "rating_category",
                "rating_year",
                "rating_month",
                "rating_day",
                "rating_dayofweek"
            ]
        }
    }

    # Save metadata as JSON
    output_file = FEATURE_DIR / "feature_metadata.json"

    with open(output_file, "w", encoding="utf-8") as file:

        json.dump(
            feature_metadata,
            file,
            indent=4
        )

    print("\nRegistered Feature Groups:")

    for name, details in feature_metadata.items():

        print("\nFeature Group:", name)
        print("File:", details["file"])
        print("Feature Count:", len(details["features"]))

    print("\nFeature Metadata Registry Created Successfully!")

    print("Location:", output_file)


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    create_feature_registry()
