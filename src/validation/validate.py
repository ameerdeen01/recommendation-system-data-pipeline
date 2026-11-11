
import pandas as pd
from pathlib import Path


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed"
ERROR_DIR = BASE_DIR / "data" / "errors"

ERROR_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. MISSING VALUES VALIDATION
# ==========================================

def check_missing_values():

    print("\n===================================")
    print("     MISSING VALUES VALIDATION")
    print("===================================")

    datasets = {
        "Users": "users_ingested.csv",
        "Movies": "movies_ingested.csv",
        "Ratings": "ratings_ingested.csv"
    }

    for dataset_name, filename in datasets.items():

        df = pd.read_csv(DATA_DIR / filename)

        missing = df.isnull().sum()
        total_missing = missing.sum()

        print(f"\n{dataset_name} Dataset")
        print("Total Records:", len(df))
        print("Total Missing Values:", total_missing)

        if total_missing > 0:
            print("\nMissing values by column:")
            print(missing[missing > 0])
        else:
            print("No missing values found.")

    print("\nMissing Values Validation Completed!")


# ==========================================
# 2. DUPLICATE RECORDS VALIDATION
# ==========================================

def check_duplicates():

    print("\n===================================")
    print("     DUPLICATE RECORD VALIDATION")
    print("===================================")

    datasets = {
        "Users": (
            "users_ingested.csv",
            ["user_id"]
        ),
        "Movies": (
            "movies_ingested.csv",
            ["movie_id"]
        ),
        "Ratings": (
            "ratings_ingested.csv",
            ["user_id", "movie_id"]
        )
    }

    for dataset_name, (filename, keys) in datasets.items():

        df = pd.read_csv(DATA_DIR / filename)

        duplicates = df[
            df.duplicated(subset=keys, keep=False)
        ]

        print(f"\n{dataset_name} Dataset")
        print("Total Records:", len(df))
        print("Duplicate Records:", len(duplicates))

        if not duplicates.empty:

            error_file = (
                ERROR_DIR /
                f"{dataset_name.lower()}_duplicates.csv"
            )

            duplicates.to_csv(
                error_file,
                index=False
            )

            print("Duplicates saved to:", error_file)

        else:
            print("No duplicate records found.")

    print("\nDuplicate Validation Completed!")


# ==========================================
# 3. DATA TYPE VALIDATION
# ==========================================

def check_data_types():

    print("\n===================================")
    print("       DATA TYPE VALIDATION")
    print("===================================")

    datasets = {
        "Users": "users_ingested.csv",
        "Movies": "movies_ingested.csv",
        "Ratings": "ratings_ingested.csv"
    }

    for dataset_name, filename in datasets.items():

        df = pd.read_csv(DATA_DIR / filename)

        print(f"\n{dataset_name} Dataset")
        print("Total Records:", len(df))

        print("\nColumn Data Types:")

        for column, dtype in df.dtypes.items():

            print(f"{column}: {dtype}")

    print("\nData Type Validation Completed!")


# ==========================================
# 4. INVALID RATINGS VALIDATION
# ==========================================

def check_invalid_ratings():

    print("\n===================================")
    print("      INVALID RATINGS VALIDATION")
    print("===================================")

    ratings = pd.read_csv(
        DATA_DIR / "ratings_ingested.csv"
    )

    # Valid MovieLens rating range: 1 to 5
    invalid_ratings = ratings[
        ~ratings["rating"].between(1, 5)
        | (ratings["rating"] % 1 != 0)
    ]

    valid_ratings = ratings[
        ratings["rating"].between(1, 5)
        & (ratings["rating"] % 1 == 0)
    ]

    print("\nTotal Ratings:", len(ratings))
    print("Valid Ratings:", len(valid_ratings))
    print("Invalid Ratings:", len(invalid_ratings))

    # Save invalid records separately
    if not invalid_ratings.empty:

        error_file = ERROR_DIR / "invalid_ratings.csv"

        invalid_ratings.to_csv(
            error_file,
            index=False
        )

        print("\nInvalid records saved to:", error_file)

        print("\nInvalid Rating Samples:")
        print(invalid_ratings.head())

    else:
        print("\nNo invalid ratings found.")

    print("\nInvalid Ratings Validation Completed!")



# ==========================================
# 5. INVALID TIMESTAMP VALIDATION
# ==========================================

def check_invalid_timestamps():

    print("\n===================================")
    print("    INVALID TIMESTAMP VALIDATION")
    print("===================================")

    ratings = pd.read_csv(
        DATA_DIR / "ratings_ingested.csv"
    )

    # Convert timestamp to numeric
    timestamps = pd.to_numeric(
        ratings["timestamp"],
        errors="coerce"
    )

    # Convert Unix timestamps into UTC dates
    dates = pd.to_datetime(
        timestamps,
        unit="s",
        errors="coerce",
        utc=True
    )

    # Define reasonable timestamp boundaries
    min_date = pd.Timestamp("1990-01-01", tz="UTC")
    max_date = pd.Timestamp("2030-01-01", tz="UTC")

    # Identify invalid timestamps
    invalid_mask = (
        timestamps.isna()
        | (timestamps <= 0)
        | (timestamps % 1 != 0)
        | dates.isna()
        | (dates < min_date)
        | (dates >= max_date)
    )

    invalid_records = ratings[invalid_mask].copy()

    valid_records = ratings[~invalid_mask]

    print("\nTotal Records:", len(ratings))
    print("Valid Timestamps:", len(valid_records))
    print("Invalid Timestamps:", len(invalid_records))

    # Save invalid records
    if not invalid_records.empty:

        error_file = ERROR_DIR / "invalid_timestamps.csv"

        invalid_records.to_csv(
            error_file,
            index=False
        )

        print("\nInvalid records saved to:", error_file)

        print("\nInvalid Timestamp Samples:")
        print(invalid_records.head())

    else:
        print("\nNo invalid timestamps found.")

    print("\nInvalid Timestamp Validation Completed!")


# ==========================================
# 6. REFERENTIAL INTEGRITY VALIDATION
# ==========================================

def check_referential_integrity():

    print("\n===================================")
    print("  REFERENTIAL INTEGRITY VALIDATION")
    print("===================================")

    users = pd.read_csv(
        DATA_DIR / "users_ingested.csv"
    )

    movies = pd.read_csv(
        DATA_DIR / "movies_ingested.csv"
    )

    ratings = pd.read_csv(
        DATA_DIR / "ratings_ingested.csv"
    )

    # Check whether user IDs exist
    invalid_users = ~ratings["user_id"].isin(
        users["user_id"]
    )

    # Check whether movie IDs exist
    invalid_movies = ~ratings["movie_id"].isin(
        movies["movie_id"]
    )

    # Identify records with invalid references
    invalid_mask = invalid_users | invalid_movies

    invalid_records = ratings[invalid_mask].copy()

    valid_records = ratings[~invalid_mask]

    print("\nTotal Ratings:", len(ratings))

    print("Valid References:", len(valid_records))

    print("Invalid User References:", invalid_users.sum())

    print("Invalid Movie References:", invalid_movies.sum())

    print("Total Invalid Records:", len(invalid_records))

    # Save invalid records
    if not invalid_records.empty:

        error_file = (
            ERROR_DIR / "invalid_references.csv"
        )

        invalid_records.to_csv(
            error_file,
            index=False
        )

        print("\nInvalid records saved to:", error_file)

        print("\nInvalid Record Samples:")
        print(invalid_records.head())

    else:

        print("\nAll user and movie references are valid.")

    print("\nReferential Integrity Validation Completed!")


# ==========================================
# 7. GENERATE DATA VALIDATION REPORT
# ==========================================

def generate_validation_report():

    print("\n===================================")
    print("    GENERATING VALIDATION REPORT")
    print("===================================")

    users = pd.read_csv(DATA_DIR / "users_ingested.csv")
    movies = pd.read_csv(DATA_DIR / "movies_ingested.csv")
    ratings = pd.read_csv(DATA_DIR / "ratings_ingested.csv")

    report = []

    def add_result(check, dataset, total, failed, status):

        report.append({
            "validation_check": check,
            "dataset": dataset,
            "total_records": total,
            "failed_records": failed,
            "status": status
        })

    # 1. Missing values
    for name, df in [
        ("Users", users),
        ("Movies", movies),
        ("Ratings", ratings)
    ]:

        missing = int(df.isnull().sum().sum())

        add_result(
            "Missing Values",
            name,
            len(df),
            missing,
            "PASS" if missing == 0 else "WARNING"
        )

    # 2. Duplicate records
    duplicate_checks = [
        ("Users", users, ["user_id"]),
        ("Movies", movies, ["movie_id"]),
        ("Ratings", ratings, ["user_id", "movie_id"])
    ]

    for name, df, keys in duplicate_checks:

        duplicates = int(
            df.duplicated(subset=keys, keep=False).sum()
        )

        add_result(
            "Duplicate Records",
            name,
            len(df),
            duplicates,
            "PASS" if duplicates == 0 else "FAIL"
        )

    # 3. Data type checks
    expected_types = {
        "Users": {
            "user_id": "integer",
            "age": "integer",
            "gender": "string",
            "occupation": "string",
            "zip_code": "string"
        },
        "Movies": {
            "movie_id": "integer",
            "title": "string"
        },
        "Ratings": {
            "user_id": "integer",
            "movie_id": "integer",
            "rating": "integer",
            "timestamp": "integer"
        }
    }

    for name, df in [
        ("Users", users),
        ("Movies", movies),
        ("Ratings", ratings)
    ]:

        failures = 0

        for column, expected in expected_types[name].items():

            actual = df[column].dtype

            if expected == "integer" and not pd.api.types.is_integer_dtype(actual):
                failures += 1

            elif expected == "string" and not pd.api.types.is_string_dtype(actual):
                failures += 1

        add_result(
            "Data Types",
            name,
            len(df),
            failures,
            "PASS" if failures == 0 else "FAIL"
        )

    # 4. Invalid ratings
    invalid_ratings = (
        ~ratings["rating"].between(1, 5)
        | (ratings["rating"] % 1 != 0)
    ).sum()

    add_result(
        "Invalid Ratings",
        "Ratings",
        len(ratings),
        int(invalid_ratings),
        "PASS" if invalid_ratings == 0 else "FAIL"
    )

    # 5. Invalid timestamps
    timestamps = pd.to_numeric(
        ratings["timestamp"],
        errors="coerce"
    )

    dates = pd.to_datetime(
        timestamps,
        unit="s",
        errors="coerce",
        utc=True
    )

    invalid_timestamps = (
        timestamps.isna()
        | (timestamps <= 0)
        | (timestamps % 1 != 0)
        | dates.isna()
        | (dates < pd.Timestamp("1990-01-01", tz="UTC"))
        | (dates >= pd.Timestamp("2030-01-01", tz="UTC"))
    ).sum()

    add_result(
        "Invalid Timestamps",
        "Ratings",
        len(ratings),
        int(invalid_timestamps),
        "PASS" if invalid_timestamps == 0 else "FAIL"
    )

    # 6. Referential integrity
    invalid_users = ~ratings["user_id"].isin(users["user_id"])
    invalid_movies = ~ratings["movie_id"].isin(movies["movie_id"])

    invalid_references = int(
        (invalid_users | invalid_movies).sum()
    )

    add_result(
        "Referential Integrity",
        "Ratings",
        len(ratings),
        invalid_references,
        "PASS" if invalid_references == 0 else "FAIL"
    )

    # Save report
    report_df = pd.DataFrame(report)

    report_file = BASE_DIR / "reports" / "validation_report.csv"

    report_file.parent.mkdir(parents=True, exist_ok=True)

    report_df.to_csv(report_file, index=False)

    print(report_df.to_string(index=False))

    print("\nValidation report saved successfully!")
    print("Location:", report_file)




# ==========================================
# MAIN EXECUTION
# ==========================================

def main():

    print("\n===================================")
    print("       DATA VALIDATION PIPELINE")
    print("===================================")

    check_missing_values()

    check_duplicates()

    check_data_types()

    check_invalid_ratings()

    check_invalid_timestamps()

    check_referential_integrity()

    generate_validation_report()

    print("\n===================================")
    print("   DATA VALIDATION COMPLETED")
    print("===================================")


if __name__ == "__main__":
    main()
