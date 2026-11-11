# End-to-End Data Management Pipeline for a Movie Recommendation System

**Author:** Ameer Deen  
**Student ID:** 2025AG05037  
**Institution:** BITS Pilani – Work Integrated Learning Programmes (WILP)  
**Programme:** M.Tech Artificial Intelligence and Machine Learning  
**Course:** Data Management for Machine Learning  
**Course Code:** AIMLZG529  
**Semester:** S1-26  
**GitHub:** https://github.com/ameerdeen01

---

## 1. Project Overview

This project implements an end-to-end data management and machine learning pipeline for a movie recommendation system using the MovieLens 100K public dataset.

The objective is to build a clean, modular, and reproducible pipeline covering data ingestion, validation, preparation, feature engineering, model training, evaluation, and orchestration.

The project compares two recommendation approaches:

- Popularity-Based Recommendation
- User-Based Collaborative Filtering

The pipeline emphasizes data quality, structured data processing, feature management, model evaluation, and reproducibility.

## 2. Problem Statement

Recommendation systems depend on reliable and well-structured data to generate meaningful recommendations.

Raw datasets may contain missing values, duplicate records, inconsistent data types, invalid ratings, and other data quality issues.

The objective of this project is to develop an end-to-end data management pipeline that transforms raw MovieLens data into validated, cleaned, processed, and feature-enriched datasets suitable for recommendation model development.

## 3. Project Objectives

- Implement structured data ingestion.
- Identify and validate data quality issues.
- Handle missing values and duplicate records.
- Standardize and transform datasets.
- Perform time-based training and testing data splitting.
- Generate user, movie, and interaction features.
- Maintain feature metadata through a feature registry.
- Train multiple recommendation models.
- Evaluate models using MAE and RMSE.
- Compare model performance.
- Orchestrate the complete workflow using Prefect.
- Ensure reproducibility through documented execution instructions.

## 4. Dataset

**Dataset:** MovieLens 100K  
**Source:** https://grouplens.org/datasets/movielens/100k/

Dataset details:

- 943 users
- 1,682 movies
- 100,000 ratings
- Rating scale: 1 to 5

The dataset contains user information, movie metadata, and user-movie ratings.

### Dataset Setup

1. Download the MovieLens 100K dataset.
2. Extract the archive.
3. Place the `ml-100k` folder inside:

   `data/raw/`

Expected files:

```text
data/raw/ml-100k/
├── u.user
├── u.item
└── u.data
```

Raw dataset files are excluded from Git using `.gitignore`. Download the dataset separately before running the pipeline.

## 5. Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Prefect
- Git
- GitHub
- MovieLens 100K

## 6. Project Architecture

The project follows a modular architecture that separates ingestion, validation, preparation, feature engineering, training, evaluation, and orchestration.

### Project Directory Structure

```text
recommendation-system-data-pipeline/
│
├── data/
│   ├── raw/
│   │   └── ml-100k/
│   ├── processed/
│   ├── features/
│   │   └── scalers/
│   └── errors/
│
├── models/
│   ├── popularity_model.csv
│   └── collaborative_filtering_model.pkl
│
├── reports/
│   ├── validation_report.csv
│   └── model_comparison.csv
│
├── screenshots/
│
├── src/
│   ├── ingestion/
│   │   └── ingest.py
│   ├── validation/
│   │   └── validate.py
│   ├── preparation/
│   │   └── prepare.py
│   ├── features/
│   │   ├── feature_engineering.py
│   │   └── feature_registry.py
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── evaluate_cf.py
│   │   └── compare_models.py
│   └── orchestration/
│       └── pipeline.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Architecture Workflow

```text
MovieLens 100K Dataset
          |
          v
    Data Ingestion
          |
          v
    Data Validation
          |
          v
    Data Preparation
          |
          v
  Feature Engineering
          |
          v
    Feature Registry
          |
          v
    Model Training
          |
          v
   Model Evaluation
          |
          v
   Model Comparison
          |
          v
 Reports and Artifacts

 Prefect orchestrates the complete pipeline.
```

## 7. Data Pipeline Workflow

The pipeline consists of the following stages:

1. **Data Ingestion:** Loads users, movies, and ratings from the MovieLens dataset.
2. **Data Validation:** Checks missing values, duplicates, data types, invalid ratings, timestamps, and referential integrity.
3. **Data Preparation:** Handles missing values, removes duplicates, standardizes data, and creates time-based training and testing datasets.
4. **Feature Engineering:** Generates user, movie, and interaction features and applies feature scaling.
5. **Feature Registry:** Stores feature metadata for tracking and reference.
6. **Model Training:** Trains popularity-based and collaborative filtering recommendation models.
7. **Model Evaluation:** Evaluates models using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
8. **Model Comparison:** Compares model performance and identifies the better-performing approach.
9. **Pipeline Orchestration:** Uses Prefect to execute the complete workflow.

## 8. Data Quality Validation Results

The validation stage checks the quality and consistency of incoming MovieLens datasets.

### Validation Summary

| Validation Check | Result |
|---|---|
| User records | 943 |
| Movie records | 1,682 |
| Rating records | 100,000 |
| Duplicate records | No duplicates detected |
| Invalid ratings | None detected |
| Invalid timestamps | None detected |
| Referential integrity issues | None detected |

### Missing Values Identified

| Dataset | Column | Missing Values |
|---|---|---:|
| Movies | release_date | 1 |
| Movies | video_release_date | 1,682 |
| Movies | imdb_url | 3 |

Missing values are handled during the data preparation stage according to the implemented cleaning logic.

Detailed validation results are stored in:

`reports/validation_report.csv`

## 9. Data Preparation Results

The preparation stage transforms the ingested datasets into structured datasets for feature engineering and model development.

### Processing Activities

| Processing Activity | Output |
|---|---|
| Missing value handling | Cleaned datasets |
| Duplicate removal | Deduplicated datasets |
| Data standardization | Standardized datasets |
| Time-based splitting | 80% training / 20% testing |
| Training interactions | 80,000 |
| Testing interactions | 20,000 |
| Interaction matrix | 751 users × 1,616 movies |

The time-based split preserves the temporal ordering of interactions, allowing the model to be evaluated on later interactions.

## 10. Feature Engineering

Feature engineering generates meaningful attributes from user, movie, and rating interaction data.

### User Features

- User activity information.
- User-level rating statistics.

### Movie Features

- Movie popularity information.
- Movie-level attributes.

### Interaction Features

- User-movie interaction attributes.
- Time-related interaction information.

### Feature Transformation

Scikit-learn is used for feature scaling.

Fitted scalers are saved using Joblib for reuse.

Generated feature datasets and metadata are stored under:

- `data/features/user_features.csv`
- `data/features/movie_features.csv`
- `data/features/interaction_features.csv`
- `data/features/feature_metadata.json`
- `data/features/scalers/`

## 11. Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- Visual Studio Code (recommended)

### Clone the Repository

```bash
git clone https://github.com/ameerdeen01/recommendation-system-data-pipeline.git

cd recommendation-system-data-pipeline
```

### Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
```

### Activate the Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download the Dataset

Download and extract MovieLens 100K from:

https://grouplens.org/datasets/movielens/100k/

Place the extracted `ml-100k` folder under:

`data/raw/ml-100k/`

## 12. Execute the Pipeline

Run the orchestration module from the project root:

```bash
python -m src.orchestration.pipeline
```

The Prefect orchestration executes the pipeline stages sequentially.

Generated outputs are stored in:

- `data/processed/`
- `data/features/`
- `data/errors/`
- `models/`
- `reports/`

## 13. Model Evaluation Results

The models were evaluated using 20,000 test interactions.

### Evaluation Metrics

**Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted ratings. Lower values indicate better prediction accuracy.

**Root Mean Squared Error (RMSE):** Measures prediction error while penalizing larger errors more heavily. Lower values indicate better prediction accuracy.

### Evaluation Results

| Model | Test Interactions | MAE | RMSE |
|---|---:|---:|---:|
| Popularity-Based | 20,000 | 0.8439 | 1.0392 |
| Collaborative Filtering | 20,000 | 0.9202 | 1.1017 |

### Model Comparison

The Popularity-Based model achieved lower MAE and RMSE than the User-Based Collaborative Filtering model on the evaluated test set.

Therefore, the Popularity-Based model performed better according to these evaluation metrics in this experiment.

These results are specific to the current dataset, implementation, and evaluation approach.

Detailed comparison results are stored in:

`reports/model_comparison.csv`

## 14. Project Outputs

| Output | Description |
|---|---|
| Ingested datasets | Raw data loaded into structured datasets |
| Cleaned datasets | Missing value and duplicate handling |
| Standardized datasets | Consistently formatted datasets |
| Training dataset | 80,000 interactions |
| Testing dataset | 20,000 interactions |
| Interaction matrix | User-item interaction representation |
| User features | User-level engineered attributes |
| Movie features | Movie-level engineered attributes |
| Interaction features | User-movie interaction attributes |
| Feature metadata | Feature registry information |
| Popularity model | Trained popularity-based model |
| Collaborative filtering model | Trained user-based similarity model |
| Validation report | Data quality validation results |
| Model comparison | Comparative evaluation metrics |

## 15. Screenshots and Evidence

The `screenshots/` directory contains evidence of the implementation, including:

- Data ingestion.
- Missing value identification.
- Duplicate validation.
- Data type validation.
- Invalid rating and timestamp checks.
- Referential integrity validation.
- Data preparation and standardization.
- Time-based training and testing split.
- Interaction matrix generation.
- Feature engineering.
- Feature transformation.
- Feature registry.
- Model training.
- Model evaluation.
- Model comparison.
- Prefect pipeline execution.

## 16. Limitations

- The implementation uses the MovieLens 100K dataset.
- Collaborative filtering currently uses user-based similarity.
- The evaluation is based on a time-based train-test split.
- Popularity-based recommendations are not personalized.
- Model performance may vary with different datasets, users, and evaluation strategies.

## 17. Future Enhancements

- Implement item-based collaborative filtering.
- Explore matrix factorization techniques.
- Develop hybrid recommendation approaches.
- Improve model hyperparameter tuning.
- Introduce automated data quality monitoring.
- Implement experiment tracking and model versioning.
- Extend the pipeline to larger recommendation datasets.
- Explore recommendation diversity and personalization metrics.

## 18. Reproducibility

To reproduce the complete pipeline:

1. Clone the GitHub repository.
2. Create and activate a Python virtual environment.
3. Install dependencies from `requirements.txt`.
4. Download and extract the MovieLens 100K dataset.
5. Place the raw dataset in `data/raw/ml-100k/`.
6. Execute the orchestration module.
7. Review the generated datasets, models, and evaluation reports.

The raw dataset is excluded from Git and must be downloaded separately.

### Complete Execution Command

```bash
python -m src.orchestration.pipeline
```

## 19. References

1. GroupLens Research – MovieLens Datasets: https://grouplens.org/datasets/movielens/
2. Pandas Documentation: https://pandas.pydata.org/docs/
3. NumPy Documentation: https://numpy.org/doc/
4. Scikit-learn Documentation: https://scikit-learn.org/stable/
5. Prefect Documentation: https://docs.prefect.io/
6. Joblib Documentation: https://joblib.readthedocs.io/

---

**Project:** End-to-End Data Management Pipeline for a Movie Recommendation System using MovieLens 100K.

**Academic Submission:** BITS Pilani – WILP | AIMLZG529 | S1-26
