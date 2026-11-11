
import subprocess
import sys
from pathlib import Path

from prefect import flow, task


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@task(name="Run Pipeline Stage")
def run_stage(stage_name, script_path):
    print(f"\n{'=' * 50}")
    print(f"STARTING: {stage_name}")
    print(f"{'=' * 50}")

    full_path = PROJECT_ROOT / script_path

    subprocess.run(
        [sys.executable, str(full_path)],
        cwd=PROJECT_ROOT,
        check=True
    )

    print(f"\nCOMPLETED: {stage_name}")


@flow(name="Recommendation System Data Pipeline")
def recommendation_pipeline():

    stages = [
        ("Data Ingestion", "src/ingestion/ingest.py"),
        ("Data Validation", "src/validation/validate.py"),
        ("Data Preparation", "src/preparation/prepare.py"),
        ("Feature Engineering", "src/features/feature_engineering.py"),
        ("Feature Registry", "src/features/feature_registry.py"),
        ("Model Training", "src/training/train.py"),
        ("Popularity Evaluation", "src/training/evaluate.py"),
        ("Collaborative Filtering Evaluation", "src/training/evaluate_cf.py"),
        ("Model Comparison", "src/training/compare_models.py")
    ]

    for stage_name, script_path in stages:
        run_stage(stage_name, script_path)

    print("\nRECOMMENDATION PIPELINE COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    recommendation_pipeline()
