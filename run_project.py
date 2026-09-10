import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_script(script_path):
    """Run one Python script and stop if it fails."""

    full_path = PROJECT_ROOT / script_path

    print("\n" + "=" * 70)
    print(f"RUNNING: {script_path}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, str(full_path)],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        print(f"\nERROR: {script_path} failed.")
        sys.exit(result.returncode)


def main():
    print("=" * 70)
    print("       ZEPTO DATA & AI PLATFORM")
    print("=" * 70)

    # ---------------------------------------------------------
    # MODULE 1 - DATA PIPELINE
    # ---------------------------------------------------------
    print("\nMODULE 1 - DATA PIPELINE")

    run_script("data_pipeline/scraper.py")
    run_script("data_pipeline/clean_data.py")
    run_script("data_pipeline/database.py")
    run_script("data_pipeline/sql_analysis.py")

    # ---------------------------------------------------------
    # MODULE 2 - ANALYTICS & MACHINE LEARNING
    # ---------------------------------------------------------
    print("\nMODULE 2 - ANALYTICS & MACHINE LEARNING")

    run_script("analytics/01_eda.py")
    run_script("analytics/02_classification.py")
    run_script("analytics/03_regression.py")

    # ---------------------------------------------------------
    # MODULE 3 - SUPPORT ASSISTANT
    # ---------------------------------------------------------
    print("\nMODULE 3 - SUPPORT ASSISTANT")

    run_script("support_assistant/create_policies.py")
    run_script("support_assistant/embeddings.py")
    run_script("support_assistant/retrieval.py")
    run_script("support_assistant/graph.py")

    # ---------------------------------------------------------
    # START FASTAPI
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STARTING FASTAPI")
    print("=" * 70)

    api_command = [
        sys.executable,
        "-m",
        "uvicorn",
        "support_assistant.api:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]

    print("\nAPI will be available at:")
    print("http://127.0.0.1:8000/docs")
    print("\nPress CTRL+C to stop the server.\n")

    subprocess.run(
        api_command,
        cwd=PROJECT_ROOT
    )


if __name__ == "__main__":
    main()