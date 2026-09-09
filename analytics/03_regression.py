"""
Module 2 - Regression

This script:
1. Loads the cleaned Titanic dataset.
2. Predicts fare using the other available features.
3. Uses a preprocessing pipeline for numeric and categorical columns.
4. Trains a multivariate Linear Regression model.
5. Calculates MAE, RMSE, R2 and Adjusted R2.
6. Creates and saves a residual plot.
7. Checks for possible heteroscedasticity.
8. Saves the complete preprocessing + model pipeline.
9. Reloads the saved pipeline and verifies predictions.
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib

# Prevent Windows GUI/backend problems.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "titanic.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


MODEL_FILE = os.path.join(
    OUTPUT_DIR,
    "fare_regression_pipeline.joblib"
)

RESULT_FILE = os.path.join(
    OUTPUT_DIR,
    "regression_results.txt"
)

RESIDUAL_PLOT = os.path.join(
    OUTPUT_DIR,
    "fare_regression_residuals.png"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def calculate_adjusted_r2(r2, n, p):
    """
    Calculate Adjusted R-squared.

    r2 = R-squared
    n  = number of observations
    p  = number of predictor variables
    """

    if n <= p + 1:
        return np.nan

    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TITANIC FARE REGRESSION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    print("\nLoading cleaned Titanic dataset...")

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Could not find cleaned dataset: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    print(f"Rows loaded: {len(df)}")
    print(f"Columns loaded: {len(df.columns)}")

    print("\nColumns:")
    print(df.columns.tolist())

    # --------------------------------------------------------
    # 2. TARGET AND FEATURES
    # --------------------------------------------------------

    target = "fare"

    if target not in df.columns:
        raise ValueError(
            "The 'fare' column was not found in titanic.csv"
        )

    # Remove target from features.
    X = df.drop(columns=[target])
    y = df[target]

    print("\nTarget variable:")
    print("fare")

    print("\nFeature columns:")
    print(X.columns.tolist())

    # --------------------------------------------------------
    # 3. TRAIN / TEST SPLIT
    # --------------------------------------------------------

    print("\nCreating train/test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # --------------------------------------------------------
    # 4. IDENTIFY COLUMN TYPES
    # --------------------------------------------------------

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("\nNumeric features:")
    print(numeric_features)

    print("\nCategorical features:")
    print(categorical_features)

    # --------------------------------------------------------
    # 5. NUMERIC PREPROCESSING
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # --------------------------------------------------------
    # 6. CATEGORICAL PREPROCESSING
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # --------------------------------------------------------
    # 7. COLUMN TRANSFORMER
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    # --------------------------------------------------------
    # 8. COMPLETE REGRESSION PIPELINE
    # --------------------------------------------------------

    full_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "regressor",
                LinearRegression()
            )
        ]
    )

    print("\nTraining multivariate Linear Regression...")

    full_pipeline.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # --------------------------------------------------------
    # 9. PREDICTIONS
    # --------------------------------------------------------

    y_pred = full_pipeline.predict(X_test)

    # --------------------------------------------------------
    # 10. METRICS
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    # Get number of transformed predictors.
    transformed_X_train = full_pipeline.named_steps[
        "preprocessor"
    ].transform(X_train)

    p = transformed_X_train.shape[1]
    n = len(y_test)

    adjusted_r2 = calculate_adjusted_r2(
        r2,
        n,
        p
    )

    print("\n" + "=" * 70)
    print("REGRESSION RESULTS")
    print("=" * 70)

    print(f"MAE          : {mae:.4f}")
    print(f"RMSE         : {rmse:.4f}")
    print(f"R-squared    : {r2:.4f}")
    print(f"Adjusted R2  : {adjusted_r2:.4f}")

    print(f"\nObservations used for evaluation: {n}")
    print(f"Predictors after preprocessing: {p}")

    # --------------------------------------------------------
    # 11. RESIDUALS
    # --------------------------------------------------------

    residuals = y_test.to_numpy() - y_pred

    # --------------------------------------------------------
    # 12. RESIDUAL PLOT
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.7
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel("Predicted Fare")
    plt.ylabel("Residuals")
    plt.title("Titanic Fare Regression - Residual Plot")

    plt.tight_layout()

    plt.savefig(
        RESIDUAL_PLOT,
        dpi=150
    )

    plt.close()

    print(f"\nResidual plot saved to:")
    print(RESIDUAL_PLOT)

    # --------------------------------------------------------
    # 13. SIMPLE HETEROSCEDASTICITY INTERPRETATION
    # --------------------------------------------------------

    # Divide predictions into three groups and compare
    # residual spread.

    residual_df = pd.DataFrame(
        {
            "prediction": y_pred,
            "residual": residuals
        }
    )

    residual_df["prediction_group"] = pd.qcut(
        residual_df["prediction"],
        q=3,
        duplicates="drop"
    )

    group_std = residual_df.groupby(
        "prediction_group",
        observed=True
    )["residual"].std()

    print("\nResidual spread by prediction group:")
    print(group_std)

    if len(group_std) >= 2:

        smallest_std = group_std.min()
        largest_std = group_std.max()

        if smallest_std == 0:
            spread_ratio = np.inf
        else:
            spread_ratio = largest_std / smallest_std

        if spread_ratio > 2:
            hetero_conclusion = (
                "The residual spread changes substantially across "
                "prediction ranges, suggesting possible "
                "heteroscedasticity."
            )
        else:
            hetero_conclusion = (
                "The residual spread is reasonably similar across "
                "prediction ranges, so there is no strong visual "
                "evidence of heteroscedasticity."
            )

    else:

        hetero_conclusion = (
            "There were not enough prediction groups to assess "
            "heteroscedasticity reliably."
        )

    print("\nHETEROSCEDASTICITY CONCLUSION:")
    print(hetero_conclusion)

    # --------------------------------------------------------
    # 14. SAVE COMPLETE PIPELINE
    # --------------------------------------------------------

    joblib.dump(
        full_pipeline,
        MODEL_FILE
    )

    print("\nComplete regression pipeline saved to:")
    print(MODEL_FILE)

    # --------------------------------------------------------
    # 15. RELOAD PIPELINE
    # --------------------------------------------------------

    print("\nReloading saved pipeline...")

    loaded_pipeline = joblib.load(
        MODEL_FILE
    )

    reloaded_predictions = loaded_pipeline.predict(
        X_test.head(5)
    )

    print("Reload successful.")

    print("\nPredictions from reloaded pipeline:")

    for i, prediction in enumerate(
        reloaded_predictions,
        start=1
    ):
        print(
            f"Row {i}: predicted fare = {prediction:.2f}"
        )

    # --------------------------------------------------------
    # 16. SAVE TEXT RESULTS
    # --------------------------------------------------------

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "TITANIC FARE REGRESSION RESULTS\n"
        )

        file.write(
            "=" * 50 + "\n\n"
        )

        file.write(
            f"Training rows: {len(X_train)}\n"
        )

        file.write(
            f"Testing rows: {len(X_test)}\n\n"
        )

        file.write(
            f"MAE: {mae:.6f}\n"
        )

        file.write(
            f"RMSE: {rmse:.6f}\n"
        )

        file.write(
            f"R2: {r2:.6f}\n"
        )

        file.write(
            f"Adjusted R2: {adjusted_r2:.6f}\n\n"
        )

        file.write(
            f"Predictors after preprocessing: {p}\n\n"
        )

        file.write(
            "Heteroscedasticity conclusion:\n"
        )

        file.write(
            hetero_conclusion + "\n"
        )

    # --------------------------------------------------------
    # 17. FINAL MESSAGE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("REGRESSION COMPLETE")
    print("=" * 70)

    print("\nFiles created:")

    print(
        f"- {RESULT_FILE}"
    )

    print(
        f"- {RESIDUAL_PLOT}"
    )

    print(
        f"- {MODEL_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()