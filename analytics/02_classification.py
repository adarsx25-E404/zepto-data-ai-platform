"""
Module 2 - Classification

This script:
1. Loads the cleaned Titanic dataset.
2. Creates train/test datasets.
3. Builds Logistic Regression, Decision Tree and Random Forest models.
4. Evaluates accuracy, precision, recall and F1.
5. Creates confusion matrices.
6. Calculates ROC/AUC.
7. Handles class imbalance using class_weight and SMOTE.
8. Tunes Random Forest using GridSearchCV.
9. Saves metrics, charts and the best model.
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

DATA_FILE = "analytics/titanic.csv"
OUTPUT_DIR = "analytics/outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("=" * 70)
print("TITANIC CLASSIFICATION")
print("=" * 70)

print("\nLoading cleaned Titanic dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Rows loaded: {len(df)}")
print(f"Columns loaded: {len(df.columns)}")

print("\nColumns:")
print(list(df.columns))


# ---------------------------------------------------------
# SELECT FEATURES AND TARGET
# ---------------------------------------------------------

TARGET = "survived"

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
]

X = df[features].copy()
y = df[TARGET].copy()


print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget proportions:")
print(y.value_counts(normalize=True))


# ---------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTrain rows:", len(X_train))
print("Test rows:", len(X_test))


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]

categorical_features = [
    "sex",
    "embarked",
]


numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features,
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features,
        ),
    ]
)


# ---------------------------------------------------------
# EVALUATION FUNCTION
# ---------------------------------------------------------

results = []


def evaluate_model(model, model_name):
    """
    Train and evaluate a classification model.
    """

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    probabilities = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")

    results.append(
        {
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": auc,
        }
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Did Not Survive", "Survived"],
    ).plot(ax=ax)

    ax.set_title(
        f"Confusion Matrix - {model_name}"
    )

    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    plt.savefig(
        f"{OUTPUT_DIR}/{filename}_confusion_matrix.png",
        dpi=150,
    )

    plt.close()

    return model


# ---------------------------------------------------------
# BASELINE MODELS
# ---------------------------------------------------------

logistic_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)


decision_tree_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            DecisionTreeClassifier(
                random_state=42,
            ),
        ),
    ]
)


random_forest_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


logistic_model = evaluate_model(
    logistic_pipeline,
    "Logistic Regression",
)


decision_tree_model = evaluate_model(
    decision_tree_pipeline,
    "Decision Tree",
)


random_forest_model = evaluate_model(
    random_forest_pipeline,
    "Random Forest",
)


# ---------------------------------------------------------
# ROC CURVES
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ROC CURVES")
print("=" * 70)


plt.figure(figsize=(8, 6))


models_for_roc = [
    (
        logistic_model,
        "Logistic Regression",
    ),
    (
        decision_tree_model,
        "Decision Tree",
    ),
    (
        random_forest_model,
        "Random Forest",
    ),
]


for model, name in models_for_roc:

    probabilities = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities,
    )

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.3f})",
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Guess",
)


plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curves - Titanic Classification")

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/classification_roc_curves.png",
    dpi=150,
)

plt.close()


# ---------------------------------------------------------
# CLASS IMBALANCE - BALANCED CLASS WEIGHTS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CLASS IMBALANCE HANDLING")
print("=" * 70)

print("\nTraining Logistic Regression with class_weight='balanced'...")


balanced_logistic = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)


balanced_logistic = evaluate_model(
    balanced_logistic,
    "Logistic Regression Balanced",
)


# ---------------------------------------------------------
# SMOTE
# ---------------------------------------------------------

print("\nTraining Logistic Regression with SMOTE...")


smote_logistic = ImbPipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "smote",
            SMOTE(
                random_state=42,
            ),
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)


smote_logistic = evaluate_model(
    smote_logistic,
    "Logistic Regression SMOTE",
)


# ---------------------------------------------------------
# RANDOM FOREST GRID SEARCH
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("RANDOM FOREST GRID SEARCH")
print("=" * 70)


rf_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


param_grid = {
    "model__n_estimators": [
        100,
        200,
    ],
    "model__max_depth": [
        None,
        5,
        10,
    ],
    "model__min_samples_split": [
        2,
        5,
    ],
}


grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)


grid_search.fit(
    X_train,
    y_train,
)


print("\nBest parameters:")
print(grid_search.best_params_)

print(
    f"\nBest cross-validation F1: "
    f"{grid_search.best_score_:.4f}"
)


best_rf = grid_search.best_estimator_


best_rf = evaluate_model(
    best_rf,
    "Random Forest Tuned",
)


# ---------------------------------------------------------
# RANDOM FOREST OOB SCORE
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("RANDOM FOREST OOB SCORE")
print("=" * 70)


oob_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                oob_score=True,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


oob_pipeline.fit(
    X_train,
    y_train,
)


oob_score = (
    oob_pipeline
    .named_steps["model"]
    .oob_score_
)


print(f"OOB Score: {oob_score:.4f}")


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

results_df = pd.DataFrame(results)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


results_df.to_csv(
    f"{OUTPUT_DIR}/classification_results.csv",
    index=False,
)


# ---------------------------------------------------------
# SAVE BEST MODEL
# ---------------------------------------------------------

joblib.dump(
    best_rf,
    f"{OUTPUT_DIR}/best_random_forest_pipeline.joblib",
)


# ---------------------------------------------------------
# SAVE SUMMARY
# ---------------------------------------------------------

with open(
    f"{OUTPUT_DIR}/classification_summary.txt",
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "Titanic Classification Summary\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )

    file.write(
        results_df.to_string(
            index=False
        )
    )

    file.write(
        "\n\nBest Random Forest Parameters:\n"
    )

    file.write(
        str(grid_search.best_params_)
    )

    file.write(
        "\n\nBest Cross-Validation F1:\n"
    )

    file.write(
        f"{grid_search.best_score_:.4f}"
    )

    file.write(
        "\n\nRandom Forest OOB Score:\n"
    )

    file.write(
        f"{oob_score:.4f}"
    )


# ---------------------------------------------------------
# FINISHED
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CLASSIFICATION COMPLETE")
print("=" * 70)

print(
    f"\nResults saved to: {OUTPUT_DIR}"
)

print(
    "\nBest model saved to:"
)

print(
    f"{OUTPUT_DIR}/best_random_forest_pipeline.joblib"
)

print("\n")