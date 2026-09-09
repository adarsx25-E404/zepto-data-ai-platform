import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = "analytics/outputs"
RAW_CSV = "analytics/titanic.csv"
CLEANED_CSV = "analytics/titanic.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD TITANIC DATASET
# ============================================================

print("=" * 70)
print("MODULE 2 - TITANIC ANALYTICS")
print("=" * 70)

print("\n1. Loading Titanic dataset...")

df = sns.load_dataset("titanic")

print("Dataset loaded successfully.")


# ============================================================
# SAVE OFFLINE FALLBACK IMMEDIATELY
# ============================================================

df.to_csv(RAW_CSV, index=False)

print(f"Offline dataset saved to: {RAW_CSV}")


# ============================================================
# 2. PROFILE THE DATA
# ============================================================

print("\n" + "=" * 70)
print("2. DATASET PROFILE")
print("=" * 70)

print("\nDATASET SHAPE:")
print(df.shape)

print("\nDATASET INFO:")
df.info()

print("\nDESCRIPTIVE STATISTICS:")
print(df.describe(include="all"))


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()
missing_percent = (df.isnull().mean() * 100).round(2)

missing_report = pd.DataFrame(
    {
        "missing_count": missing,
        "missing_percent": missing_percent
    }
)

missing_report = missing_report[
    missing_report["missing_count"] > 0
]

print(missing_report)

missing_report.to_csv(
    f"{OUTPUT_DIR}/missing_values_report.csv"
)


# ============================================================
# 3. MISSING VALUE HANDLING
# ============================================================

print("\n" + "=" * 70)
print("3. MISSING VALUE HANDLING")
print("=" * 70)

print(
    "\nRule used:"
    "\n- Under 5% missing: drop affected rows."
    "\n- 5% to 30% missing: impute."
    "\n- Above 30% missing: drop column or treat missing as a category."
)

# We make decisions using the measured percentages.

for column in missing_report.index:

    percent = missing_percent[column]

    print(
        f"\nColumn: {column}"
        f"\nMissing percentage: {percent:.2f}%"
    )

    if percent < 5:
        print("Decision: Drop rows with missing values.")

    elif percent <= 30:
        print("Decision: Impute missing values.")

    else:
        print(
            "Decision: High missingness. "
            "Column will be dropped if not needed."
        )


# AGE
# Between 5% and 30% missing -> median imputation
if "age" in df.columns and df["age"].isnull().any():

    age_missing_percent = df["age"].isnull().mean() * 100

    if 5 <= age_missing_percent <= 30:

        df["age"] = df["age"].fillna(
            df["age"].median()
        )

        print(
            f"\nAge imputed using median."
            f" Missing rate was {age_missing_percent:.2f}%."
        )


# EMBARKED
# Usually below 5%, so rows are dropped.
if "embarked" in df.columns and df["embarked"].isnull().any():

    embarked_missing_percent = (
        df["embarked"].isnull().mean() * 100
    )

    if embarked_missing_percent < 5:

        df = df.dropna(
            subset=["embarked"]
        )

        print(
            f"Rows with missing embarked values dropped."
            f" Missing rate was {embarked_missing_percent:.2f}%."
        )


# DECK
# Very high missingness, so drop it.
if "deck" in df.columns:

    deck_missing_percent = (
        df["deck"].isnull().mean() * 100
    )

    if deck_missing_percent > 30:

        df = df.drop(
            columns=["deck"]
        )

        print(
            f"Deck column dropped."
            f" Missing rate was {deck_missing_percent:.2f}%."
        )


# EMBARK_TOWN
# Redundant with embarked, so drop it.
if "embark_town" in df.columns:

    df = df.drop(
        columns=["embark_town"]
    )

    print(
        "embark_town dropped because it duplicates "
        "the information in embarked."
    )


# ============================================================
# SAVE CLEANED DATASET
# ============================================================

df.to_csv(
    CLEANED_CSV,
    index=False
)

print(
    f"\nCleaned dataset saved to: {CLEANED_CSV}"
)

print(
    f"Cleaned dataset shape: {df.shape}"
)


# ============================================================
# 4. UNIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("4. UNIVARIATE ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# AGE HISTOGRAM
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["age"],
    kde=True
)

plt.title("Distribution of Passenger Age")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/age_histogram.png"
)

plt.close()


# ------------------------------------------------------------
# AGE BOXPLOT
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x=df["age"]
)

plt.title("Box Plot of Passenger Age")
plt.xlabel("Age")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/age_boxplot.png"
)

plt.close()


# ------------------------------------------------------------
# FARE HISTOGRAM
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["fare"],
    kde=True
)

plt.title("Distribution of Passenger Fare")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/fare_histogram.png"
)

plt.close()


# ------------------------------------------------------------
# FARE BOXPLOT
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x=df["fare"]
)

plt.title("Box Plot of Passenger Fare")
plt.xlabel("Fare")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/fare_boxplot.png"
)

plt.close()


# ============================================================
# IQR OUTLIER COUNTS
# ============================================================

def count_iqr_outliers(series):

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = series[
        (series < lower) |
        (series > upper)
    ]

    return len(outliers), lower, upper


age_outliers, age_lower, age_upper = (
    count_iqr_outliers(df["age"])
)

fare_outliers, fare_lower, fare_upper = (
    count_iqr_outliers(df["fare"])
)


print("\nIQR OUTLIER RESULTS:")

print(
    f"Age outliers: {age_outliers}"
)

print(
    f"Age lower boundary: {age_lower:.2f}"
)

print(
    f"Age upper boundary: {age_upper:.2f}"
)

print(
    f"Fare outliers: {fare_outliers}"
)

print(
    f"Fare lower boundary: {fare_lower:.2f}"
)

print(
    f"Fare upper boundary: {fare_upper:.2f}"
)


# ============================================================
# FARE SKEWNESS
# ============================================================

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode().iloc[0]

print("\nFARE SKEWNESS CHECK:")

print(
    f"Mean: {fare_mean:.2f}"
)

print(
    f"Median: {fare_median:.2f}"
)

print(
    f"Mode: {fare_mode:.2f}"
)

if fare_mean > fare_median:

    print(
        "Conclusion: Fare is positively/right skewed "
        "because the mean is greater than the median."
    )

else:

    print(
        "Conclusion: Fare does not show strong "
        "positive skew based on mean vs median."
    )


# ============================================================
# 5. BIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("5. BIVARIATE ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# SURVIVAL RATE BY SEX
# ------------------------------------------------------------

sex_survival = (
    df.groupby("sex")["survived"]
    .mean()
    .round(4)
)

print("\nSURVIVAL RATE BY SEX:")
print(sex_survival)


# ------------------------------------------------------------
# SURVIVAL RATE BY CLASS
# ------------------------------------------------------------

class_survival = (
    df.groupby("pclass")["survived"]
    .mean()
    .round(4)
)

print("\nSURVIVAL RATE BY PASSENGER CLASS:")
print(class_survival)


# ------------------------------------------------------------
# SURVIVAL RATE BY SEX + CLASS
# ------------------------------------------------------------

sex_class_survival = (
    df.groupby(
        ["sex", "pclass"]
    )["survived"]
    .mean()
    .round(4)
)

print("\nSURVIVAL RATE BY SEX AND CLASS:")
print(sex_class_survival)


# Save results
sex_survival.to_csv(
    f"{OUTPUT_DIR}/survival_by_sex.csv"
)

class_survival.to_csv(
    f"{OUTPUT_DIR}/survival_by_class.csv"
)

sex_class_survival.to_csv(
    f"{OUTPUT_DIR}/survival_by_sex_class.csv"
)


# ============================================================
# CORRELATION MATRIX
# ============================================================

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = (
    df[correlation_columns]
    .corr()
)

print(
    "\nCORRELATION MATRIX:"
)

print(correlation_matrix)


plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title(
    "Titanic Numeric Feature Correlation Matrix"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/correlation_heatmap.png"
)

plt.close()


# ------------------------------------------------------------
# FIND TWO STRONGEST CORRELATIONS
# ------------------------------------------------------------

pairs = []

for i in range(
    len(correlation_columns)
):

    for j in range(
        i + 1,
        len(correlation_columns)
    ):

        col1 = correlation_columns[i]
        col2 = correlation_columns[j]

        value = correlation_matrix.loc[
            col1,
            col2
        ]

        pairs.append(
            (
                col1,
                col2,
                value,
                abs(value)
            )
        )


pairs = sorted(
    pairs,
    key=lambda x: x[3],
    reverse=True
)


print(
    "\nTWO STRONGEST CORRELATIONS:"
)

for pair in pairs[:2]:

    print(
        f"{pair[0]} <-> {pair[1]}: "
        f"{pair[2]:.4f}"
    )


# ============================================================
# 6. MULTIVARIATE DATA STORY CHARTS
# ============================================================

print("\n" + "=" * 70)
print("6. MULTIVARIATE DATA STORY")
print("=" * 70)


# ------------------------------------------------------------
# CHART 1: Survival by Sex
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="sex",
    y="survived"
)

plt.title(
    "Survival Rate by Sex"
)

plt.ylabel(
    "Survival Rate"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/story_01_survival_by_sex.png"
)

plt.close()


# ------------------------------------------------------------
# CHART 2: Survival by Passenger Class
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="pclass",
    y="survived"
)

plt.title(
    "Survival Rate by Passenger Class"
)

plt.ylabel(
    "Survival Rate"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/story_02_survival_by_class.png"
)

plt.close()


# ------------------------------------------------------------
# CHART 3: Sex + Class
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title(
    "Survival Rate by Sex and Passenger Class"
)

plt.ylabel(
    "Survival Rate"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/story_03_sex_class.png"
)

plt.close()


# ------------------------------------------------------------
# CHART 4: Age vs Fare
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived",
    alpha=0.7
)

plt.title(
    "Age vs Fare by Survival Status"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/story_04_age_fare_survival.png"
)

plt.close()


# ------------------------------------------------------------
# CHART 5: Class + Fare + Survival
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="pclass",
    y="fare",
    hue="survived"
)

plt.title(
    "Fare Distribution by Class and Survival"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/story_05_class_fare_survival.png"
)

plt.close()


# ============================================================
# 7. EXPLORATORY STANDARDIZATION CHECK
# ============================================================

print("\n" + "=" * 70)
print("7. STANDARDIZATION CHECK")
print("=" * 70)


scaler = StandardScaler()

standardized = scaler.fit_transform(
    df[["age", "fare"]]
)

standardized_df = pd.DataFrame(
    standardized,
    columns=["age_standardized", "fare_standardized"]
)


print("\nBEFORE STANDARDIZATION:")

print(
    df[["age", "fare"]]
    .agg(["mean", "std"])
)


print("\nAFTER STANDARDIZATION:")

print(
    standardized_df
    .agg(["mean", "std"])
)


standardized_df.to_csv(
    f"{OUTPUT_DIR}/standardization_check.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print(
    f"Final cleaned rows: {len(df)}"
)

print(
    f"Final cleaned columns: {len(df.columns)}"
)

print(
    f"Cleaned dataset: {CLEANED_CSV}"
)

print(
    f"Charts and reports: {OUTPUT_DIR}"
)

print("=" * 70)