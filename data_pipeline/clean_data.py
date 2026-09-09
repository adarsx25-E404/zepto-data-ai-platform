import pandas as pd
import re
from pathlib import Path


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

INPUT_FILE = Path("data_pipeline/outputs/books_raw.csv")
OUTPUT_FILE = Path("data_pipeline/outputs/books_cleaned.csv")


# --------------------------------------------------
# PROJECT CONSTANT
# --------------------------------------------------

GBP_TO_INR = 105.50


# --------------------------------------------------
# CLEAN PRICE
# --------------------------------------------------

def clean_price(value):
    """
    Convert a price such as £51.77 or Â£51.77
    into a numeric float such as 51.77.
    """

    if pd.isna(value):
        return None

    value = str(value)

    # Find the numeric part of the price
    match = re.search(r"\d+(?:\.\d+)?", value)

    if match:
        return float(match.group())

    return None


# --------------------------------------------------
# CLEAN RATING
# --------------------------------------------------

def clean_rating(value):
    """
    Convert rating text or numeric rating into
    an integer from 1 to 5.
    """

    if pd.isna(value):
        return None

    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }

    value_text = str(value).strip()

    # If rating is already numeric
    try:
        number = int(float(value_text))

        if 1 <= number <= 5:
            return number

    except ValueError:
        pass

    # If rating is text
    if value_text in rating_map:
        return rating_map[value_text]

    return None


# --------------------------------------------------
# CLEAN AVAILABILITY
# --------------------------------------------------

def clean_stock(value):
    """
    Convert availability text into True/False.
    """

    if pd.isna(value):
        return False

    value = str(value).strip().lower()

    return value.startswith("in stock")


# --------------------------------------------------
# MAIN CLEANING FUNCTION
# --------------------------------------------------

def main():

    print("=" * 60)
    print("BOOK DATA CLEANING")
    print("=" * 60)

    # Check that raw file exists
    if not INPUT_FILE.exists():

        print(
            f"\nERROR: Raw file not found:"
            f"\n{INPUT_FILE}"
        )

        print(
            "\nRun the scraper first:"
        )

        print(
            "python data_pipeline\\scraper.py"
        )

        return

    # --------------------------------------------------
    # LOAD RAW DATA
    # --------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print(
        f"\nRaw rows loaded: {len(df)}"
    )

    print(
        f"Raw columns: {list(df.columns)}"
    )

    # --------------------------------------------------
    # CLEAN PRICE
    # --------------------------------------------------

    df["price_gbp"] = df["price_gbp"].apply(
        clean_price
    )

    # --------------------------------------------------
    # CLEAN RATING
    # --------------------------------------------------

    df["rating"] = df["rating"].apply(
        clean_rating
    )

    # --------------------------------------------------
    # CLEAN AVAILABILITY
    # --------------------------------------------------

    df["in_stock"] = df["availability"].apply(
        clean_stock
    )

    # --------------------------------------------------
    # HANDLE NUMERIC PARSING FAILURES
    # --------------------------------------------------

    price_missing = df["price_gbp"].isna().sum()
    rating_missing = df["rating"].isna().sum()

    print(
        f"\nPrice parsing failures: {price_missing}"
    )

    print(
        f"Rating parsing failures: {rating_missing}"
    )

    # Median imputation for numeric fields
    if price_missing > 0:

        median_price = df["price_gbp"].median()

        df["price_gbp"] = df["price_gbp"].fillna(
            median_price
        )

        print(
            f"Missing prices replaced with median: "
            f"{median_price:.2f}"
        )

    if rating_missing > 0:

        median_rating = df["rating"].median()

        df["rating"] = df["rating"].fillna(
            round(median_rating)
        )

        print(
            f"Missing ratings replaced with median: "
            f"{round(median_rating)}"
        )

    # --------------------------------------------------
    # CONVERT GBP TO INR
    # --------------------------------------------------

    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    ).round(2)

    # --------------------------------------------------
    # KEEP REQUIRED FINAL COLUMNS
    # --------------------------------------------------

    df = df[
        [
            "title",
            "category",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock"
        ]
    ]

    # --------------------------------------------------
    # SET CORRECT DATA TYPES
    # --------------------------------------------------

    df["price_gbp"] = df["price_gbp"].astype(float)

    df["price_inr"] = df["price_inr"].astype(float)

    df["rating"] = df["rating"].astype(int)

    df["in_stock"] = df["in_stock"].astype(bool)

    # --------------------------------------------------
    # SAVE CLEANED DATA
    # --------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)

    print(
        f"\nCleaned rows: {len(df)}"
    )

    print(
        f"Categories: {df['category'].nunique()}"
    )

    print(
        f"\nGBP → INR rate used: {GBP_TO_INR}"
    )

    print(
        "\nFinal columns:"
    )

    print(
        list(df.columns)
    )

    print(
        "\nData types:"
    )

    print(
        df.dtypes
    )

    print(
        "\nFirst 5 cleaned rows:"
    )

    print(
        df.head().to_string(index=False)
    )

    print(
        f"\nSaved to:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()