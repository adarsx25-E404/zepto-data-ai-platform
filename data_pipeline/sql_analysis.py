import sqlite3
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

DATABASE_FILE = Path(
    "data_pipeline/outputs/books.db"
)

CLEANED_FILE = Path(
    "data_pipeline/outputs/books_cleaned.csv"
)

OUTPUT_FILE = Path(
    "data_pipeline/outputs/sql_results.txt"
)


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():

    print("=" * 70)
    print("SQL AND PANDAS ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------

    if not DATABASE_FILE.exists():

        print(
            f"\nERROR: Database not found:"
            f"\n{DATABASE_FILE}"
        )

        print(
            "\nRun this first:"
        )

        print(
            "python data_pipeline\\database.py"
        )

        return

    if not CLEANED_FILE.exists():

        print(
            f"\nERROR: Cleaned CSV not found:"
            f"\n{CLEANED_FILE}"
        )

        return

    # --------------------------------------------------
    # CONNECT TO DATABASE
    # --------------------------------------------------

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    # --------------------------------------------------
    # LOAD CLEANED DATA INTO PANDAS
    # --------------------------------------------------

    df = pd.read_csv(
        CLEANED_FILE
    )

    # --------------------------------------------------
    # QUERY 1
    # SELECT + WHERE
    # --------------------------------------------------

    query_1 = """
        SELECT
            title,
            price_inr,
            rating,
            in_stock
        FROM books
        WHERE rating >= 4
    """

    result_1 = pd.read_sql(
        query_1,
        connection
    )

    print("\n" + "=" * 70)
    print("QUERY 1 — SELECT + WHERE")
    print("=" * 70)

    print(query_1.strip())

    print("\nOUTPUT:")

    print(
        result_1.to_string(index=False)
    )

    # --------------------------------------------------
    # QUERY 2
    # ORDER BY
    # --------------------------------------------------

    query_2 = """
        SELECT
            title,
            price_inr
        FROM books
        ORDER BY price_inr DESC
    """

    result_2 = pd.read_sql(
        query_2,
        connection
    )

    print("\n" + "=" * 70)
    print("QUERY 2 — ORDER BY")
    print("=" * 70)

    print(query_2.strip())

    print("\nOUTPUT:")

    print(
        result_2.to_string(index=False)
    )

    # --------------------------------------------------
    # QUERY 3
    # LIMIT
    # --------------------------------------------------

    query_3 = """
        SELECT
            title,
            price_inr,
            rating
        FROM books
        ORDER BY rating DESC, price_inr DESC
        LIMIT 10
    """

    result_3 = pd.read_sql(
        query_3,
        connection
    )

    print("\n" + "=" * 70)
    print("QUERY 3 — LIMIT")
    print("=" * 70)

    print(query_3.strip())

    print("\nOUTPUT:")

    print(
        result_3.to_string(index=False)
    )

    # --------------------------------------------------
    # QUERY 4
    # DISTINCT
    # --------------------------------------------------

    query_4 = """
        SELECT DISTINCT
            category_name
        FROM categories
        ORDER BY category_name
    """

    result_4 = pd.read_sql(
        query_4,
        connection
    )

    print("\n" + "=" * 70)
    print("QUERY 4 — DISTINCT")
    print("=" * 70)

    print(query_4.strip())

    print("\nOUTPUT:")

    print(
        result_4.to_string(index=False)
    )

    # --------------------------------------------------
    # QUERY 5
    # BETWEEN
    # --------------------------------------------------

    query_5 = """
        SELECT
            title,
            price_gbp,
            price_inr,
            rating
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp
    """

    result_5 = pd.read_sql(
        query_5,
        connection
    )

    print("\n" + "=" * 70)
    print("QUERY 5 — BETWEEN")
    print("=" * 70)

    print(query_5.strip())

    print("\nOUTPUT:")

    print(
        result_5.to_string(index=False)
    )

    # --------------------------------------------------
    # QUERY 6
    # JOIN
    # --------------------------------------------------

    join_query = """
        SELECT
            books.book_id,
            books.title,
            categories.category_name,
            books.price_gbp,
            books.price_inr,
            books.rating,
            books.in_stock
        FROM books
        JOIN categories
            ON books.category_id =
               categories.category_id
        ORDER BY
            categories.category_name,
            books.rating DESC,
            books.price_inr DESC
        LIMIT 10
    """

    join_result_sql = pd.read_sql(
        join_query,
        connection
    )

    join_result_sql["in_stock"] = (
    join_result_sql["in_stock"].astype(bool)
)

    print("\n" + "=" * 70)
    print("QUERY 6 — JOIN")
    print("=" * 70)

    print(join_query.strip())

    print("\nOUTPUT FROM pd.read_sql():")

    print(
        join_result_sql.to_string(index=False)
    )

    # --------------------------------------------------
    # PANDAS MERGE
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("PANDAS MERGE — REPRODUCING JOIN")
    print("=" * 70)

    # Create category table in pandas
    categories_df = (
        df[["category"]]
        .drop_duplicates()
        .sort_values("category")
        .reset_index(drop=True)
    )

    categories_df["category_id"] = (
        categories_df.index + 1
    )

    categories_df = categories_df[
        [
            "category_id",
            "category"
        ]
    ]

    # Create books dataframe with book_id
    books_df = df.copy()

    books_df["book_id"] = (
        books_df.index + 1
    )

    books_df = books_df[
        [
            "book_id",
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category"
        ]
    ]

    # Reproduce the JOIN using pandas merge
    join_result_pandas = books_df.merge(
        categories_df,
        on="category",
        how="inner"
    )

    # Select the same columns as SQL
    join_result_pandas = join_result_pandas[
        [
            "book_id",
            "title",
            "category",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock"
        ]
    ]

    # Rename category to match SQL output
    join_result_pandas = join_result_pandas.rename(
        columns={
            "category": "category_name"
        }
    )

    # Apply same ordering and LIMIT
    join_result_pandas = (
        join_result_pandas
        .sort_values(
            [
                "category_name",
                "rating",
                "price_inr"
            ],
            ascending=[
                True,
                False,
                False
            ]
        )
        .head(10)
        .reset_index(drop=True)
    )

    print("\nOUTPUT FROM pd.merge():")

    print(
        join_result_pandas.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # COMPARE SQL JOIN AND PANDAS MERGE
    # --------------------------------------------------

    sql_compare = join_result_sql.copy()

    pandas_compare = join_result_pandas.copy()

    # Make column order identical
    pandas_compare = pandas_compare[
        sql_compare.columns
    ]

    # Reset indexes
    sql_compare = (
        sql_compare
        .reset_index(drop=True)
    )

    pandas_compare = (
        pandas_compare
        .reset_index(drop=True)
    )

    # Compare values
    are_equal = sql_compare.equals(
        pandas_compare
    )

    print("\n" + "=" * 70)
    print("JOIN EQUIVALENCE CHECK")
    print("=" * 70)

    print(
        f"pd.read_sql() and pd.merge() match: "
        f"{are_equal}"
    )

    # --------------------------------------------------
    # SAVE RESULTS TO TEXT FILE
    # --------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "SQL AND PANDAS ANALYSIS RESULTS\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            "QUERY 1 — SELECT + WHERE\n"
        )

        file.write(
            query_1.strip() + "\n\n"
        )

        file.write(
            result_1.to_string(index=False)
        )

        file.write(
            "\n\n" + "=" * 70 + "\n\n"
        )

        file.write(
            "QUERY 2 — ORDER BY\n"
        )

        file.write(
            query_2.strip() + "\n\n"
        )

        file.write(
            result_2.to_string(index=False)
        )

        file.write(
            "\n\n" + "=" * 70 + "\n\n"
        )

        file.write(
            "QUERY 3 — LIMIT\n"
        )

        file.write(
            query_3.strip() + "\n\n"
        )

        file.write(
            result_3.to_string(index=False)
        )

        file.write(
            "\n\n" + "=" * 70 + "\n\n"
        )

        file.write(
            "QUERY 4 — DISTINCT\n"
        )

        file.write(
            query_4.strip() + "\n\n"
        )

        file.write(
            result_4.to_string(index=False)
        )

        file.write(
            "\n\n" + "=" * 70 + "\n\n"
        )

        file.write(
            "QUERY 5 — BETWEEN\n"
        )

        file.write(
            query_5.strip() + "\n\n"
        )

        file.write(
            result_5.to_string(index=False)
        )

        file.write(
            "\n\n" + "=" * 70 + "\n\n"
        )

        file.write(
            "QUERY 6 — JOIN\n"
        )

        file.write(
            join_query.strip() + "\n\n"
        )

        file.write(
            "pd.read_sql() result:\n"
        )

        file.write(
            join_result_sql.to_string(
                index=False
            )
        )

        file.write(
            "\n\npd.merge() result:\n"
        )

        file.write(
            join_result_pandas.to_string(
                index=False
            )
        )

        file.write(
            "\n\nJOIN MATCH: "
            + str(are_equal)
        )

    # --------------------------------------------------
    # CLOSE DATABASE
    # --------------------------------------------------

    connection.close()

    print("\n" + "=" * 70)
    print("SQL ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"\nResults saved to:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()