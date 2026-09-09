import sqlite3
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

INPUT_FILE = Path(
    "data_pipeline/outputs/books_cleaned.csv"
)

DATABASE_FILE = Path(
    "data_pipeline/outputs/books.db"
)


# --------------------------------------------------
# DATABASE CREATION
# --------------------------------------------------

def create_database():

    print("=" * 60)
    print("CREATING SQLITE DATABASE")
    print("=" * 60)

    # ----------------------------------------------
    # Check cleaned CSV
    # ----------------------------------------------

    if not INPUT_FILE.exists():

        print(
            f"\nERROR: Cleaned file not found:"
            f"\n{INPUT_FILE}"
        )

        print(
            "\nRun the cleaning script first:"
        )

        print(
            "python data_pipeline\\clean_data.py"
        )

        return

    # ----------------------------------------------
    # Load cleaned data
    # ----------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print(
        f"\nCleaned rows loaded: {len(df)}"
    )

    # ----------------------------------------------
    # Connect to SQLite
    # ----------------------------------------------

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    # ----------------------------------------------
    # Enable foreign keys
    # ----------------------------------------------

    cursor.execute(
        "PRAGMA foreign_keys = ON"
    )

    # ----------------------------------------------
    # Remove old tables if they exist
    # ----------------------------------------------

    cursor.execute(
        "DROP TABLE IF EXISTS books"
    )

    cursor.execute(
        "DROP TABLE IF EXISTS categories"
    )

    # ----------------------------------------------
    # Create categories table
    # ----------------------------------------------

    cursor.execute(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT UNIQUE NOT NULL
        )
        """
    )

    # ----------------------------------------------
    # Create books table
    # ----------------------------------------------

    cursor.execute(
        """
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
        """
    )

    # ----------------------------------------------
    # Insert categories
    # ----------------------------------------------

    categories = (
        df["category"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    for category_id, category_name in enumerate(
        categories,
        start=1
    ):

        cursor.execute(
            """
            INSERT INTO categories
            (category_id, category_name)
            VALUES (?, ?)
            """,
            (
                category_id,
                category_name
            )
        )

    # ----------------------------------------------
    # Create category lookup
    # ----------------------------------------------

    category_lookup = {
        category_name: category_id
        for category_id, category_name
        in enumerate(categories, start=1)
    }

    # ----------------------------------------------
    # Insert books
    # ----------------------------------------------

    for book_id, row in enumerate(
        df.itertuples(index=False),
        start=1
    ):

        category_id = category_lookup[
            row.category
        ]

        cursor.execute(
            """
            INSERT INTO books
            (
                book_id,
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                book_id,
                row.title,
                float(row.price_gbp),
                float(row.price_inr),
                int(row.rating),
                int(row.in_stock),
                category_id
            )
        )

    # ----------------------------------------------
    # Save database
    # ----------------------------------------------

    connection.commit()

    # ----------------------------------------------
    # Validation
    # ----------------------------------------------

    category_count = cursor.execute(
        "SELECT COUNT(*) FROM categories"
    ).fetchone()[0]

    book_count = cursor.execute(
        "SELECT COUNT(*) FROM books"
    ).fetchone()[0]

    print(
        f"\nCategories inserted: {category_count}"
    )

    print(
        f"Books inserted: {book_count}"
    )

    # ----------------------------------------------
    # Show table names
    # ----------------------------------------------

    tables = cursor.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
        """
    ).fetchall()

    print("\nTables created:")

    for table in tables:

        print(
            f"- {table[0]}"
        )

    # ----------------------------------------------
    # Show sample JOIN
    # ----------------------------------------------

    print("\nSample JOIN result:")

    join_query = """
        SELECT
            books.book_id,
            books.title,
            categories.category_name,
            books.price_inr,
            books.rating
        FROM books
        JOIN categories
            ON books.category_id =
               categories.category_id
        LIMIT 5
    """

    join_result = pd.read_sql(
        join_query,
        connection
    )

    print(
        join_result.to_string(index=False)
    )

    # ----------------------------------------------
    # Close database
    # ----------------------------------------------

    connection.close()

    print(
        f"\nDatabase saved to:"
        f"\n{DATABASE_FILE}"
    )

    print("\n" + "=" * 60)
    print("DATABASE CREATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    create_database()