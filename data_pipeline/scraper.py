import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import csv


BASE_URL = "https://books.toscrape.com/"
OUTPUT_FILE = "data_pipeline/outputs/books_raw.csv"


def get_soup(url):
    """Download a webpage and return BeautifulSoup."""

    response = requests.get(url, timeout=20)
    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


def get_categories():
    """Get all book categories from the website."""

    soup = get_soup(BASE_URL)

    categories = []

    for link in soup.select(".side_categories ul li ul li a"):
        category_name = link.get_text(strip=True)

        category_url = urljoin(
            BASE_URL,
            link.get("href")
        )

        categories.append({
            "category": category_name,
            "url": category_url
        })

    return categories


def scrape_category(category_name, category_url):
    """Scrape all books from one category, including pagination."""

    books = []
    current_url = category_url
    page_number = 1

    while current_url:

        print(
            f"  Page {page_number}: {current_url}"
        )

        soup = get_soup(current_url)

        page_books = soup.select(
            "article.product_pod"
        )

        print(
            f"  Books on this page: {len(page_books)}"
        )

        for book in page_books:

            title_element = book.select_one(
                "h3 a"
            )

            price_element = book.select_one(
                ".price_color"
            )

            availability_element = book.select_one(
                ".availability"
            )

            rating_element = book.select_one(
                "p.star-rating"
            )

            title = ""

            if title_element:
                title = title_element.get(
                    "title",
                    ""
                ).strip()

            price = ""

            if price_element:
                price = price_element.get_text(
                    strip=True
                )

            availability = ""

            if availability_element:
                availability = availability_element.get_text(
                    " ",
                    strip=True
                )

            rating = ""

            if rating_element:

                rating_classes = rating_element.get(
                    "class",
                    []
                )

                rating_words = {
                    "One": 1,
                    "Two": 2,
                    "Three": 3,
                    "Four": 4,
                    "Five": 5
                }

                for word, number in rating_words.items():

                    if word in rating_classes:
                        rating = number
                        break

            books.append({
                "title": title,
                "category": category_name,
                "price_gbp": price,
                "rating": rating,
                "availability": availability
            })

        next_button = soup.select_one(
            "li.next a"
        )

        if next_button:

            next_url = urljoin(
                current_url,
                next_button.get("href")
            )

            current_url = next_url
            page_number += 1

        else:

            current_url = None

    return books


def save_to_csv(books):
    """Save scraped books to CSV."""

    fieldnames = [
        "title",
        "category",
        "price_gbp",
        "rating",
        "availability"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(books)


def main():

    print("=" * 60)
    print("ZEPTO BOOK SCRAPER")
    print("=" * 60)

    categories = get_categories()

    print(
        f"\nCategories found: {len(categories)}"
    )

    selected_categories = categories[:3]

    print("\nSelected categories:")

    for category in selected_categories:

        print(
            f"- {category['category']}"
        )

    all_books = []

    for category in selected_categories:

        print(
            f"\nScraping category: {category['category']}"
        )

        books = scrape_category(
            category["category"],
            category["url"]
        )

        print(
            f"Total books in {category['category']}: {len(books)}"
        )

        all_books.extend(books)

    save_to_csv(all_books)

    print("\n" + "=" * 60)
    print(
        f"TOTAL BOOKS SCRAPED: {len(all_books)}"
    )
    print("=" * 60)

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()