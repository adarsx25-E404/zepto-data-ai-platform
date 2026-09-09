# Zepto Data & AI Platform

A capstone project containing three modules:

1. Data Pipeline
2. Analytics & Machine Learning
3. GenAI Support Assistant

---

# Module 1 — Data Pipeline

## Overview

The Data Pipeline module collects book information from Books to Scrape, cleans the data, converts prices from GBP to INR, stores the data in a normalized SQLite database, and performs SQL and pandas analysis.

## Data Source

Website:

https://books.toscrape.com/

The scraper collects book information from multiple book categories.

### Fields collected

- `title`
- `category`
- `price_gbp`
- `rating`
- `availability`

The final dataset contains 69 books across 3 categories.

---

## Installation

Create and activate a Python virtual environment:

```bash
python -m venv .venv