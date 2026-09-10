# Zepto Data & AI Platform

A capstone project containing three modules:

1. Data Pipeline
2. Analytics & Machine Learning
3. GenAI Support Assistant

The project demonstrates web scraping, data cleaning, SQL analysis, exploratory data analysis, machine learning, model evaluation, regression, retrieval-augmented generation (RAG), LangGraph orchestration, FastAPI, and Docker.

---

## Project Structure

```text
zepto-data-ai-platform/
│
├── data_pipeline/
│   ├── scraper.py
│   ├── clean_data.py
│   ├── database.py
│   ├── sql_analysis.py
│   └── outputs/
│
├── analytics/
│   ├── 01_eda.py
│   ├── 02_classification.py
│   ├── 03_regression.py
│   ├── titanic.csv
│   └── outputs/
│
├── support_assistant/
│   ├── docs/
│   ├── data/
│   ├── embeddings.py
│   ├── retrieval.py
│   ├── graph.py
│   ├── api.py
│   ├── create_policies.py
│   ├── Dockerfile
│   └── README.md
│
├── .gitignore
├── requirements.txt
├── run_project.py
└── README.md