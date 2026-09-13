# Zepto Data & AI Platform

A complete end-to-end capstone project combining **data engineering, web scraping, SQL analytics, exploratory data analysis, machine learning, and an AI-powered support assistant** into a single Python platform.

The project is organized into three major modules:

1. **Data Pipeline & SQL Analytics**
2. **EDA & Machine Learning**
3. **AI Support Assistant**

The implementation is designed to be reproducible locally and includes optional API and deployment support.

---

# 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Project Architecture](#-project-architecture)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [Running the Complete Project](#-running-the-complete-project)
- [Module 1 - Data Pipeline and SQL Analytics](#-module-1---data-pipeline-and-sql-analytics)
- [Module 2 - EDA and Machine Learning](#-module-2---eda-and-machine-learning)
- [Module 3 - AI Support Assistant](#-module-3---ai-support-assistant)
- [FastAPI Usage](#-fastapi-usage)
- [Docker](#-docker)
- [Vercel Deployment](#-vercel-deployment)
- [Design Decisions](#-design-decisions)
- [Generated Outputs](#-generated-outputs)
- [Testing](#-testing)
- [Final Results](#-final-results)
- [Git Workflow](#-git-workflow)
- [Conclusion](#-conclusion)

---

# 📌 Project Overview

This capstone demonstrates a complete data and AI workflow starting from raw data collection and ending with machine-learning models and an AI support application.

The project contains three modules.

## Module 1 - Data Pipeline & SQL Analytics

The first module implements a complete data pipeline:


Website
   ↓
Web Scraping
   ↓
Raw CSV
   ↓
Data Cleaning
   ↓
Clean CSV
   ↓
SQLite Database
   ↓
SQL Analytics
   ↓
Pandas Analysis


The module:

* Scrapes books from Books to Scrape.
* Collects title, price, rating, availability, and category information.
* Cleans the scraped data.
* Converts GBP prices to INR.
* Uses the required fixed exchange rate.
* Stores normalized data in SQLite.
* Executes multiple SQL queries.
* Performs SQL joins.
* Reproduces relational joins using pandas.
* Saves intermediate and final outputs.

---

## Module 2 - EDA & Machine Learning

The second module performs a complete analytics and machine-learning workflow using the Titanic dataset.

The workflow includes:

* Dataset loading.
* Data cleaning.
* Missing-value analysis.
* Outlier detection.
* Univariate analysis.
* Multivariate analysis.
* Survival-rate analysis.
* Correlation analysis.
* Feature standardization.
* Stratified train/test splitting.
* Classification.
* Class imbalance handling.
* SMOTE.
* Random Forest hyperparameter tuning.
* Decision-tree visualization.
* ROC/AUC evaluation.
* Fare regression.
* Residual analysis.
* Model persistence using Joblib.

---

## Module 3 - AI Support Assistant

The third module implements an offline-capable Zepto policy support assistant.

The system uses:

* Sentence Transformers.
* `all-MiniLM-L6-v2`.
* ChromaDB.
* LangGraph.
* Pydantic.
* FastAPI.
* Uvicorn.
* Docker.

The assistant supports:

* Policy intent classification.
* Top-3 vector retrieval.
* Structured prompting.
* Few-shot prompting.
* Negative constraints to reduce unsupported answers.
* Pydantic output validation.
* Mock/offline LLM operation.
* Optional real-LLM operation.
* FastAPI API access.

---

#  Project Architecture

                         ZEPTO DATA & AI PLATFORM
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
       MODULE 1              MODULE 2             MODULE 3
   Data Pipeline & SQL     EDA & Machine Learning   AI Assistant
              │                   │                   │
              ▼                   ▼                   ▼
        Web Scraping          Titanic Data       Policy Documents
              │                   │                   │
              ▼                   ▼                   ▼
          Cleaning                EDA             Text Chunking
              │                   │                   │
              ▼                   ▼                   ▼
          SQLite DB          ML Pipelines       Embeddings
              │                   │                   │
              ▼                   ▼                   ▼
        SQL Analytics       Classification       ChromaDB
              │                   │                   │
              │                   ├── Logistic       │
              │                   ├── Tree           │
              │                   ├── Random Forest  │
              │                   ├── SMOTE          │
              │                   └── Regression     │
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  │
                                  ▼
                       Reproducible Capstone

---

#  Technologies Used

## Module 1

* Python
* Requests
* BeautifulSoup
* Pandas
* NumPy
* SQLite
* SQL

## Module 2

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* Joblib

## Module 3

* Python
* Sentence Transformers
* `all-MiniLM-L6-v2`
* ChromaDB
* LangGraph
* FastAPI
* Pydantic
* Uvicorn
* Docker

## Deployment

* Vercel
* FastAPI

---

#  Project Structure


zepto-data-ai-platform/
│
├── analytics/
│   ├── 01_eda.py
│   ├── 02_classification.py
│   ├── 03_regression.py
│   ├── titanic.csv
│   └── outputs/
│       ├── classification_results.csv
│       ├── classification_summary.txt
│       ├── classification_roc_curves.png
│       ├── decision_tree_plot.png
│       ├── best_random_forest_pipeline.joblib
│       ├── regression_results.txt
│       ├── fare_regression_residuals.png
│       └── fare_regression_pipeline.joblib
│
├── data_pipeline/
│   ├── scrape_books.py
│   ├── clean_books.py
│   ├── database.py
│   ├── sql_analysis.py
│   ├── outputs/
│   │   ├── books_raw.csv
│   │   ├── books_clean.csv
│   │   ├── books.db
│   │   └── query_results.txt
│   └── ...
│
├── support_assistant/
│   ├── api.py
│   ├── graph.py
│   ├── embeddings.py
│   ├── retrieval.py
│   ├── create_policies.py
│   ├── Dockerfile
│   ├── docs/
│   │   ├── doc_01.txt
│   │   ├── doc_02.txt
│   │   ├── doc_03.txt
│   │   ├── doc_04.txt
│   │   ├── doc_05.txt
│   │   ├── doc_06.txt
│   │   ├── doc_07.txt
│   │   └── doc_08.txt
│   └── data/
│       └── chroma_db/
│
├── api/
│   └── index.py
│
├── requirements.txt
├── requirements-full.txt
├── vercel.json
├── run_project.py
└── README.md


---

#  Installation and Setup

## 1. Clone the Repository


git clone https://github.com/adarsx25-E404/zepto-data-ai-platform.git
cd zepto-data-ai-platform


---

## 2. Create a Virtual Environment

### Windows PowerShell


python -m venv .venv


Activate the environment:


.\.venv\Scripts\Activate.ps1


If PowerShell blocks script execution, run:


Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser


Then activate again:


.\.venv\Scripts\Activate.ps1


---

## 3. Install Dependencies

The complete local project uses:

powershell
pip install -r requirements-full.txt


The full dependency file contains the libraries required for:

* Data scraping
* Data processing
* Machine learning
* Imbalanced learning
* Embeddings
* ChromaDB
* LangGraph
* FastAPI
* Uvicorn
* Pydantic

The root `requirements.txt` is intentionally lightweight for the serverless deployment adapter.

For the complete capstone execution, use:

powershell
pip install -r requirements-full.txt


---

# ▶️ Running the Complete Project

The three modules can be executed independently.

Recommended execution order:

text
Module 1
   ↓
Module 2
   ↓
Module 3


A convenience runner is also available:

powershell
python .\run_project.py


For maximum control and reproducibility, the individual commands for each module are provided below.

---

# 📚 MODULE 1 - DATA PIPELINE AND SQL ANALYTICS

## Objective

Module 1 demonstrates a complete ETL-style data pipeline.

text
Extract
  ↓
Transform
  ↓
Load
  ↓
Analyze


---

## Step 1 - Scrape Book Data

Run:

powershell
python .\data_pipeline\scrape_books.py


The scraper collects:

* Book title
* Price in GBP
* Star rating text
* Availability
* Category

The implementation collects more than the required minimum of 60 books.

### Scraped Dataset

A total of:

text
69 books


were collected from three categories.

| Category           | Number of Books |
| ------------------ | --------------: |
| Travel             |              11 |
| Mystery            |              32 |
| Historical Fiction |              26 |
| **Total**          |          **69** |

---

## Step 2 - Clean the Data

Run:

powershell
python .\data_pipeline\clean_books.py


The cleaning process:

* Converts price values into numeric form.
* Converts rating values into numeric form.
* Cleans availability values.
* Standardizes categories.
* Removes unnecessary text formatting.
* Creates INR price values.

---

## 💷 GBP to INR Conversion

The required fixed exchange rate is:

text
1 GBP = 105.50 INR


The conversion is:

text
price_inr = price_gbp × 105.50


A fixed exchange rate is used instead of a live exchange-rate API so that the results remain deterministic and reproducible.

---

## Step 3 - Load into SQLite

Run:

powershell
python .\data_pipeline\database.py


The SQLite database is created under:

text
data_pipeline/outputs/books.db


The database uses a normalized relational structure with multiple tables and primary-key/foreign-key relationships.

---

## Step 4 - Execute SQL Queries

Run:

powershell
python .\data_pipeline\sql_analysis.py


The SQL analysis includes the required query operations:

text
SELECT
WHERE
ORDER BY
LIMIT
DISTINCT
IN
BETWEEN
JOIN


The query results are saved under:

text
data_pipeline/outputs/


---

## Step 5 - Pandas SQL Analysis

The project also demonstrates SQL-to-pandas analysis using:

python
pd.read_sql()


The SQL result sets are loaded into pandas DataFrames.

The equivalent relational join is also implemented using:

python
pd.merge()


This demonstrates that the same relational operation can be represented using both SQL and pandas.

---

# 📊 MODULE 2 - EDA AND MACHINE LEARNING

## Objective

Module 2 performs a complete exploratory analysis and machine-learning workflow using the Titanic dataset.

text
Titanic Dataset
      ↓
Data Loading
      ↓
EDA
      ↓
Missing Values
      ↓
Outliers
      ↓
Correlation
      ↓
Train/Test Split
      ↓
Preprocessing
      ↓
Classification
      ↓
Class Imbalance
      ↓
Random Forest Tuning
      ↓
Regression
      ↓
Model Persistence


---

# 📥 Dataset Loading

The Titanic dataset is loaded and stored locally as:

text
analytics/titanic.csv


The resulting dataset contains:

text
889 rows
13 columns


The local CSV is then reused by the analytics scripts.

---

# 🧹 Missing Value Analysis

The missing-value strategy follows percentage-based thresholds.

| Column      | Missing Values | Percentage | Decision              |
| ----------- | -------------: | ---------: | --------------------- |
| age         |            177 |     19.87% | Impute                |
| embarked    |              2 |      0.22% | Drop rows             |
| deck        |            688 |     77.22% | Drop column           |
| embark_town |              2 |      0.22% | Drop rows / redundant |

### Reasoning

* Features with a moderate amount of missing data are imputed.
* Very small numbers of missing categorical observations are removed.
* `deck` contains too many missing values to provide reliable information and is therefore removed.
* `embark_town` is redundant with the embarkation information and has only a small number of missing values.

---

# 📈 Univariate Analysis

The EDA includes:

* Age histogram
* Age boxplot
* Fare histogram
* Fare boxplot
* IQR outlier analysis
* Fare mean
* Fare median
* Fare mode
* Fare skewness

---

# 📊 Outlier Analysis

IQR analysis identified:

text
Age outliers  = 65
Fare outliers = 114


The fare distribution contains several high-value observations.

This contributes to the strong right skew of the fare variable.

---

# 💰 Fare Statistics

The observed fare statistics are:

| Statistic | Value |
| --------- | ----: |
| Mean      | 32.10 |
| Median    | 14.45 |
| Mode      |  8.05 |

The mean is substantially greater than the median, showing that the fare distribution is positively/right skewed.

---

# 👥 Survival Analysis

## Survival by Sex

| Sex    | Survival Rate |
| ------ | ------------: |
| Female |        74.04% |
| Male   |        18.89% |

Female passengers had a substantially higher survival rate than male passengers.

---

## Survival by Passenger Class

| Passenger Class | Survival Rate |
| --------------- | ------------: |
| 1st Class       |        62.62% |
| 2nd Class       |        47.28% |
| 3rd Class       |        24.24% |

Survival probability decreases as passenger class moves from first class to third class.

---

## Survival by Sex and Passenger Class

| Sex    | Class | Survival Rate |
| ------ | ----: | ------------: |
| Female |     1 |        96.74% |
| Female |     2 |        92.11% |
| Female |     3 |        50.00% |
| Male   |     1 |        36.89% |
| Male   |     2 |        15.74% |
| Male   |     3 |        13.54% |

The combined sex and passenger-class analysis reveals stronger patterns than either feature alone.

Female first- and second-class passengers had particularly high survival rates, while male second- and third-class passengers had much lower survival rates.

---

# 🔗 Correlation Analysis

The required correlation variables are:

text
survived
pclass
age
sibsp
parch
fare


The following variables are intentionally excluded:

text
adult_male
alone


Important correlations with survival include:

| Feature | Correlation with Survival |
| ------- | ------------------------: |
| pclass  |                   -0.3355 |
| fare    |                    0.2553 |
| parch   |                    0.0832 |
| age     |                   -0.0698 |
| sibsp   |                   -0.0340 |

The strongest absolute off-diagonal relationships are approximately:

text
pclass ↔ fare   ≈ -0.55
sibsp  ↔ parch  ≈  0.41


### Interpretation

The negative relationship between `pclass` and `fare` reflects the numerical coding of passenger class and the tendency for higher-status passengers to have higher fares.

The positive relationship between `sibsp` and `parch` indicates that passengers travelling with siblings/spouses were also more likely to travel with parents/children.

The correlation between `pclass` and survival also indicates that passenger class is an important predictor of survival.

---

# 📊 Multivariate Analysis

Multiple multivariate visualizations are generated to analyze relationships between passenger attributes.

The analysis includes relationships such as:

* Survival by sex and passenger class.
* Age and survival.
* Fare and survival.
* Passenger class and fare.
* Correlation relationships between numerical features.

### Interpretation

The multivariate analysis shows that survival is influenced by multiple passenger characteristics rather than one isolated variable.

Sex and passenger class show particularly strong differences in survival outcomes.

Fare also provides useful information because it is related to passenger class and socioeconomic position.

---

# 📏 Feature Standardization

Age and fare are standardized using:

text
StandardScaler


Approximate original statistics:

| Feature |   Mean | Standard Deviation |
| ------- | -----: | -----------------: |
| Age     | 29.315 |             12.985 |
| Fare    | 32.097 |             49.698 |

After standardization:

text
Mean ≈ 0
Standard deviation ≈ 1


The transformation is performed inside the preprocessing pipeline.

This ensures that preprocessing parameters are learned only from the training data.

---

# ✂️ Train/Test Split

The classification dataset is split using a stratified train/test split:

python
train_test_split(
    test_size=0.20,
    random_state=42,
    stratify=y
)


Result:

text
Training rows = 711
Testing rows  = 178


Target distribution:

text
Not Survived = 549
Survived     = 340


Approximately:

text
61.75% Not Survived
38.25% Survived


Stratification ensures that both training and test datasets retain a similar class distribution.

---

# 🧪 Machine Learning Preprocessing

The preprocessing pipeline handles:

* Numerical missing values.
* Categorical missing values.
* Numerical standardization.
* One-hot encoding.

The preprocessing is fitted only on the training set.

The test set is transformed using the already-fitted preprocessing pipeline.

This prevents data leakage.

---

# 🤖 Classification Models

Three primary classification algorithms are evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Each model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

---

# 📋 Classification Results

| Model               |   Accuracy |  Precision | Recall |         F1 | ROC-AUC |
| ------------------- | ---------: | ---------: | -----: | ---------: | ------: |
| Logistic Regression |     0.8090 |     0.7833 | 0.6912 |     0.7344 |  0.8610 |
| Decision Tree       |     0.7697 |     0.6901 | 0.7206 |     0.7050 |  0.7541 |
| Random Forest       |     0.8090 |     0.7656 | 0.7206 |     0.7424 |  0.8196 |
| Logistic Balanced   |     0.7921 |     0.7183 | 0.7500 |     0.7338 |  0.8612 |
| Logistic + SMOTE    |     0.7978 |     0.7353 | 0.7353 |     0.7353 |  0.8667 |
| Tuned Random Forest | **0.8315** | **0.8654** | 0.6618 | **0.7500** |  0.8389 |

---

# ⚖️ Class Imbalance Handling

The classification target is moderately imbalanced.

The project evaluates multiple strategies.

## Baseline

Models are first evaluated without explicit balancing.

## Class Weighting

Balanced Logistic Regression uses:

python
class_weight="balanced"


This increases the importance of the minority class during training.

## SMOTE

SMOTE is also evaluated.

SMOTE is applied only to training data through an imbalanced-learn pipeline.

The test set remains untouched.

This prevents synthetic observations from contaminating the evaluation data.

---

# 🌲 Random Forest Hyperparameter Tuning

Random Forest hyperparameters are optimized using:

text
GridSearchCV


The search space includes:

text
n_estimators
max_depth
max_features


The Random Forest estimator uses:

python
RandomForestClassifier(
    oob_score=True,
    random_state=42,
    n_jobs=-1
)


The best parameters were:

text
n_estimators = 200
max_depth = 5
max_features = sqrt


Best cross-validation F1-score:

text
0.7408


Out-of-bag score:

text
0.8073


---

# 🏆 Best Classification Model

The tuned Random Forest achieved:

text
Accuracy  = 0.8315
Precision = 0.8654
Recall    = 0.6618
F1-score  = 0.7500
ROC-AUC   = 0.8389


Based on the selected F1-focused tuning objective, the tuned Random Forest provides the strongest overall classification result.

The trained complete pipeline is saved as:

text
analytics/outputs/best_random_forest_pipeline.joblib


---

# 🌳 Decision Tree Visualization

The project generates a labeled Decision Tree visualization using:

python
plot_tree()


The output is:

text
analytics/outputs/decision_tree_plot.png


The visualization shows the learned decision structure and feature-based splitting rules.

---

# 📈 ROC Curves

ROC curves are generated for the classification models.

Output:

text
analytics/outputs/classification_roc_curves.png


The ROC-AUC comparison demonstrates the ranking performance of the different classifiers.

Logistic Regression and SMOTE Logistic Regression provide particularly strong ROC-AUC values, while the tuned Random Forest provides the strongest overall F1/accuracy combination.

---

# 💵 FARE REGRESSION

The regression task predicts:

text
fare


using the available passenger-related features.

The regression pipeline includes preprocessing followed by the regression estimator.

---

# 📊 Regression Results

| Metric      |  Result |
| ----------- | ------: |
| MAE         | 18.3735 |
| RMSE        | 41.2921 |
| R²          |  0.3609 |
| Adjusted R² |  0.2702 |

The model uses approximately:

text
22 predictors
178 test observations


---

# 📈 Regression Interpretation

The R² value of approximately 0.36 indicates that the model explains a meaningful but limited portion of the variation in passenger fares.

The RMSE is considerably larger than the MAE, indicating that some observations have relatively large prediction errors.

Residual analysis shows that the residual spread becomes wider at higher fare levels.

Therefore, the regression results suggest possible **heteroscedasticity**, meaning that prediction error variance is not constant across the range of predicted fares.

The residual plot is saved as:

text
analytics/outputs/fare_regression_residuals.png


---

# 💾 Model Persistence

The project saves complete preprocessing + estimator pipelines using Joblib.

## Classification

text
analytics/outputs/best_random_forest_pipeline.joblib


## Regression

text
analytics/outputs/fare_regression_pipeline.joblib


The saved pipelines can be reloaded and used on raw input data without manually rebuilding the preprocessing stages.

---

# 🤖 MODULE 3 - AI SUPPORT ASSISTANT

## Objective

The third module implements an offline-capable Zepto policy support assistant using a retrieval-based architecture.

The complete architecture is:

text
Policy Documents
       ↓
Document Chunking
       ↓
Sentence Transformer
       ↓
Embeddings
       ↓
ChromaDB
       ↓
User Query
       ↓
Intent Classification
       ↓
Top-3 Retrieval
       ↓
Answer Generation
       ↓
Pydantic Validation
       ↓
Final Response


---

# 📚 Policy Documents

The assistant uses eight local policy documents:

text
support_assistant/docs/doc_01.txt
support_assistant/docs/doc_02.txt
support_assistant/docs/doc_03.txt
support_assistant/docs/doc_04.txt
support_assistant/docs/doc_05.txt
support_assistant/docs/doc_06.txt
support_assistant/docs/doc_07.txt
support_assistant/docs/doc_08.txt


The system is designed to operate locally without requiring an external API in mock mode.

---

# 🧩 Document Chunking

The policy documents are divided into smaller chunks.

The current configuration uses approximately:

text
Chunk size = 500 characters
Overlap    = 100 characters


Chunk overlap helps preserve context between neighboring text segments.

---

# 🧠 Embeddings

The embedding model is:

text
all-MiniLM-L6-v2


Sentence Transformers converts policy chunks into numerical vector representations.

These vectors are stored in ChromaDB.

---

# 🗄️ ChromaDB

The local vector database is stored under:

text
support_assistant/data/chroma_db/


The collection is:

text
zepto_support_policies


The embedding script clears previous records before ingestion to avoid duplicate entries during repeated runs.

---

# 🔎 Retrieval

For each user query, the system retrieves the:

text
Top 3


most relevant policy chunks from ChromaDB.

The retrieval mechanism is used in both assistant workflow paths so that the system maintains a consistent retrieval-based architecture.

---

# 🧠 LangGraph Workflow

The assistant is implemented using LangGraph `StateGraph`.

The state is represented using:

python
TypedDict


The main nodes are:

text
classify_intent
retrieve_and_answer
direct_answer


The workflow is:

text
                  ┌─────────────────────┐
                  │    User Query       │
                  └──────────┬──────────┘
                             ↓
                  ┌─────────────────────┐
                  │   classify_intent   │
                  └──────────┬──────────┘
                             ↓
                    ┌────────┴────────┐
                    │                 │
              Policy Query       Other Query
                    │                 │
                    ↓                 ↓
          ┌──────────────────┐ ┌───────────────┐
          │retrieve_and_answer│ │ direct_answer │
          └─────────┬────────┘ └───────┬───────┘
                    │                  │
                    └─────────┬────────┘
                              ↓
                    ┌─────────────────┐
                    │ Final Response  │
                    └─────────────────┘


---

# 🎯 Intent Classification

The required keyword heuristic is used for policy intent detection.

Supported keywords include:

text
delivery
return
refund
membership
tracking
cancel
gift card
support hours


Queries containing these policy-related terms are routed through the retrieval-and-answer path.

Other questions are handled through the direct-answer path.

---

# 📝 Structured Prompt

The assistant uses a structured prompt containing:

text
ROLE
CONTEXT
TASK
NEGATIVE CONSTRAINT
FORMAT
LENGTH
FEW-SHOT EXAMPLE


The prompt design helps constrain the generated answer to the retrieved policy context.

The negative constraint reduces the likelihood of unsupported policy claims.

The few-shot example demonstrates the expected response format.

---

# 🧾 Pydantic Output Validation

The final answer is validated using a Pydantic schema:

text
FinalAnswer


The response structure contains:

text
answer
sources
confidence


The confidence value is constrained between:

text
0.0 and 1.0


This ensures that the final API output follows a predictable structured format.

---

# 📴 Offline Mock Mode

The project supports a fully offline mock configuration.

Set:

text
MOCK_LLM=1


In mock mode:

* No external LLM API is required.
* Policy documents remain local.
* ChromaDB remains local.
* Retrieval remains local.
* The assistant returns a deterministic response based on retrieved context.

This makes the project suitable for reproducible demonstrations and grading environments without API credentials.

---

# 🌐 Optional Real LLM Mode

A real LLM can optionally be enabled using:

text
MOCK_LLM=0


When real LLM mode is enabled, the generated response is validated against the Pydantic schema.

If validation fails, the system supports retries.

The maximum generation attempts are:

text
3 total attempts


This corresponds to one initial attempt plus up to two additional retries.

---

# 🚀 Running Module 3

## Step 1 - Activate the environment

powershell
.\.venv\Scripts\Activate.ps1


---

## Step 2 - Install dependencies

powershell
pip install -r requirements-full.txt


---

## Step 3 - Build the vector database

Run:

powershell
python .\support_assistant\embeddings.py


This:

* Reads the eight policy documents.
* Splits the documents into chunks.
* Generates Sentence Transformer embeddings.
* Stores embeddings in ChromaDB.
* Performs a retrieval test.

---

## Step 4 - Start the FastAPI server

Run:

powershell
uvicorn support_assistant.api:app --reload


The local API will then be available through the FastAPI application.

---

# 🔌 FASTAPI USAGE

The main endpoint is:

text
POST /ask


---

## Example Request

json
{
  "query": "What is the refund policy?"
}


---

## PowerShell Example

powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/ask" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"query":"What is the refund policy?"}'
 

---

# ❤️ Health Check

The API provides:

.text
GET /health


Example:

text
http://127.0.0.1:8000/health


This endpoint can be used to verify that the FastAPI service is running.

---

# 📖 Interactive API Documentation

FastAPI automatically provides Swagger documentation at:

text
/docs


When running locally:

text
http://127.0.0.1:8000/docs


The documentation allows the API endpoints to be tested directly from a browser.

---

# 🧪 Example Queries

## Refund

json
{
  "query": "What is the refund policy?"
}


## Delivery

json
{
  "query": "How long does delivery take?"
}


## Return

json
{
  "query": "Can I return an item?"
}


## Membership

json
{
  "query": "What are the membership benefits?"
}


---

# 🐳 DOCKER

A Dockerfile is included at:

text
support_assistant/Dockerfile


Build the Docker image:

powershell
docker build -t zepto-support-assistant .\support_assistant


Run the container:

powershell
docker run -p 8000:8000 zepto-support-assistant


The FastAPI application will then be accessible through:

text
http://127.0.0.1:8000


---

# ☁️ VERCEL DEPLOYMENT

The repository includes a lightweight serverless deployment adapter:

text
api/index.py


and deployment configuration:

text
vercel.json


The deployed application exposes:

text
/
 /health
 /docs
 /ask


The Vercel adapter is intentionally lightweight.

It does not import the full:

text
SentenceTransformer
ChromaDB
LangGraph


stack because those dependencies create a very large serverless bundle.

Instead, the lightweight deployment adapter provides the production-facing API while the complete capstone AI implementation remains under:

text
support_assistant/


The full local implementation remains the primary implementation for the graded AI assistant.

Deployment is an additional capability and is not required for the core local implementation.

---

# 🔐 Environment Configuration

The default offline configuration is:

text
MOCK_LLM=1


Optional real LLM operation:

text
MOCK_LLM=0


API credentials, if required for real LLM operation, should be stored as environment variables rather than hard-coded into the source code.

---

# 🎯 DESIGN DECISIONS

## 1. SQLite for Data Storage

SQLite was selected for Module 1 because it is lightweight, portable, serverless, and appropriate for a self-contained capstone project.

It allows relational database concepts such as primary keys, foreign keys, normalization, joins, and analytical SQL queries to be demonstrated without requiring a separate database server.

---

## 2. Fixed GBP-to-INR Exchange Rate

The project uses:

text
1 GBP = 105.50 INR


as required.

A fixed rate was preferred over a live exchange-rate API so that the generated dataset remains deterministic and reproducible.

---

## 3. Stratified Train/Test Split

The classification target is imbalanced.

Therefore, a stratified train/test split is used to maintain a similar class distribution in both datasets.

---

## 4. Pipeline-Based Preprocessing

Scikit-learn preprocessing is placed inside pipelines.

This ensures that:

* Imputation is learned from training data.
* Scaling is learned from training data.
* Encoding is learned from training data.
* Test data does not influence preprocessing parameters.

This prevents data leakage.

---

## 5. SMOTE Only on Training Data

SMOTE is implemented through an imbalanced-learn pipeline.

Synthetic observations are created only during training.

The test dataset remains unchanged.

This ensures that the evaluation represents performance on unseen real observations.

---

## 6. F1-Score for Random Forest Tuning

F1-score is used as the GridSearchCV scoring metric because the classification target is imbalanced and both precision and recall are important.

---

## 7. Random Forest OOB Score

The Random Forest estimator uses:

python
oob_score=True


This provides an additional internal evaluation estimate using out-of-bag observations.

---

## 8. ChromaDB for Local Vector Retrieval

ChromaDB was selected because it provides a simple local vector database suitable for an offline retrieval-based assistant.

It eliminates the need for an externally hosted vector database.

---

## 9. Sentence Transformers

`all-MiniLM-L6-v2` provides compact semantic embeddings that can be generated locally.

This makes it suitable for an offline capstone environment.

---

## 10. LangGraph

LangGraph was selected to make the assistant workflow explicit and modular.

The workflow separates:

text
Intent Classification
Retrieval + Answer
Direct Answer


rather than placing all logic inside a single function.

---

## 11. Mock LLM

Mock mode makes the assistant reproducible without requiring external API credentials.

This is useful for local testing, demonstrations, and grading.

---

## 12. Pydantic Validation

Pydantic provides structured validation for final assistant responses.

The expected schema contains:

text
answer
sources
confidence


This makes the API response predictable and machine-readable.

---

# 📂 GENERATED OUTPUTS

## Module 1

text
data_pipeline/outputs/books_raw.csv
data_pipeline/outputs/books_clean.csv
data_pipeline/outputs/books.db
data_pipeline/outputs/query_results.txt


---

## Module 2

text
analytics/titanic.csv

analytics/outputs/classification_results.csv
analytics/outputs/classification_summary.txt
analytics/outputs/classification_roc_curves.png
analytics/outputs/decision_tree_plot.png
analytics/outputs/best_random_forest_pipeline.joblib

analytics/outputs/regression_results.txt
analytics/outputs/fare_regression_residuals.png
analytics/outputs/fare_regression_pipeline.joblib


---

## Module 3

text
support_assistant/docs/
support_assistant/data/chroma_db/


---

# 🧪 TESTING AND VALIDATION

The project was tested across all three modules.

## Module 1 Validation

Verified:

* Book scraping.
* 69 books collected.
* Three categories represented.
* Data cleaning.
* GBP-to-INR conversion.
* SQLite database generation.
* Normalized relational structure.
* SQL queries.
* SQL joins.
* `pd.read_sql()`.
* `pd.merge()`.

---

## Module 2 Validation

Verified:

* Titanic dataset loading.
* Local CSV generation.
* Missing-value analysis.
* IQR outlier analysis.
* Survival-rate analysis.
* Correlation analysis.
* Standardization.
* Stratified train/test split.
* Classification pipelines.
* Confusion matrices.
* Accuracy.
* Precision.
* Recall.
* F1-score.
* ROC-AUC.
* Class weighting.
* SMOTE.
* Random Forest GridSearchCV.
* OOB score.
* Decision-tree visualization.
* Fare regression.
* MAE.
* RMSE.
* R².
* Adjusted R².
* Residual analysis.
* Joblib model persistence.
* Model reloading.

---

## Module 3 Validation

Verified:

* Eight policy documents.
* Document chunking.
* Sentence Transformer embeddings.
* ChromaDB storage.
* Top-3 retrieval.
* Intent classification.
* LangGraph workflow.
* Mock LLM operation.
* Structured prompt.
* Few-shot example.
* Negative constraint.
* Pydantic validation.
* FastAPI server.
* `/ask`.
* `/health`.
* `/docs`.
* Docker configuration.

---

# 📊 FINAL RESULTS

## Module 1

The pipeline successfully collected:

text
69 books


from three categories.

The data was cleaned, converted to INR using:

text
1 GBP = 105.50 INR


and stored in a normalized SQLite database.

The project also demonstrates the required SQL operations and pandas equivalents.

---

## Module 2

The tuned Random Forest produced:

text
Accuracy  = 83.15%
Precision = 86.54%
Recall    = 66.18%
F1-score  = 0.7500
ROC-AUC   = 0.8389
OOB Score = 0.8073


The best parameters were:

text
n_estimators = 200
max_depth = 5
max_features = sqrt


The best cross-validation F1-score was:

text
0.7408


---

## Regression

The fare regression model produced:

text
MAE           = 18.3735
RMSE          = 41.2921
R²            = 0.3609
Adjusted R²   = 0.2702


The results indicate that the model captures some meaningful fare variation but does not explain all of the variability in fares.

Residual analysis also suggests possible heteroscedasticity because residual spread increases at higher fare levels.

---

## Module 3

The support assistant successfully implements a local retrieval-based workflow using:

text
8 policy documents
        ↓
500-character chunks
        ↓
all-MiniLM-L6-v2
        ↓
ChromaDB
        ↓
Top-3 retrieval
        ↓
LangGraph
        ↓
Structured answer
        ↓
Pydantic validation


The system can run completely offline using:

text
MOCK_LLM=1


and optionally supports a real LLM configuration.

---

# 🏆 FINAL RECOMMENDATION

The tuned Random Forest is the recommended classification model because it achieves the highest accuracy and F1-score among the evaluated models, making it the strongest overall choice for the selected evaluation objective.

However, its recall is lower than the balanced Logistic Regression alternatives, so the final model choice would depend on whether minimizing false negatives is more important than maximizing overall F1 and precision.

The fare regression model provides useful predictive information but has moderate explanatory power and evidence of non-constant residual variance, so it should be treated as a baseline rather than a highly accurate fare predictor.

The AI support assistant provides a reproducible offline architecture using local embeddings, vector retrieval, LangGraph orchestration, structured prompting, and Pydantic validation.

Overall, the project demonstrates an end-to-end combination of **data engineering, SQL analytics, machine learning, retrieval-based AI, API development, and deployment**.

---

# 🌿 GIT WORKFLOW

The repository uses feature branches during development.

Important branches include:

text
main
feature/analytics
feature/data-pipeline
feature/support-assistant
feature/vercel-deployment


The project includes feature development followed by integration into the main branch.

The Vercel deployment work was developed on:

text
feature/vercel-deployment


and subsequently merged into:

text
main


The support assistant was also developed through a dedicated feature branch and integrated into the main branch.

This provides a visible Git history demonstrating feature-based development.

---

# 📜 GIT HISTORY HIGHLIGHTS

Representative commits include:

text
dc867c8 Merge Vercel deployment into main
9ace161 Add Vercel deployment configuration
a2e8c4c Complete analytics report
69f0d7f Finalize capstone project updates
89f8d84 Document support assistant module
cb15969 Build support assistant implementation
04bf42b Document analytics module
7ac6233 Add analytics and machine learning results
48bf1a Add data pipeline results and database
95ba09 Build data pipeline processing scripts
ac33d62 Initial project setup


---

# 🔗 GITHUB REPOSITORY

Repository:

text
https://github.com/adarsx25-E404/zepto-data-ai-platform


The repository contains the complete source code, scripts, datasets, generated outputs, model pipelines, support assistant implementation, Docker configuration, deployment configuration, and documentation.

---

# 🚀 QUICK START

For a quick local demonstration:

powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install complete dependencies
pip install -r requirements-full.txt

# 3. Run Module 1
python .\data_pipeline\scrape_books.py
python .\data_pipeline\clean_books.py
python .\data_pipeline\database.py
python .\data_pipeline\sql_analysis.py

# 4. Run Module 2
python .\analytics\01_eda.py
python .\analytics\02_classification.py
python .\analytics\03_regression.py

# 5. Build Module 3 vector database
python .\support_assistant\embeddings.py

# 6. Start Module 3 API
uvicorn support_assistant.api:app --reload


Then open:

text
http://127.0.0.1:8000/docs


and test:

json
{
  "query": "What is the refund policy?"
}


---

# 📌 ONE-COMMAND CONVENIENCE RUNNER

The repository also contains:

text
run_project.py


which can be used as a convenience entry point:

powershell
python .\run_project.py


For detailed debugging or individual module execution, use the module-specific commands documented above.

---

# 📄 CAPSTONE SUMMARY

| Component             | Implementation                                      |
| --------------------- | --------------------------------------------------- |
| Web Scraping          | Requests + BeautifulSoup                            |
| Data Cleaning         | Pandas                                              |
| Database              | SQLite                                              |
| SQL Analytics         | SQL + Pandas                                        |
| EDA                   | Pandas + Matplotlib + Seaborn                       |
| Classification        | Logistic Regression + Decision Tree + Random Forest |
| Imbalance Handling    | Class Weight + SMOTE                                |
| Hyperparameter Tuning | GridSearchCV                                        |
| Regression            | Scikit-learn                                        |
| Model Persistence     | Joblib                                              |
| Embeddings            | Sentence Transformers                               |
| Embedding Model       | all-MiniLM-L6-v2                                    |
| Vector Database       | ChromaDB                                            |
| AI Workflow           | LangGraph                                           |
| Structured Output     | Pydantic                                            |
| API                   | FastAPI                                             |
| Local Server          | Uvicorn                                             |
| Containerization      | Docker                                              |
| Deployment            | Vercel                                              |
| Version Control       | Git + GitHub                                        |

---

# 🎓 CONCLUSION

This capstone project demonstrates a complete progression from raw data acquisition to analytical insights, machine-learning models, and an AI-powered application.

Module 1 establishes the data-engineering foundation through scraping, cleaning, relational database design, SQL analytics, and pandas operations.

Module 2 extends the platform into statistical analysis and machine learning, including preprocessing, classification, imbalance handling, hyperparameter tuning, regression, evaluation, and model persistence.

Module 3 demonstrates a modern retrieval-based AI architecture using local policy documents, embeddings, vector search, LangGraph orchestration, structured prompts, Pydantic validation, and FastAPI.

Together, the three modules form a reproducible end-to-end **Data & AI Platform** that demonstrates practical skills across **data engineering, analytics, machine learning, AI application development, API development, containerization, and deployment**.

---

# 📜 License

This repository was created as an academic capstone project for demonstrating practical implementation of:

* Data Engineering
* Web Scraping
* SQL
* Data Cleaning
* Exploratory Data Analysis
* Machine Learning
* Imbalanced Classification
* Regression
* Model Evaluation
* Vector Search
* Retrieval-Augmented AI
* LangGraph
* FastAPI
* Docker
* Serverless Deployment
* Git/GitHub Workflow
