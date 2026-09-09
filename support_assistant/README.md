# Zepto Customer Support Assistant

## Overview

A policy-grounded customer support assistant built with:

- Sentence Transformers
- all-MiniLM-L6-v2
- ChromaDB
- LangGraph
- FastAPI
- Pydantic

The graded baseline uses deterministic offline `MOCK_LLM` mode.
No LLM API key is required.

## Project Structure

```text
support_assistant/
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
├── data/
│   └── chroma_db/
├── create_policies.py
├── embeddings.py
├── retrieval.py
├── graph.py
├── api.py
├── Dockerfile
└── README.md

---

## RAG Architecture

```text
Policy Documents
      |
      v
Ingestion / Chunking
      |
      v
all-MiniLM-L6-v2
      |
      v
ChromaDB
      |
      v
classify_intent
      |
      +----------------------+
      |                      |
      v                      v
policy_question       general_question
      |                      |
      v                      v
retrieve_and_answer    direct_answer
      |                      |
      +----------+-----------+
                 |
                 v
            FinalAnswer
                 |
                 v
              FastAPI
                /ask

                ---

## LangGraph Nodes

The graph contains the required nodes:

- `classify_intent`
- `retrieve_and_answer`
- `direct_answer`

A conditional edge after `classify_intent` selects the correct path.

## Pydantic Output

The final response uses the `FinalAnswer` Pydantic model.

Fields:

- `answer`
- `sources`
- `confidence`

The confidence value is between 0 and 1.

## FastAPI

Start the API:

```powershell
uvicorn support_assistant.api:app