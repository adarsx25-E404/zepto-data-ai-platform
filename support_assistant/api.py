"""
Zepto Customer Support Assistant - FastAPI

Required endpoint:
    POST /ask

Request:
    {"query": "..."}

Response:
    {
        "answer": "...",
        "sources": [...],
        "confidence": 1.0
    }
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from support_assistant.graph import (
    support_graph,
    FinalAnswer
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Zepto Customer Support Assistant",
    description=(
        "Offline deterministic support assistant "
        "using LangGraph, ChromaDB and "
        "all-MiniLM-L6-v2."
    ),
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class AskRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Customer's support question"
    )


# ============================================================
# HEALTH / HOME
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Zepto Customer Support Assistant is running."
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# ASK ENDPOINT
# ============================================================

@app.post(
    "/ask",
    response_model=FinalAnswer
)
def ask_question(
    request: AskRequest
):

    result = support_graph.invoke(
        {
            "query": request.query
        }
    )

    validated_response = FinalAnswer(
        answer=result.get(
            "answer",
            ""
        ),
        sources=result.get(
            "sources",
            []
        ),
        confidence=result.get(
            "confidence",
            0.0
        )
    )

    return validated_response