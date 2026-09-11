"""
Vercel deployment entry point for the Zepto Customer Support Assistant.

This is a lightweight deployment adapter.

The full capstone RAG implementation remains in:
    support_assistant/graph.py

The Vercel version intentionally avoids importing:
    - sentence-transformers
    - torch
    - chromadb
    - langgraph

This keeps the serverless function small enough for Vercel.
"""

from pathlib import Path
import re

from fastapi import FastAPI
from pydantic import BaseModel, Field


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Zepto Customer Support Assistant",
    description=(
        "Lightweight Vercel deployment of the Zepto "
        "Customer Support Assistant."
    ),
    version="1.0.0",
)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class AskRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Customer's support question",
    )


class FinalAnswer(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


# ============================================================
# POLICY DOCUMENTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "support_assistant" / "docs"


POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


def load_policy_documents():
    """
    Load the small policy text files.

    This deployment adapter uses simple lexical retrieval
    instead of the full SentenceTransformer + ChromaDB stack.
    """

    documents = []

    if not DOCS_DIR.exists():
        return documents

    for path in sorted(DOCS_DIR.glob("*.txt")):
        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).strip()

            if text:
                documents.append(
                    {
                        "id": path.name,
                        "text": text,
                    }
                )

        except Exception:
            continue

    return documents


POLICY_DOCUMENTS = load_policy_documents()


# ============================================================
# LIGHTWEIGHT RETRIEVAL
# ============================================================

def tokenize(text: str) -> set[str]:
    """
    Convert text into lowercase word tokens.
    """

    return set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )
    )


def retrieve_policy_documents(
    query: str,
    top_k: int = 3,
):
    """
    Lightweight lexical retrieval.

    Scores policy documents using word overlap.
    """

    query_tokens = tokenize(query)

    scored = []

    for document in POLICY_DOCUMENTS:
        document_tokens = tokenize(
            document["text"]
        )

        overlap = query_tokens.intersection(
            document_tokens
        )

        score = len(overlap)

        if score > 0:
            scored.append(
                (
                    score,
                    document,
                )
            )

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        document
        for _, document in scored[:top_k]
    ]


# ============================================================
# INTENT CLASSIFICATION
# ============================================================

def classify_intent(query: str) -> str:
    """
    Match the original project's required offline
    keyword classification behavior.
    """

    lowered = query.lower()

    if any(
        keyword in lowered
        for keyword in POLICY_KEYWORDS
    ):
        return "policy_question"

    return "general_question"


# ============================================================
# HEALTH / HOME
# ============================================================

@app.get("/")
def home():
    return {
        "message": (
            "Zepto Customer Support Assistant "
            "is running."
        ),
        "deployment": "Vercel",
        "mode": "lightweight",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "documents_loaded": len(
            POLICY_DOCUMENTS
        ),
    }


# ============================================================
# ASK ENDPOINT
# ============================================================

@app.post(
    "/ask",
    response_model=FinalAnswer,
)
def ask_question(
    request: AskRequest,
):
    query = request.query.strip()

    intent = classify_intent(query)

    # --------------------------------------------------------
    # GENERAL QUESTION
    # --------------------------------------------------------

    if intent == "general_question":
        return FinalAnswer(
            answer=(
                "I can only answer questions about "
                "Zepto policies right now."
            ),
            sources=[],
            confidence=1.0,
        )

    # --------------------------------------------------------
    # POLICY QUESTION
    # --------------------------------------------------------

    retrieved = retrieve_policy_documents(
        query,
        top_k=3,
    )

    if not retrieved:
        return FinalAnswer(
            answer=(
                "Based on the available policy documents, "
                "no relevant policy context was found."
            ),
            sources=[],
            confidence=0.0,
        )

    top_document = retrieved[0]

    # Keep response concise.
    snippet = top_document["text"][:500].strip()

    sources = [
        document["id"]
        for document in retrieved
    ]

    return FinalAnswer(
        answer=(
            "Based on the retrieved policy context: "
            + snippet
        ),
        sources=sources,
        confidence=1.0,
    )