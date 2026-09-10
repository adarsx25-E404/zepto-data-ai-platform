"""
Zepto Support Assistant
Module 3 - LangGraph RAG workflow

Required graded baseline:
    MOCK_LLM unset or MOCK_LLM=1

Optional extension:
    MOCK_LLM=0

The graph contains:
    1. classify_intent
    2. retrieve_and_answer
    3. direct_answer

The final answer is validated with Pydantic.
"""

import json
import os
from pathlib import Path
from typing import TypedDict

import chromadb
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "zepto_support_policies"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Required offline graded mode by default.
# Set MOCK_LLM=0 only for the optional real-LLM extension.
MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"

TOP_K = 3

REAL_LLM_MODEL = "gpt-4o-mini"


# ============================================================
# EXACT REQUIRED CLASSIFICATION KEYWORDS
# ============================================================

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


# ============================================================
# PYDANTIC FINAL RESPONSE SCHEMA
# ============================================================

class FinalAnswer(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


# ============================================================
# LANGGRAPH STATE
# ============================================================

class SupportState(TypedDict, total=False):
    query: str
    intent: str

    retrieved_documents: list[str]
    retrieved_sources: list[str]

    answer: str
    sources: list[str]
    confidence: float


# ============================================================
# LOAD CHROMADB
# ============================================================

print("Loading Zepto support resources...")

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(
    f"ChromaDB records loaded: {collection.count()}"
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print(
    f"Embedding model loaded: {EMBEDDING_MODEL}"
)

print(
    f"MOCK_LLM mode: {'ON' if MOCK_LLM else 'OFF'}"
)


# ============================================================
# STRUCTURED PROMPT TEMPLATE
# ============================================================
# This template is used by the optional real-LLM generation path.
#
# Required sections:
#   ROLE
#   CONTEXT
#   TASK
#   FORMAT
#   LENGTH
#
# It also includes:
#   NEGATIVE CONSTRAINT
#   FEW-SHOT EXAMPLE
#
# Double braces are used for literal JSON braces because this
# template is later processed with .format(...).

PROMPT_TEMPLATE = """
ROLE:
You are a Zepto customer-support assistant.

CONTEXT:
Use only the policy context supplied below. Do not use outside
knowledge for policy claims.

TASK:
Answer the customer's question using the supplied policy context.

NEGATIVE CONSTRAINT:
Do not invent, assume, or add Zepto policy information that is not
present in the supplied context.

FORMAT:
Return valid JSON with exactly these fields:
- answer: string
- sources: list of chunk/document IDs
- confidence: float between 0 and 1
Do not add Markdown fences or extra fields.

LENGTH:
Keep the answer concise and directly relevant to the customer.

FEW-SHOT EXAMPLE:
Customer question:
"Can I return a product?"

Context:
"A product may be returned according to the applicable return policy."

Example answer:
{{
  "answer": "The product may be returned according to the applicable return policy.",
  "sources": ["doc_02.txt"],
  "confidence": 1.0
}}

POLICY CONTEXT:
{context}

CUSTOMER QUESTION:
{question}
"""


# ============================================================
# OPTIONAL REAL LLM HELPERS
# ============================================================

def call_real_llm(prompt: str) -> str:
    """
    Optional MOCK_LLM=0 extension.

    This branch is NOT required for the graded baseline.
    It expects an OPENAI_API_KEY and the openai package.
    """

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The optional real-LLM path requires the "
            "'openai' package. Leave MOCK_LLM unset or "
            "set MOCK_LLM=1 for the graded offline baseline."
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required only when "
            "MOCK_LLM=0."
        )

    client_llm = OpenAI(
        api_key=api_key
    )

    response = client_llm.responses.create(
        model=REAL_LLM_MODEL,
        input=prompt
    )

    return response.output_text


def validate_real_llm_answer(
    raw_output: str,
    question: str,
    context: str
) -> FinalAnswer:
    """
    Validate real-LLM JSON output.
    Retry up to 2 additional times if validation fails.
    """

    for attempt in range(3):

        try:
            parsed = json.loads(raw_output)

            validated = FinalAnswer.model_validate(
                parsed
            )

            return validated

        except Exception:

            corrective_prompt = f"""
ROLE:
You are a JSON-formatting correction assistant.

CONTEXT:
Use only the supplied policy context.

TASK:
Correct the previous response so it follows the required response schema.

NEGATIVE CONSTRAINT:
Do not invent policy information.

FORMAT:
Return ONLY valid JSON with exactly:
{{
  "answer": "string",
  "sources": ["string"],
  "confidence": 0.0
}}

Do not add Markdown fences or extra fields.

LENGTH:
Keep the answer concise.

FEW-SHOT EXAMPLE:
Question:
"How long do I have to return a product?"

Context:
"Non-perishable packaged items may be returned within 7 days."

Example answer:
{{
  "answer": "Non-perishable packaged items may be returned within 7 days.",
  "sources": ["doc_02.txt"],
  "confidence": 1.0
}}

Question:
{question}

Context:
{context}

Previous response:
{raw_output}
"""

            if attempt < 2:
                raw_output = call_real_llm(
                    corrective_prompt
                )

    return FinalAnswer(
        answer=(
            "ERROR: The optional real-LLM response "
            "could not be validated after 3 attempts."
        ),
        sources=[],
        confidence=0.0
    )


# ============================================================
# NODE 1 — CLASSIFY INTENT
# ============================================================

def classify_intent(
    state: SupportState
):
    """
    Classify:
        policy_question
        general_question

    MOCK_LLM=1:
        exact keyword heuristic required by assignment.

    MOCK_LLM=0:
        optional real-LLM classification.
    """

    query = state["query"]

    # --------------------------------------------------------
    # REQUIRED OFFLINE MOCK MODE
    # --------------------------------------------------------

    if MOCK_LLM:

        lowered = query.lower()

        if any(
            keyword in lowered
            for keyword in POLICY_KEYWORDS
        ):
            intent = "policy_question"

        else:
            intent = "general_question"

    # --------------------------------------------------------
    # OPTIONAL REAL LLM MODE
    # --------------------------------------------------------

    else:

        classification_prompt = f"""
ROLE:
You are a Zepto customer-support intent classifier.

CONTEXT:
Determine whether the customer query requires the Zepto policy corpus.

TASK:
Classify the customer query as exactly one of:
policy_question
general_question

NEGATIVE CONSTRAINT:
Return only one classification label.
Do not add explanations or extra text.

FORMAT:
Return exactly one of:
policy_question
general_question

LENGTH:
Return exactly one classification label.

FEW-SHOT EXAMPLE:
Customer query:
"What is the refund policy?"

Classification:
policy_question

Customer query:
"Hello"

Classification:
general_question

Customer query:
{query}
"""

        raw = call_real_llm(
            classification_prompt
        ).strip().lower()

        if "policy_question" in raw:
            intent = "policy_question"

        else:
            intent = "general_question"

    print(
        f"\nIntent: {intent}"
    )

    return {
        "intent": intent
    }


# ============================================================
# NODE 2 — RETRIEVE AND ANSWER
# ============================================================

def retrieve_and_answer(
    state: SupportState
):
    """
    Retrieve top-3 ChromaDB chunks.

    Retrieval always uses local embeddings + ChromaDB.

    Generation:
        MOCK_LLM=1 -> deterministic canned answer
        MOCK_LLM=0 -> optional real LLM using PROMPT_TEMPLATE
    """

    query = state["query"]

    print(
        "\nRetrieving top-3 policy chunks..."
    )

    query_embedding = embedding_model.encode(
        [query]
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=TOP_K
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    ids = results.get(
        "ids",
        [[]]
    )[0]

    if not documents:

        final = FinalAnswer(
            answer=(
                "Based on the retrieved context: "
                "No relevant policy context was found."
            ),
            sources=[],
            confidence=0.0
        )

        return {
            "retrieved_documents": [],
            "retrieved_sources": [],
            "answer": final.answer,
            "sources": final.sources,
            "confidence": final.confidence
        }

    top_chunk = documents[0]

    retrieved_sources = []

    for metadata in metadatas:

        source = metadata.get(
            "source",
            "unknown"
        )

        if source not in retrieved_sources:
            retrieved_sources.append(source)

    # --------------------------------------------------------
    # REQUIRED MOCK GENERATION
    # --------------------------------------------------------

    if MOCK_LLM:

        # Deterministic baseline answer.
        snippet = top_chunk[:200].strip()

        final = FinalAnswer(
            answer=(
                "Based on the retrieved context: "
                + snippet
            ),
            sources=ids,
            confidence=1.0
        )

    # --------------------------------------------------------
    # OPTIONAL REAL LLM GENERATION
    # --------------------------------------------------------

    else:

        context_parts = []

        for doc_id, document in zip(
            ids,
            documents
        ):

            context_parts.append(
                f"{doc_id}: {document}"
            )

        context = "\n\n".join(
            context_parts
        )

        # Required structured five-part prompt is actually used.
        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=query
        )

        raw_output = call_real_llm(
            prompt
        )

        final = validate_real_llm_answer(
            raw_output,
            query,
            context
        )

    print(
        f"Retrieved sources: {retrieved_sources}"
    )

    return {
        "retrieved_documents": documents,
        "retrieved_sources": retrieved_sources,
        "answer": final.answer,
        "sources": final.sources,
        "confidence": final.confidence
    }


# ============================================================
# NODE 3 — DIRECT ANSWER
# ============================================================

def direct_answer(
    state: SupportState
):
    """
    Answer general questions.

    MOCK_LLM=1:
        fixed canned response.

    MOCK_LLM=0:
        optional real LLM answer using the same structured
        five-part prompt pattern and JSON validation.
    """

    query = state["query"]

    # --------------------------------------------------------
    # REQUIRED MOCK GENERATION
    # --------------------------------------------------------

    if MOCK_LLM:

        final = FinalAnswer(
            answer=(
                "I can only answer questions about "
                "Zepto policies right now."
            ),
            sources=[],
            confidence=1.0
        )

    # --------------------------------------------------------
    # OPTIONAL REAL LLM GENERATION
    # --------------------------------------------------------

    else:

        context = (
            "No policy retrieval is required for this "
            "general customer question."
        )

        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=query
        )

        raw_output = call_real_llm(
            prompt
        )

        final = validate_real_llm_answer(
            raw_output,
            query,
            context
        )

    return {
        "answer": final.answer,
        "sources": final.sources,
        "confidence": final.confidence
    }


# ============================================================
# ROUTER
# ============================================================

def route_question(
    state: SupportState
):
    """
    Conditional routing after classify_intent.
    """

    if state["intent"] == "policy_question":

        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# BUILD STATEGRAPH
# ============================================================

builder = StateGraph(
    SupportState
)

builder.add_node(
    "classify_intent",
    classify_intent
)

builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

builder.add_node(
    "direct_answer",
    direct_answer
)


builder.add_edge(
    START,
    "classify_intent"
)


builder.add_conditional_edges(
    "classify_intent",
    route_question,
    {
        "retrieve_and_answer":
            "retrieve_and_answer",

        "direct_answer":
            "direct_answer"
    }
)


builder.add_edge(
    "retrieve_and_answer",
    END
)

builder.add_edge(
    "direct_answer",
    END
)


support_graph = builder.compile()


# ============================================================
# TEST
# ============================================================

def run_test(
    query: str
):
    """
    Run one complete graph request.
    """

    result = support_graph.invoke(
        {
            "query": query
        }
    )

    # Validate final result again.
    validated = FinalAnswer(
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

    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(query)

    print("\nINTENT:")
    print(result.get("intent"))

    print("\nANSWER:")
    print(validated.answer)

    print("\nSOURCES:")
    print(validated.sources)

    print("\nCONFIDENCE:")
    print(validated.confidence)

    return validated


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("ZEpto SUPPORT ASSISTANT - LANGGRAPH TEST")
    print("=" * 70)

    test_questions = [
        "How long do I have to return a product?",
        "Can I cancel my order after it has been packed?",
        "Hello"
    ]

    for question in test_questions:
        run_test(question)

    print("\n" + "=" * 70)
    print("LANGGRAPH TEST COMPLETE")
    print("=" * 70)