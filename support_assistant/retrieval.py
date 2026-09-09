"""
Module 3 - Support Assistant
Semantic Retrieval Module

This script:
1. Loads the existing ChromaDB vector database.
2. Loads the same embedding model.
3. Converts a customer question into an embedding.
4. Retrieves the most relevant policy documents.
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

CHROMA_DIR = BASE_DIR / "data" / "chroma_db"


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

COLLECTION_NAME = "zepto_support_policies"

TOP_K = 3


# ---------------------------------------------------------
# LOAD VECTOR DATABASE
# ---------------------------------------------------------

def load_collection():
    """Load the existing ChromaDB collection."""

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            "ChromaDB database was not found. "
            "Run embeddings.py first."
        )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return collection


# ---------------------------------------------------------
# RETRIEVE RELEVANT POLICIES
# ---------------------------------------------------------

def retrieve_policies(
    question,
    collection,
    model,
    top_k=TOP_K
):
    """Retrieve the most relevant policy chunks."""

    question_embedding = model.encode(
        [question]
    )

    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=top_k
    )

    retrieved = []

    documents = results.get("documents", [[]])[0]

    metadatas = results.get("metadatas", [[]])[0]

    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        retrieved.append(
            {
                "text": document,
                "source": metadata["source"],
                "distance": distance
            }
        )

    return retrieved


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

def display_results(question, results):

    print("\n" + "=" * 60)
    print("SEMANTIC RETRIEVAL TEST")
    print("=" * 60)

    print(f"\nQuestion:")
    print(question)

    print(
        f"\nTop {len(results)} relevant policy chunks:"
    )

    for index, result in enumerate(
        results,
        start=1
    ):

        print("\n" + "-" * 60)

        print(f"Result {index}")

        print(
            f"Source: {result['source']}"
        )

        print(
            f"Distance: {result['distance']:.4f}"
        )

        print(
            f"Text:\n{result['text']}"
        )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("ZEpto SUPPORT ASSISTANT - RETRIEVAL")
    print("=" * 60)

    print("\nLoading ChromaDB...")

    collection = load_collection()

    print(
        f"Collection records: {collection.count()}"
    )

    print(
        f"\nLoading embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    question = (
        "Can I cancel my order after it has shipped?"
    )

    results = retrieve_policies(
        question,
        collection,
        model
    )

    display_results(
        question,
        results
    )

    print("\n" + "=" * 60)
    print("RETRIEVAL TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()