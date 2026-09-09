"""
Module 3 - Document Embeddings

This script:
1. Reads the 8 support policy documents.
2. Splits each document into smaller text chunks.
3. Creates embeddings using all-MiniLM-L6-v2.
4. Stores the embeddings in ChromaDB.
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DOCS_DIR = BASE_DIR / "docs"

CHROMA_DIR = BASE_DIR / "data" / "chroma_db"


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

COLLECTION_NAME = "zepto_support_policies"

CHUNK_SIZE = 500

CHUNK_OVERLAP = 100


# ---------------------------------------------------------
# LOAD DOCUMENTS
# ---------------------------------------------------------

def load_documents():
    """Read all .txt policy documents."""

    documents = []

    policy_files = sorted(DOCS_DIR.glob("*.txt"))

    print(f"Policy documents found: {len(policy_files)}")

    for file_path in policy_files:

        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            continue

        documents.append(
            {
                "source": file_path.name,
                "text": text
            }
        )

        print(f"Loaded: {file_path.name}")

    return documents


# ---------------------------------------------------------
# SPLIT DOCUMENTS INTO CHUNKS
# ---------------------------------------------------------

def split_into_chunks(text):
    """Split text into overlapping chunks."""

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# ---------------------------------------------------------
# CREATE CHUNKS
# ---------------------------------------------------------

def create_chunks(documents):
    """Create chunks from all policy documents."""

    all_chunks = []

    for document in documents:

        chunks = split_into_chunks(
            document["text"]
        )

        for index, chunk in enumerate(chunks):

            all_chunks.append(
                {
                    "id": f"{document['source']}_{index}",
                    "text": chunk,
                    "source": document["source"]
                }
            )

    return all_chunks


# ---------------------------------------------------------
# CREATE CHROMADB COLLECTION
# ---------------------------------------------------------

def create_vector_store(chunks, model):
    """Create the ChromaDB vector store."""

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # Remove previous records so that
    # running the script again does not create duplicates.
    existing = collection.get()

    if existing["ids"]:
        collection.delete(
            ids=existing["ids"]
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    ids = [
        chunk["id"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "source": chunk["source"]
        }
        for chunk in chunks
    ]

    print("\nCreating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    return collection


# ---------------------------------------------------------
# TEST SEARCH
# ---------------------------------------------------------

def test_search(collection, model):
    """Test semantic search against the vector store."""

    question = (
        "How long do I have to return a product?"
    )

    print("\n" + "=" * 60)
    print("VECTOR SEARCH TEST")
    print("=" * 60)

    print(f"Question: {question}")

    question_embedding = model.encode(
        [question]
    )

    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=3
    )

    print("\nRetrieved policy chunks:")

    for index, document in enumerate(
        results["documents"][0],
        start=1
    ):

        source = results["metadatas"][0][index - 1]["source"]

        print(f"\nResult {index}")
        print(f"Source: {source}")
        print(f"Text: {document}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("ZEpto SUPPORT ASSISTANT - EMBEDDING PIPELINE")
    print("=" * 60)

    # 1. Load policy documents
    documents = load_documents()

    if len(documents) < 8:

        raise ValueError(
            "Expected at least 8 policy documents."
        )

    # 2. Create chunks
    chunks = create_chunks(
        documents
    )

    print(
        f"\nTotal chunks created: {len(chunks)}"
    )

    # 3. Load embedding model
    print(
        f"\nLoading embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # 4. Create ChromaDB
    collection = create_vector_store(
        chunks,
        model
    )

    print(
        f"\nChromaDB records stored: "
        f"{collection.count()}"
    )

    # 5. Test semantic search
    test_search(
        collection,
        model
    )

    print("\n" + "=" * 60)
    print("EMBEDDING PIPELINE COMPLETE")
    print("=" * 60)

    print(
        f"\nVector database saved to:"
        f"\n{CHROMA_DIR}"
    )


if __name__ == "__main__":
    main()
    