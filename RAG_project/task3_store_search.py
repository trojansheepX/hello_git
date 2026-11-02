"""Task 3 - Store every chunk + embedding + source in a persistent ChromaDB and search it."""
import sys

import chromadb
from sentence_transformers import SentenceTransformer

from rag_config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K, TEST_QUESTIONS
from task2_chunk_embed import build_chunks

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # We pass our own embeddings, so no embedding function is needed.
    # "hnsw:space": "cosine" makes ChromaDB return cosine distances.
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )


def build_database():
    """Chunk, embed and save all documents into ChromaDB (safe to re-run)."""
    chunks = build_chunks()
    embeddings = get_model().encode([c["text"] for c in chunks]).tolist()
    collection = get_collection()
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    print(f"Saved {collection.count()} chunks to ChromaDB at: {CHROMA_DIR}")
    return collection


def search(question: str, k: int = TOP_K) -> list[dict]:
    """Return the top-k chunks for a question with their cosine distance."""
    query_embedding = get_model().encode([question]).tolist()
    result = get_collection().query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"id": i, "text": doc, "source": meta["source"], "distance": dist}
        for i, doc, meta, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def print_results(question: str, results: list[dict]):
    print("=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)
    for rank, r in enumerate(results, start=1):
        print(f"[{rank}] {r['id']} | source: {r['source']} | cosine distance: {r['distance']:.4f}")
        print(r["text"].strip(), "\n")


def main():
    build_database()
    questions = sys.argv[1:] or TEST_QUESTIONS
    for question in questions:
        print_results(question, search(question))


if __name__ == "__main__":
    main()
