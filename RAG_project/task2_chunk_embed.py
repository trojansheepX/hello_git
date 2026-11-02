"""Task 2 - Split documents into 120-token chunks (20-token overlap) and embed them."""
from sentence_transformers import SentenceTransformer

from rag_config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, ENCODING, chunk_text, load_documents


def build_chunks() -> list[dict]:
    """Return a list of {"id", "text", "source"} dictionaries for all documents."""
    chunks = []
    for filename, text in load_documents().items():
        for i, chunk in enumerate(chunk_text(text)):
            chunks.append({"id": f"{filename}-chunk-{i}", "text": chunk, "source": filename})
    return chunks


def main():
    chunks = build_chunks()
    print(f"Chunk size: {CHUNK_SIZE} tokens | Overlap: {CHUNK_OVERLAP} tokens")
    for source in sorted({c["source"] for c in chunks}):
        print(f"  {source}: {sum(c['source'] == source for c in chunks)} chunks")
    print(f"Total chunk count: {len(chunks)}\n")

    for c in chunks:
        print(f"--- {c['id']} ({len(ENCODING.encode(c['text']))} tokens) ---")
        print(c["text"].strip(), "\n")

    sample = chunks[0]
    print("=" * 80)
    print(f"SAMPLE CHUNK: {sample['id']}")
    print("=" * 80)
    print(sample["text"].strip(), "\n")

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode([c["text"] for c in chunks])
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Vector size: {embeddings.shape[1]}")
    print(f"First 5 values of sample vector: {embeddings[0][:5]}")


if __name__ == "__main__":
    main()
