"""LangChain version of the same RAG retrieval pipeline.

Reads every .txt file in Documents/, splits it with LangChain's token splitter, embeds with the
project's local all-MiniLM-L6-v2 model, stores the chunks in a SEPARATE collection inside the
project's chroma_db folder, then lets you search interactively.
Run:  python langchain_demo.py      (type 'exit' or press Enter on an empty line to stop)
"""
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_config import CHROMA_DIR, PROJECT_DIR, load_documents
from task3_store_search import get_model

COLLECTION = "langchain_cedarline"   # separate collection, so the Task 3 data is not touched
CHUNK_SIZE = 180
CHUNK_OVERLAP = 35


# Reuse our existing local embedding model (all-MiniLM-L6-v2)
class LocalEmbeddings(Embeddings):

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return get_model().encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return get_model().encode([text])[0].tolist()


def main():
    # STEP 1: READ DOCUMENTS
    documents = [
        Document(page_content=text, metadata={"source": f"Documents/{filename}"})
        for filename, text in load_documents().items()
    ]
    for doc in documents:
        print("Document loaded:", doc.metadata["source"])

    # STEP 2: CHUNK DOCUMENTS
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = f"lc_chunk_{i:03d}"

    print(f"\nChunk size: {CHUNK_SIZE} tokens | Overlap: {CHUNK_OVERLAP} tokens")
    print("Total chunks:", len(chunks))
    print("First chunk:", chunks[0].page_content[:130])

    # STEP 3: STORE IN CHROMADB (same persistent folder as Task 3, separate collection)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if COLLECTION in [item.name for item in client.list_collections()]:
        client.delete_collection(COLLECTION)

    store = Chroma.from_documents(
        documents=chunks,
        embedding=LocalEmbeddings(),
        ids=[c.metadata["chunk_id"] for c in chunks],
        collection_name=COLLECTION,
        client=client,
        collection_configuration={"hnsw": {"space": "cosine"}},
    )
    print("Stored chunks:", client.get_collection(COLLECTION).count(),
          f"in collection '{COLLECTION}' at {CHROMA_DIR.relative_to(PROJECT_DIR)}")

    # STEP 4: SEARCH AND RETRIEVE
    while True:
        try:
            question = input("\nEnter your question (exit to stop): ").strip()
        except EOFError:
            break
        if question.lower() in {"", "exit", "quit"}:
            break

        results = store.similarity_search_with_score(question, k=3)
        for document, distance in results:
            print("\n" + "-" * 60)
            print("Chunk:", document.metadata["chunk_id"])
            print("Source:", document.metadata["source"])
            print("Cosine distance:", round(distance, 4))
            print("Text:", document.page_content)


if __name__ == "__main__":
    main()
