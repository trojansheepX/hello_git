
from pathlib import Path

import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENT_PATH = PROJECT_ROOT / "RAG_project" / "Documents" / "Employee_Workplace_Policy.txt"
DATABASE_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "llamaindex_brightpath"

if not DOCUMENT_PATH.exists():
    raise SystemExit(f"Document not found: {DOCUMENT_PATH}")


# STEP 1: READ DOCUMENT
documents = SimpleDirectoryReader(
    input_files=[str(DOCUMENT_PATH)]
).load_data()

print("Documents loaded:", len(documents))


# STEP 2: CREATE NODES (CHUNKS)
splitter = SentenceSplitter(
    chunk_size=180,
    chunk_overlap=35
)

nodes = splitter.get_nodes_from_documents(documents)

print("Total nodes:", len(nodes))
print("First node:", nodes[0].get_content()[:200])


# STEP 3: LOCAL EMBEDDING MODEL
embedding_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding dimensions:",
      len(embedding_model.get_text_embedding("annual leave")))


# STEP 4: CONNECT TO CHROMADB
DATABASE_PATH.mkdir(exist_ok=True)

client = chromadb.PersistentClient(path=str(DATABASE_PATH))

# Rebuild only the LlamaIndex collection on each execution.
existing_names = [c.name for c in client.list_collections()]

if COLLECTION_NAME in existing_names:
    client.delete_collection(COLLECTION_NAME)

collection = client.get_or_create_collection(COLLECTION_NAME)

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

storage = StorageContext.from_defaults(
    vector_store=vector_store
)

# STEP 5: CREATE VECTOR INDEX
index = VectorStoreIndex(
    nodes,
    storage_context=storage,
    embed_model=embedding_model
)

print("Stored nodes:", collection.count())


# STEP 6: SEMANTIC SEARCH
retriever = index.as_retriever(similarity_top_k=3)

while True:
    question = input("\nQuestion (exit to stop): ").strip()

    if question.lower() == "exit":
        break

    if not question:
        continue

    results = retriever.retrieve(question)

    for i, result in enumerate(results, start=1):
        print("\n" + "=" * 60)
        print("Result:", i)
        print("Node ID:", result.node.node_id)
        print("Relevance score:", result.score)
        print("Text:", result.node.get_content())
