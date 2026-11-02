"""Shared settings and helper functions used by every task script."""
from pathlib import Path

import tiktoken

PROJECT_DIR = Path(__file__).parent
DOCUMENTS_DIR = PROJECT_DIR / "Documents"
CHROMA_DIR = PROJECT_DIR / "chroma_db"          # persistent ChromaDB folder (kept in the project)
COLLECTION_NAME = "company_documents"

CHUNK_SIZE = 120        # tokens per chunk
CHUNK_OVERLAP = 20      # tokens shared between neighbouring chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 2

# tiktoken encoding used for counting and chunking tokens
ENCODING = tiktoken.get_encoding("cl100k_base")

TEST_QUESTIONS = [
    "How many annual leave days do employees receive?",
    "Which days must employees work from the office?",
    "How quickly should a lost company laptop be reported?",
    "How quickly are Priority 1 incidents acknowledged?",
    "What is the company travel allowance per day?",
]


def load_documents() -> dict[str, str]:
    """Return {filename: text} for every .txt file in the Documents folder."""
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(DOCUMENTS_DIR.glob("*.txt"))
    }


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks of `size` tokens, each overlapping the previous one by `overlap` tokens."""
    tokens = ENCODING.encode(text)
    step = size - overlap
    chunks = []
    for start in range(0, len(tokens), step):
        chunks.append(ENCODING.decode(tokens[start:start + size]))
        if start + size >= len(tokens):
            break
    return chunks
