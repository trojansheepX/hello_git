"""DEMO 16: Show why RAG returned an answer: source trace + cosine distance.

No paid API needed. This is evidence traceability, not an explanation of LLM internals.
Run:  python explainability_sources.py ["your question"]
"""
import sys

from task3_store_search import build_database, search

DEFAULT_QUESTION = "How many annual leave days do full-time employees receive?"
MARKER = "Full-time employees receive"   # sentence to centre the evidence excerpt on, if present


def evidence_excerpt(text: str, marker: str = MARKER) -> str:
    """Return a short window of the chunk around the marker sentence (or the chunk start)."""
    text = " ".join(text.split())
    pos = text.lower().find(marker.lower())
    if pos >= 0:
        start, end = max(0, pos - 25), pos + 170
    else:
        start, end = 0, 180
    return ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")


def main():
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    build_database()
    results = search(question, k=3)

    print("\nQUESTION:", question)
    print("\nRETRIEVAL EVIDENCE (smaller cosine distance = closer match)")
    for rank, item in enumerate(results, start=1):
        print(f"\n{rank}. Source ID: {item['id']}")
        print("Document:", item["source"])
        print("Cosine distance:", round(item["distance"], 4))
        print("Evidence:", evidence_excerpt(item["text"]))


if __name__ == "__main__":
    main()
