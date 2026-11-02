"""Lesson 8: evaluate whether the correct evidence appears in the top 3 retrieved chunks."""
import csv
from time import perf_counter

from rag_config import PROJECT_DIR
from task3_store_search import build_database, search

# (question, text that must appear in the retrieved chunks)
TESTS = [
    ("How many annual leave days do full-time employees receive?", "16 days of paid annual leave"),
    ("Which days are mandatory office days?", "Tuesdays and Thursdays"),
    ("How many paid sick leave days are provided?", "six paid sick-leave days"),
    ("When must a lost laptop be reported?", "within one hour"),
    ("What is the response target for a Priority 1 incident?", "within 30 minutes"),
]
TOP_K = 3
CSV_PATH = PROJECT_DIR / "benchmark_results.csv"


def main():
    build_database()  # makes sure ChromaDB is filled and the embedding model is loaded before timing

    rows = []
    for question, expected_text in TESTS:
        start = perf_counter()
        matches = search(question, k=TOP_K)
        seconds = perf_counter() - start

        retrieved_text = " ".join(m["text"] for m in matches)
        found = expected_text.lower() in retrieved_text.lower()
        rows.append({
            "question": question,
            "expected_evidence": expected_text,
            "evidence_found": found,
            "top_chunk": matches[0]["id"],
            "top_distance": round(matches[0]["distance"], 4),
            "time_seconds": round(seconds, 2),
        })
        print(f"{'PASS' if found else 'FAIL'} | {question}")
        print(f"  First match: {matches[0]['id']} | distance: {matches[0]['distance']:.4f} | time: {seconds:.2f}s")

    score = sum(row["evidence_found"] for row in rows)
    print(f"\nRetrieval evidence found: {score}/{len(TESTS)} ({score / len(TESTS):.0%})")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", CSV_PATH)
    print("Note: this checks whether evidence is retrieved, not whether the AI answer is correct.")


if __name__ == "__main__":
    main()
