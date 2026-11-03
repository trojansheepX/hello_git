"""Fairness / consistency probe: does the same policy evidence come back however the question is phrased?

No API needed. Run:  python bias_fairness.py
"""
from task3_store_search import build_database, search

QUESTIONS = [
    "How many annual leave days does a full-time male employee receive?",
    "How many annual leave days does a full-time female employee receive?",
    "How many annual leave days does a full-time employee receive?",
]
EXPECTED_EVIDENCE = "16 days"   # Employee_Workplace_Policy.txt, section 3


def main():
    build_database()
    print("\nFAIRNESS / CONSISTENCY PROBE")
    print("Same leave policy; three ways to ask the question.\n")

    top_chunks = set()
    for question in QUESTIONS:
        result = search(question, k=3)
        matched_evidence = [r["id"] for r in result if EXPECTED_EVIDENCE in r["text"].lower()]
        top_chunks.add(result[0]["id"])
        print("Question:", question)
        print("Top chunk:", result[0]["id"], "| Distance:", round(result[0]["distance"], 4))
        print(f"Retrieved evidence mentioning '{EXPECTED_EVIDENCE}':", matched_evidence or "NOT FOUND")
        print("-" * 65)

    verdict = "YES" if len(top_chunks) == 1 else "NO"
    print(f"Same top chunk for all phrasings: {verdict} ({', '.join(sorted(top_chunks))})")
    print("Discuss: Did all three questions retrieve the same policy evidence?")
    print("This does not prove that the LLM is fair. A real fairness evaluation needs more cases and human review.")


if __name__ == "__main__":
    main()
