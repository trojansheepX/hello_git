"""Ask your own questions, get answers from several free OpenRouter models, and log
answers + performance metrics to rag_feedback.csv for comparing the models.

Every model gets the same retrieved chunks and is told to answer ONLY from that context.
Run:   python log_ai_feedback.py
Type a question and press Enter. Press Enter on an empty line (or type 'quit') to stop.
Optional: set OPENROUTER_MODELS=model1,model2,... in .env to choose other models.
"""
import csv
import os
import sys
from datetime import datetime
from time import perf_counter

from rag_config import PROJECT_DIR, TOP_K
from task3_store_search import build_database, print_results, search
from task4_ai_answer import API_KEY, call_openrouter

DEFAULT_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "dots-studio/dots-3-note-preview:free",
    "google/gemma-4-31b-it:free",   # often rate-limited (429) on the free tier
]
MODELS = [m.strip() for m in os.getenv("OPENROUTER_MODELS", "").split(",") if m.strip()] or DEFAULT_MODELS

CSV_PATH = PROJECT_DIR / "rag_feedback.csv"
NOT_FOUND_TEXT = "cannot find the answer"
FIELDS = [
    "timestamp", "question", "model", "status",
    "retrieved_chunks", "sources", "top_distance", "retrieval_time_s",
    "response_time_s", "prompt_tokens", "completion_tokens", "total_tokens", "tokens_per_second",
    "answer_words", "answered_from_context", "user_rating", "answer",
]


def ask_rating(model: str) -> str:
    """Optional 1-5 rating from the user; Enter skips."""
    while True:
        rating = input(f"  Rate {model} answer 1-5 (Enter to skip): ").strip()
        if rating == "" or rating in {"1", "2", "3", "4", "5"}:
            return rating
        print("  Please type a number from 1 to 5, or press Enter.")


def save_rows(rows: list[dict]):
    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def process_question(question: str, rate: bool) -> list[dict]:
    start = perf_counter()
    chunks = search(question, k=TOP_K)
    retrieval_time = round(perf_counter() - start, 3)
    print_results(question, chunks)

    rows = []
    for model in MODELS:
        print(f"Asking {model} ...")
        r = call_openrouter(question, chunks, model)
        tps = (round(r["completion_tokens"] / r["response_time_s"], 1)
               if r["completion_tokens"] and r["response_time_s"] else None)
        answered = r["status"] == "ok" and NOT_FOUND_TEXT not in r["answer"].lower()

        print(f"--- {model} | {r['status']} | {r['response_time_s']}s | "
              f"tokens: {r['total_tokens']} | {tps} tok/s")
        print(r["answer"], "\n")

        rows.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "model": model,
            "status": r["status"],
            "retrieved_chunks": "; ".join(c["id"] for c in chunks),
            "sources": "; ".join(sorted({c["source"] for c in chunks})),
            "top_distance": round(chunks[0]["distance"], 4),
            "retrieval_time_s": retrieval_time,
            "response_time_s": r["response_time_s"],
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "total_tokens": r["total_tokens"],
            "tokens_per_second": tps,
            "answer_words": len(r["answer"].split()) if r["status"] == "ok" else 0,
            "answered_from_context": answered,
            "user_rating": ask_rating(model) if rate and r["status"] == "ok" else "",
            "answer": r["answer"],
        })
    return rows


def print_summary(rows: list[dict]):
    print("=" * 80)
    print("SESSION SUMMARY (per model)")
    print("=" * 80)
    print(f"{'model':52} {'ok':>4} {'avg time s':>11} {'avg tokens':>11} {'avg rating':>11}")
    for model in MODELS:
        mrows = [r for r in rows if r["model"] == model]
        ok = [r for r in mrows if r["status"] == "ok"]
        avg = lambda vals: f"{sum(vals) / len(vals):.2f}" if vals else "-"
        print(f"{model:52} {len(ok):>2}/{len(mrows):<1} "
              f"{avg([r['response_time_s'] for r in ok]):>11} "
              f"{avg([r['total_tokens'] for r in ok if r['total_tokens']]):>11} "
              f"{avg([int(r['user_rating']) for r in ok if r['user_rating']]):>11}")


def main():
    if not API_KEY:
        sys.exit("OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key.")
    build_database()
    rate = "--no-rating" not in sys.argv
    print(f"\nModels: {', '.join(MODELS)}")
    print("Ask questions about the documents in the Documents folder.")
    print("Press Enter on an empty line (or type 'quit') to finish.\n")

    session_rows = []
    while True:
        try:
            question = input("Your question: ").strip()
        except EOFError:
            break
        if question.lower() in {"", "quit", "exit", "q"}:
            break
        rows = process_question(question, rate)
        save_rows(rows)
        session_rows.extend(rows)
        print(f"Logged {len(rows)} rows to {CSV_PATH}\n")

    if session_rows:
        print_summary(session_rows)


if __name__ == "__main__":
    main()
