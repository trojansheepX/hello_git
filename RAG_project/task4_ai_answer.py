"""Task 4 - Send the question + retrieved chunks to an AI model (via OpenRouter) and print a grounded answer.

Needs an OpenRouter API key: put OPENROUTER_API_KEY=... in a .env file in the project folder
(see .env.example) or set it as an environment variable.
"""
import os
import sys
from time import perf_counter, sleep

import requests
from dotenv import load_dotenv

from rag_config import TEST_QUESTIONS
from task3_store_search import build_database, print_results, search

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = (
    "You answer employee questions for Cedarline Systems using ONLY the context excerpts provided. "
    "Do not use outside knowledge. If the context does not contain the answer, reply exactly: "
    "\"I cannot find the answer in the provided documents.\" "
    "Keep answers short and mention the source file you used."
)


def call_openrouter(question: str, chunks: list[dict], model: str = MODEL, retries: int = 1) -> dict:
    """Like _call_openrouter_once, but retries once after a short wait if the free model is rate-limited (429)."""
    result = _call_openrouter_once(question, chunks, model)
    for _ in range(retries):
        if result["status"] != "error 429":
            break
        sleep(5)
        result = _call_openrouter_once(question, chunks, model)
    return result


def _call_openrouter_once(question: str, chunks: list[dict], model: str) -> dict:
    """Ask one model to answer from the chunks. Returns the answer plus timing and token usage."""
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text'].strip()}" for c in chunks
    )
    result = {"model": model, "status": "ok", "answer": "", "response_time_s": 0.0,
              "prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    start = perf_counter()
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 2000,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
            },
            timeout=120,
        )
        data = response.json() if response.content else {}
    except (requests.RequestException, ValueError) as e:
        result.update(status="error", answer=f"(Request failed: {e})")
        return result
    finally:
        result["response_time_s"] = round(perf_counter() - start, 2)

    if response.status_code != 200 or "error" in data:
        message = data.get("error", {}).get("message") if isinstance(data.get("error"), dict) else data.get("error")
        result.update(status=f"error {response.status_code}",
                      answer=f"(OpenRouter error: {message or response.text[:300]})")
        return result

    usage = data.get("usage") or {}
    result["prompt_tokens"] = usage.get("prompt_tokens")
    result["completion_tokens"] = usage.get("completion_tokens")
    result["total_tokens"] = usage.get("total_tokens")
    content = (data["choices"][0]["message"].get("content") or "").strip()
    if not content:
        result.update(status="empty", answer="(The model returned an empty answer.)")
    else:
        result["answer"] = content
    return result


def ask_ai(question: str, chunks: list[dict]) -> str:
    return call_openrouter(question, chunks)["answer"]


def main():
    if not API_KEY:
        sys.exit("OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key.")
    build_database()
    questions = sys.argv[1:] or TEST_QUESTIONS
    for question in questions:
        results = search(question)
        print_results(question, results)
        print(f"AI ANSWER ({MODEL}):")
        print(ask_ai(question, results))
        print()


if __name__ == "__main__":
    main()
