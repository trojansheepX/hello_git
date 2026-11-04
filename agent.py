"""AI Library Assistant.

The model (via OpenRouter, OpenAI-compatible API) decides which Python tool
to call; we execute it, send the result back, and print the final answer.

Run:  python agent.py
"""

import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from tools import reserve_book, search_books

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

if not API_KEY:
    sys.exit("OPENROUTER_API_KEY is missing. Add it to the .env file.")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = """You are a helpful AI Library Assistant.

You have two tools:
- search_books: find books by title, author, category or keyword, and check availability.
- reserve_book: reserve a book by its numeric book ID for a named customer.

Rules:
- Always use the tools; never invent books, IDs or availability.
- Availability changes after every reservation, so call search_books again for every
  availability question instead of relying on earlier results.
- To reserve a book given by title, first call search_books to get its ID, then call reserve_book.
- If no customer name is given for a reservation, ask the user for it instead of guessing.
- The reserve_book tool asks the user for confirmation itself; just call it. If it reports
  the user declined or no copies are available, explain that clearly.
- Keep answers short and friendly. List books with ID, title, author and available copies."""

# Tool schemas the model sees (OpenAI function-calling format).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_books",
            "description": "Search the library catalogue by book title, author, category or keyword. "
                           "Returns matching books with their availability.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Book title or keyword, e.g. 'Python Crash Course' or 'artificial intelligence'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reserve_book",
            "description": "Reserve one available copy of a book for a customer. Requires the numeric book ID "
                           "(use search_books first if you only know the title). The user is asked to confirm.",
            "parameters": {
                "type": "object",
                "properties": {
                    "book_id": {"type": "integer", "description": "The numeric ID of the book, e.g. 101."},
                    "customer_name": {"type": "string", "description": "Name of the customer reserving the book."},
                },
                "required": ["book_id", "customer_name"],
            },
        },
    },
]

AVAILABLE_FUNCTIONS = {
    "search_books": search_books,
    "reserve_book": reserve_book,
}

MAX_TOOL_ROUNDS = 5


def run_tool(name: str, raw_args: str) -> dict:
    func = AVAILABLE_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"Unknown tool '{name}'."}
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        return {"error": f"Could not parse arguments: {raw_args}"}
    try:
        return func(**args)
    except TypeError as exc:
        return {"error": f"Bad arguments for {name}: {exc}"}


def ask_agent(messages: list) -> str:
    """Run one user turn: let the model call tools until it produces an answer."""
    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message

        if not message.tool_calls:
            messages.append({"role": "assistant", "content": message.content or ""})
            return message.content or "(no response)"

        # Record the model's tool request in the conversation.
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in message.tool_calls
            ],
        })

        # Execute each requested tool and send the result back.
        for call in message.tool_calls:
            print(f"\n[Agent] Selected tool: {call.function.name}  args={call.function.arguments}")
            result = run_tool(call.function.name, call.function.arguments)
            print(f"[Agent] Tool result: {result.get('message', result.get('error', 'done'))}")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    return "Sorry, I couldn't complete that request (too many tool steps)."


def main() -> None:
    print("=" * 60)
    print(" AI Library Assistant")
    print(f" Model: {MODEL}")
    print(" Type your question, or 'exit' to quit.")
    print("=" * 60)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break

        checkpoint = len(messages)
        messages.append({"role": "user", "content": user_input})
        try:
            answer = ask_agent(messages)
        except Exception as exc:  # network / API errors shouldn't kill the session
            del messages[checkpoint:]  # drop the half-finished turn
            print(f"\n[Error] {exc}")
            continue
        print(f"\nAssistant: {answer}")


if __name__ == "__main__":
    main()
