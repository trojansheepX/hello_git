"""Library tools that the AI agent can call.

search_books()  - keyword search over books.json
reserve_book()  - reserve an available copy, after the user confirms
"""

import json
import re
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BOOKS_FILE = BASE_DIR / "books.json"
RESERVATIONS_FILE = BASE_DIR / "reservations.json"

# Words that carry no meaning for a search ("books related to python" -> "python").
STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "with", "about",
    "book", "books", "related", "is", "are", "any", "me", "show", "find", "available",
}

# Common abbreviations users type that won't literally appear in the data.
SYNONYMS = {
    "ai": ["artificial", "intelligence"],
    "ml": ["machine", "learning"],
}


# ---------------------------------------------------------------- JSON helpers
def _load_json(path: Path) -> list:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        content = f.read().strip()
    return json.loads(content) if content else []


def _save_json(path: Path, data: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _tokens(text: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP_WORDS]
    expanded = []
    for w in words:
        expanded.extend(SYNONYMS.get(w, [w]))
    return expanded


def _availability(book: dict) -> dict:
    copies = book["available_copies"]
    return {**book, "status": "Available" if copies > 0 else "Not available"}


# ---------------------------------------------------------------- Tool 1
def search_books(query: str) -> dict:
    """Search books by title, author or category keyword.

    Books are ranked by how many keywords they match, so
    "Python programming" returns Python books before other programming books.
    """
    books = _load_json(BOOKS_FILE)
    query = (query or "").strip()
    if not query:
        return {"query": query, "count": 0, "books": [], "message": "Please provide a search keyword."}

    phrase = query.lower()
    keywords = _tokens(query) or [phrase]

    scored = []
    for book in books:
        haystack = f"{book['title']} {book['author']} {book['category']}".lower()
        haystack_words = set(re.findall(r"[a-z0-9]+", haystack))
        score = sum(1 for k in keywords if k in haystack_words)
        if phrase in book["title"].lower():
            score += 10  # exact title / title-fragment match wins
        if score:
            scored.append((score, book))

    scored.sort(key=lambda item: (-item[0], item[1]["id"]))
    matches = [_availability(book) for _, book in scored]

    return {
        "query": query,
        "count": len(matches),
        "books": matches,
        "message": f"Found {len(matches)} matching book(s)." if matches else f"No books found for '{query}'.",
    }


# ---------------------------------------------------------------- Tool 2
def _ask_terminal(prompt: str) -> bool:
    answer = input(f"{prompt} (yes/no): ").strip().lower()
    return answer in {"y", "yes"}


def reserve_book(book_id: int, customer_name: str, confirm=_ask_terminal) -> dict:
    """Reserve one copy of a book for a customer.

    The reservation is only saved if a copy is available AND the user approves
    it via `confirm` (by default, a yes/no prompt in the terminal).
    """
    customer_name = (customer_name or "").strip()
    if not customer_name:
        return {"success": False, "message": "A customer name is required to make a reservation."}

    try:
        book_id = int(book_id)
    except (TypeError, ValueError):
        return {"success": False, "message": f"Invalid book ID: {book_id!r}."}

    books = _load_json(BOOKS_FILE)
    book = next((b for b in books if b["id"] == book_id), None)
    if book is None:
        return {"success": False, "message": f"No book found with ID {book_id}."}

    if book["available_copies"] <= 0:
        return {
            "success": False,
            "book_id": book_id,
            "title": book["title"],
            "available_copies": 0,
            "message": f"'{book['title']}' has no available copies. No reservation was created.",
        }

    approved = confirm(
        f"\nConfirm reservation of '{book['title']}' (ID {book_id}) for {customer_name}? "
        f"[{book['available_copies']} cop{'y' if book['available_copies'] == 1 else 'ies'} available]"
    )
    if not approved:
        return {
            "success": False,
            "book_id": book_id,
            "title": book["title"],
            "message": "Reservation cancelled by the user. Nothing was saved.",
        }

    reservations = _load_json(RESERVATIONS_FILE)
    next_id = max((r["reservation_id"] for r in reservations), default=0) + 1
    reservation = {
        "reservation_id": next_id,
        "book_id": book_id,
        "title": book["title"],
        "customer_name": customer_name,
        "reserved_at": datetime.now().isoformat(timespec="seconds"),
    }
    reservations.append(reservation)

    book["available_copies"] -= 1
    _save_json(BOOKS_FILE, books)
    _save_json(RESERVATIONS_FILE, reservations)

    return {
        "success": True,
        "reservation": reservation,
        "remaining_copies": book["available_copies"],
        "message": f"'{book['title']}' reserved for {customer_name}.",
    }
