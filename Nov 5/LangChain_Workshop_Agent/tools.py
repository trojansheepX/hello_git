"""Two simple, read-only LangChain tools that use a local JSON catalog."""
import json
import re
from pathlib import Path

from langchain.tools import tool

DATA_FILE = Path(__file__).resolve().parent / "workshops.json"


def load_workshops() -> list[dict]:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


@tool
def search_workshops(keyword: str) -> str:
    """Find workshop IDs, subjects, durations and fees in the catalog. Use 'all' to list every workshop."""
    catalog = load_workshops()
    query = keyword.strip().lower()
    if not query or query in {"all", "list", "available", "workshops"}:
        matches = catalog
    else:
        words = [w for w in re.findall(r"[a-z0-9]+", query) if len(w) > 2]
        matches = [
            workshop for workshop in catalog
            if any(word in (workshop["title"] + " " + " ".join(workshop["topics"])).lower() for word in words)
        ]

    if not matches:
        return "No matching workshops found. Ask to list all workshops."
    return json.dumps(matches, ensure_ascii=False, indent=2)


@tool
def calculate_workshop_fee(workshop_id: str, participants: int) -> str:
    """Calculate total workshop fee in INR from workshop ID and number of participants. Does not create bookings or take payments."""
    if not 1 <= participants <= 100:
        return "Participant count must be between 1 and 100."

    workshop = next(
        (w for w in load_workshops() if w["id"].upper() == workshop_id.strip().upper()),
        None,
    )
    if workshop is None:
        return "Unknown workshop ID. Use search_workshops to find the correct ID."

    total = workshop["fee_per_person_inr"] * participants
    return json.dumps({
        "workshop_id": workshop["id"],
        "title": workshop["title"],
        "participants": participants,
        "fee_per_person_inr": workshop["fee_per_person_inr"],
        "total_fee_inr": total,
    }, ensure_ascii=False, indent=2)
