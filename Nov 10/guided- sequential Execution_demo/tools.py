
import json
from pathlib import Path

from agent_framework import tool


DATA_FILE = Path(__file__).parent / "workshops.json"


def load_workshops():
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


# Tool 1: Search workshops
@tool(approval_mode="never_require")
def search_workshops(topic: str) -> str:
    """Find workshops matching a topic or title."""

    print(f"[TOOL CALLED] search_workshops: {topic}")

    matches = [
        w for w in load_workshops()
        if topic.lower() in w["title"].lower()
    ]

    return json.dumps(matches)


# Tool 2: Calculate fees
@tool(approval_mode="never_require")
def calculate_fee(workshop_id: str, participants: int) -> str:
    """Calculate total workshop fee for a given group size."""

    print(f"[TOOL CALLED] calculate_fee: {workshop_id}, {participants}")

    if participants < 1 or participants > 30:
        return "Participants must be between 1 and 30."

    for w in load_workshops():
        if w["id"].upper() == workshop_id.upper():
            return json.dumps({
                "workshop": w["title"],
                "participants": participants,
                "fee_per_person": w["fee"],
                "total_fee": w["fee"] * participants,
                "currency": "INR"
            })

    return "Workshop not found."
