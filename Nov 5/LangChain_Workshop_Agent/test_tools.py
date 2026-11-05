"""Run these local tests before using any paid OpenAI API calls."""
import json

from tools import search_workshops, calculate_workshop_fee

results = json.loads(search_workshops.invoke({"keyword": "Python"}))
assert any(row["id"] == "W101" for row in results)

fee = json.loads(calculate_workshop_fee.invoke({"workshop_id": "W101", "participants": 4}))
assert fee["total_fee_inr"] == 12000

assert "Unknown" in calculate_workshop_fee.invoke({"workshop_id": "BAD", "participants": 3})
assert "between 1 and 100" in calculate_workshop_fee.invoke({"workshop_id": "W101", "participants": 0})

print("4 tool tests passed. No AI API calls made.")
