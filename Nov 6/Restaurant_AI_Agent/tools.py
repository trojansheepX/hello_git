import json
from pathlib import Path

from langchain.tools import tool

from guardrails import validate_guest_count

FILE = Path(__file__).parent / "restaurants.json"


def get_restaurants():
    return json.loads(FILE.read_text(encoding="utf-8"))


def format_restaurant(item):
    return (f"{item['id']} | {item['name']} | {item['cuisine']} | {item['location']} | "
            f"INR {item['price_per_person']} per person | "
            f"{item['available_seats']} seats available")


# YOU MISSED THIS: The PDF asks for exactly TWO tools. Your extra tool
# "list_all_restaurants" dumped all 240 restaurants (~40 KB of JSON) into the
# model's context every time it was called - slow, expensive, and it fills up
# the conversation memory. Searching covers the same need, so it was removed.


# YOU MISSED THIS (3 things in this tool):
#  1. Your loop had "return" INSIDE the for-loop, so it returned only the
#     FIRST match. "Show me Indian restaurants" showed 1 restaurant out of 20.
#     Collect all matches in a list, then return.
#  2. The PDF says "accept a cuisine type OR restaurant keyword". You only
#     compared cuisine with ==, so "Spice Garden", "Hyderabad" or "garden"
#     found nothing. Use a substring check across name, cuisine and location.
#  3. The docstring and parameter names are what the MODEL reads to decide
#     when and how to call a tool. "in_cui" and a one-line docstring give it
#     very little to go on. Use clear names and describe inputs/outputs.
@tool
def search_restaurants(keyword: str) -> str:
    """Search restaurants by cuisine type, restaurant name, or city.

    Use this whenever the user asks to see restaurants, or when you need a
    restaurant's ID before calculating a cost.

    Args:
        keyword: A cuisine (e.g. "Indian"), a full or partial restaurant name
            (e.g. "Spice Garden"), or a city (e.g. "Hyderabad").
    """
    keyword = keyword.strip().lower()
    if not keyword:
        return "Please provide a cuisine, restaurant name, or city to search for."

    restaurants = get_restaurants()

    # An exact cuisine match wins, so "Indian" doesn't also return every
    # "South Indian" and "North Indian" restaurant.
    matches = [item for item in restaurants if item["cuisine"].lower() == keyword]

    if not matches:
        matches = [
            item for item in restaurants
            if keyword in item["name"].lower()
            or keyword in item["cuisine"].lower()
            or keyword in item["location"].lower()
        ]

    if not matches:
        return f"No restaurants found matching '{keyword}'."

    lines = [f"Found {len(matches)} restaurant(s) matching '{keyword}':"]
    lines += [format_restaurant(item) for item in matches]
    return "\n".join(lines)


# YOU MISSED THIS: The PDF names this tool "calculate_dining_cost"; yours was
# "calculating_dining_cost". Also:
#  - No guest-count validation (now handled by validate_guest_count).
#  - "$" was wrong - your prices are Indian rupees, so the total said "$3200"
#    for a dinner that costs INR 3200.
#  - Returning the restaurant name, price per person and guest count (not just
#    the total) puts those facts in conversation memory, so the follow-up
#    "Make that 6 people" can be recalculated reliably.
@tool
def calculate_dining_cost(restaurant_id: str, guest_count: int) -> str:
    """Calculate the estimated total dining cost for a group.

    Only estimates the cost - it does NOT book a table or take payment.
    If you only know the restaurant name, call search_restaurants first to get its ID.

    Args:
        restaurant_id: The restaurant ID, e.g. "R101".
        guest_count: Number of guests (must be between 1 and 20).
    """
    error = validate_guest_count(guest_count)
    if error:
        return error

    restaurant_id = restaurant_id.strip().upper()

    for item in get_restaurants():
        if item["id"].upper() == restaurant_id:
            price = item["price_per_person"]
            total = price * guest_count
            result = (f"Estimated cost at {item['name']} ({item['id']}): "
                      f"INR {price} x {guest_count} guests = INR {total}.")
            if guest_count > item["available_seats"]:
                result += (f" Note: only {item['available_seats']} seats are "
                           f"currently available at this restaurant.")
            return result

    return f"Sorry, no restaurant with ID '{restaurant_id}' exists."


if __name__ == "__main__":
    # Quick manual test of each tool without the LLM.
    print(search_restaurants.invoke({"keyword": "indian"}))
    print(search_restaurants.invoke({"keyword": "spice garden"}))
    print(calculate_dining_cost.invoke({"restaurant_id": "R101", "guest_count": 4}))
    print(calculate_dining_cost.invoke({"restaurant_id": "R101", "guest_count": 50}))
    print(calculate_dining_cost.invoke({"restaurant_id": "R9999", "guest_count": 5}))
