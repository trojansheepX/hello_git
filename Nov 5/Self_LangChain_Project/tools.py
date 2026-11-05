
import json
from pathlib import Path
from langchain.tools import tool

FILE = Path(__file__).parent / "workshops.json"
COUPON_FILE = Path(__file__).parent / "coupons.json"


def get_workshops():
    return json.loads(FILE.read_text(encoding="utf-8"))


def get_coupons():
    return json.loads(COUPON_FILE.read_text(encoding="utf-8"))


def find_coupon(code: str):
    """Return the active coupon matching the code (case-insensitive), or None."""
    code = code.strip().upper()
    for coupon in get_coupons():
        if coupon["code"].upper() == code and coupon.get("active", True):
            return coupon
    return None


@tool
def list_all_workshops() -> str:
    """List all available workshops with their id, title, duration and fee."""
    return json.dumps(get_workshops())


@tool
def find_cheapest_workshop() -> str:
    """Find the workshop with the lowest fee."""
    workshops = get_workshops()
    if not workshops:
        return "No workshops available."
    return json.dumps(min(workshops, key=lambda w: w["fee"]))


@tool
def search_workshops(keyword: str) -> str:
    """Search available workshops by title."""

    matches = [
        w for w in get_workshops()
        if keyword.lower() in w["title"].lower()
    ]

    return json.dumps(matches)


@tool
def calculate_workshop_fee(
    workshop_id: str,
    participants: int,
    coupon_code: str = ""
) -> str:
    """Calculate the total fee for a workshop and participant count.
    If the user provides a coupon code, pass it as coupon_code to apply the discount."""
    if participants < 1 or participants > 30:
        return "Participant count must be between 1 and 30."

    for workshop in get_workshops():

        if workshop["id"] == workshop_id:
            total = workshop["fee"] * participants
            result = {
                "workshop": workshop["title"],
                "participants": participants,
                "total_fee": total
            }

            if coupon_code:
                coupon = find_coupon(coupon_code)
                if coupon:
                    discount = total * coupon["discount_percent"] / 100
                    result.update({
                        "coupon_code": coupon["code"],
                        "discount_percent": coupon["discount_percent"],
                        "discount_amount": discount,
                        "final_fee": total - discount
                    })
                else:
                    result["coupon_error"] = (
                        f"Coupon '{coupon_code}' is invalid or expired. "
                        "No discount applied."
                    )
                    result["final_fee"] = total
            else:
                result["final_fee"] = total

            return json.dumps(result)

    return "Workshop not found."


if __name__ == "__main__":
    print(list_all_workshops.invoke({}))

    print(find_cheapest_workshop.invoke({}))

    print(search_workshops.invoke({"keyword": "Python"}))

    print(calculate_workshop_fee.invoke({
        "workshop_id": "W101",
        "participants": 4
    }))

    print(calculate_workshop_fee.invoke({
        "workshop_id": "W101",
        "participants": 4,
        "coupon_code": "WELCOME10"
    }))

    print(calculate_workshop_fee.invoke({
        "workshop_id": "W101",
        "participants": 4,
        "coupon_code": "EXPIRED10"
    }))
