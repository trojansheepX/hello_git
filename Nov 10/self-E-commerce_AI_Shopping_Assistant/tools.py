import json
from pathlib import Path

from agent_framework import tool

DATA_FILE = Path(__file__).parent / "products.json"

def load_products():
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))

#tool 1
@tool(approval_mode="never_require")
def search_products(topic: str) -> str:
    """Search products by name or category."""

    print(f"[TOOL CALLED] search_products: {topic}")

    topic = topic.strip().lower()

    matches = [
        product
        for product in load_products()
        if topic in product["name"].lower()
        or topic in product["category"].lower()
    ]

    return json.dumps(matches)


# Tool 2
@tool(approval_mode="never_require")
def calculate_order_total(product_id: str, quantity: int) -> str:
    """Calculate the total cost of an order if sufficient stock is available."""

    print(
        f"[TOOL CALLED] calculate_order_total: "
        f"{product_id}, {quantity}"
    )

    product_id = product_id.strip().lower()

    if quantity <= 0:
        return "Quantity must be greater than zero."

    for w in load_products():
        if product_id == w["id"].lower():

            if quantity <= w["stock"]:
                total_cost = quantity * w["price_inr"]

                return json.dumps({
                    "product_id": w["id"],
                    "quantity": quantity,
                    "total_cost_inr": total_cost
                })

            return "Sufficient stock not available."

    return "Product not available."


