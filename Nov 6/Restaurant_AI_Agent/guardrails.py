# YOU MISSED THIS: The PDF asks for a file named "guardrails.py".
# Your file was "gaurdrails.py" (typo). File names in a spec are part of the
# requirement - a grader or a teammate importing "guardrails" would get an ImportError.

import json
import re
from pathlib import Path

MAX_QUESTION_LENGTH = 500
MIN_GUESTS = 1
MAX_GUESTS = 20

# YOU MISSED THIS: Your phone regex r"\b[6-9]\d{9}\b" only matched 10 digits
# typed together (9876543210). Real users type "+91 98765 43210" or
# "987-654-3210". This pattern allows an optional country code and
# spaces/dashes between digit groups.
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){9}\d(?!\w)")


# ---------------------------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------------------------
# YOU MISSED THIS: The PDF requires rejecting empty or excessively long
# questions. Your loop silently skipped empty input and had no length limit,
# so a 50,000-character paste would be sent straight to OpenAI (cost + abuse).
def validate_input(text):
    """Return an error message if the input is invalid, otherwise None."""
    if not text or not text.strip():
        return "Please type a question - empty messages are not allowed."
    if len(text) > MAX_QUESTION_LENGTH:
        return (f"Your message is too long ({len(text)} characters). "
                f"Please keep it under {MAX_QUESTION_LENGTH} characters.")
    return None


# ---------------------------------------------------------------------------
# 2. Domain restriction + 3. Context handling
# ---------------------------------------------------------------------------
# YOU MISSED THIS: Domain restriction was only written in the system prompt.
# A prompt is a *request* to the model, not a guarantee - and every blocked
# question still costs an API call. The PDF asks for it in guardrails.py, so
# we check the question in code BEFORE the agent ever sees it. The system
# prompt stays as a second layer of defence.
RESTAURANT_KEYWORDS = {
    "restaurant", "restaurants", "food", "eat", "eating", "dine", "dining",
    "dinner", "lunch", "breakfast", "brunch", "cuisine", "menu", "table",
    "seat", "seats", "seating", "available", "availability", "book",
    "booking", "reservation", "reserve", "cost", "costs", "price", "prices",
    "budget", "cheap", "expensive", "affordable", "bill", "estimate",
    "estimated", "guest", "guests", "people", "person", "persons", "pax",
    "veg", "vegetarian", "non-veg", "meal", "party", "family", "cafe",
}

# Short follow-ups like "yes", "no", "6 people", "make that 8" carry no
# restaurant keyword on their own, but make sense in an ongoing conversation.
FOLLOW_UP_PATTERN = re.compile(
    r"^\s*(yes|yeah|yep|no|nope|ok|okay|sure|thanks|thank you|please|"
    r"go ahead|that one|the first one|the second one|"
    r"make (it|that)\b.*|what if\b.*|instead\b.*|"
    r"(for )?\d+( people| guests| persons| pax)?)\s*[.!?]*\s*$",
    re.IGNORECASE,
)


def _load_restaurant_terms():
    """Restaurant names, cuisines and cities from the JSON also count as on-topic."""
    data = json.loads((Path(__file__).parent / "restaurants.json").read_text(encoding="utf-8"))
    terms = set()
    for item in data:
        terms.add(item["name"].lower())
        terms.add(item["cuisine"].lower())
        terms.add(item["location"].lower())
    return terms


RESTAURANT_TERMS = _load_restaurant_terms()


def is_follow_up(text):
    return bool(FOLLOW_UP_PATTERN.match(text))


def is_restaurant_related(text, has_history=False):
    """True if the question is about restaurants, or is a follow-up in an ongoing chat."""
    lowered = text.lower()
    words = set(re.findall(r"[a-z\-]+", lowered))

    if words & RESTAURANT_KEYWORDS:
        return True
    if any(term in lowered for term in RESTAURANT_TERMS):
        return True
    if re.search(r"\br\d{3}\b", lowered):          # restaurant IDs like R101
        return True
    if has_history and is_follow_up(text):         # "yes", "6 people", ...
        return True
    return False


# ---------------------------------------------------------------------------
# 4. Tool validation
# ---------------------------------------------------------------------------
# YOU MISSED THIS: The PDF requires rejecting guest counts < 1 or > 20.
# Your tool multiplied any number, so "50 guests" or even "-3 guests" returned
# a price. Validation belongs in code the tool calls - never trust the model
# to pass sensible arguments.
def validate_guest_count(guest_count):
    """Return an error message if the guest count is invalid, otherwise None."""
    if not isinstance(guest_count, int) or isinstance(guest_count, bool):
        return "Guest count must be a whole number."
    if guest_count < MIN_GUESTS or guest_count > MAX_GUESTS:
        return (f"Invalid guest count: {guest_count}. "
                f"Please choose between {MIN_GUESTS} and {MAX_GUESTS} guests.")
    return None


# ---------------------------------------------------------------------------
# 5. PII detection + 6. Output guardrail
# ---------------------------------------------------------------------------
def contains_pii(text):
    return bool(EMAIL_PATTERN.search(text) or PHONE_PATTERN.search(text))


def redact_pii(text):
    text = EMAIL_PATTERN.sub("[EMAIL REDACTED]", text)
    text = PHONE_PATTERN.sub("[PHONE REDACTED]", text)
    return text


def clean_output(answer):
    """Output guardrail: redact any email or phone number before display."""
    if not isinstance(answer, str):
        answer = str(answer)
    return redact_pii(answer)
