
import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from guardrails import (
    clean_output,
    contains_pii,
    is_restaurant_related,
    redact_pii,
    validate_input,
)

from tools import calculate_dining_cost, search_restaurants


# STEP 1: Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
)
BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
)

if not API_KEY:
    sys.exit("OPENROUTER_API_KEY is missing. Add it to the .env file.")


# STEP 2: Create model
model = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
)


# STEP 3: Define system prompt
SYSTEM_PROMPT = """
You are a Restaurant Reservation Assistant.

Rules:
- Only help with restaurant-related questions: finding restaurants,
  cuisines, locations, prices, seat availability and estimated dining costs.
- For anything unrelated, reply exactly:
  "I can only help with restaurant searches and dining-cost estimates."
- NEVER invent restaurant names, prices or seat availability.
  Only use information returned by your tools.
- Use search_restaurants to find restaurants.
- If you need a restaurant ID for a cost estimate and do not have it,
  search by restaurant name first.
- Use calculate_dining_cost for every cost estimate.
- Guest count must be between 1 and 20.
- If the user asks for more than 20 guests, explain the limit.
- Remember the restaurant selected and guest count for follow-up questions.
- For follow-ups like "make that 6 people" or "what if 8?",
  reuse the previously selected restaurant.
- If the user says "yes" after choosing a restaurant,
  ask how many guests.
- You only estimate costs. You cannot make bookings or take payments.
- Never repeat email addresses or phone numbers.
"""


# STEP 4: Create agent and memory
memory = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[search_restaurants, calculate_dining_cost],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=memory,
)


# STEP 5: Print tool activity
def print_tool_activity(new_messages):
    """Print tool names, arguments and results."""

    for message in new_messages:

        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                print(f"[TOOL CALLED]: {call['name']}")
                print(f"[ARGUMENTS]  : {call['args']}")

        elif isinstance(message, ToolMessage):
            print(f"[TOOL RESULT]: {message.content}")


# STEP 6: Run application
def main():

    thread_number = 1
    previous_count = 0

    print("LangChain Restaurant Reservation AI Agent")
    print("Type /new for a new conversation, exit to quit")

    while True:

        question = input("\nYou: ").strip()

        # Commands
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if question.lower() == "/new":
            thread_number += 1
            previous_count = 0

            print("New conversation started. Previous details are forgotten.")
            continue

        # Guardrail 1: Input validation
        error = validate_input(question)

        if error:
            print(f"[GUARDRAIL - INPUT]: {error}")
            continue

        # Guardrail 2: PII detection
        if contains_pii(question):

            print(
                "[GUARDRAIL - PII]: Email address or phone number detected. "
                "It will be redacted before processing."
            )

            question = redact_pii(question)

        # Guardrail 3: Domain restriction
        has_history = previous_count > 0

        if not is_restaurant_related(question, has_history):

            print("[GUARDRAIL - DOMAIN]: Blocked.")

            print(
                "\nAgent: I can only help with restaurant searches "
                "and dining-cost estimates."
            )

            continue

        # Conversation configuration
        config = {
            "configurable": {
                "thread_id": f"restaurant-{thread_number}"
            }
        }

        # Run agent
        try:

            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": question,
                        }
                    ]
                },
                config=config,
            )

        except Exception as error:

            print("Error:", error)
            continue

        # Get messages generated in this turn
        messages = result["messages"]
        new_messages = messages[previous_count:]

        print_tool_activity(new_messages)

        previous_count = len(messages)

        # STEP 7: Display final answer
        final_message = messages[-1]

        if isinstance(final_message, AIMessage):

            answer = clean_output(final_message.content)
            print("\nAgent:", answer)

        else:

            print("\nAgent: Sorry, I could not generate a response.")


if __name__ == "__main__":
    main()
