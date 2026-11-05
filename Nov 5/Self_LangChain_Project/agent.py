
import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from gaurdrails import contains_pii, clean_output
from tools import (
    list_all_workshops,
    find_cheapest_workshop,
    search_workshops,
    calculate_workshop_fee,
)


# STEP 1: Load environment
load_dotenv()

# STEP 2: Create model
model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0
)

# STEP 3: Create agent# STEP 3: Create agent
agent = create_agent(
    model=model,
    tools=[
        list_all_workshops,
        find_cheapest_workshop,
        search_workshops,
        calculate_workshop_fee,
    ],
    system_prompt=(
        "You are a Workshop AI Assistant. "
        "Only answer questions about available workshops, "
        "training details and fees. "
        "Use the tools for workshop information and calculations. "
        "Never invent workshop names or prices. "
        "For unrelated questions, say: "
        "'I can only help with workshops and their fees.' "
        "If the user says yes after viewing a workshop, "
        "ask how many participants will attend. "
        "Remember the selected workshop when calculating fees. "
        "If the user gives a coupon code, pass it to the fee tool "
        "and show the discount and final fee."
    ),
    checkpointer=InMemorySaver()
)

# STEP 4: Run application
if __name__ == "__main__":

    thread_number = 1
    previous_count = 0

    print("LangChain Workshop AI Agent")
    print("Type /new for a new conversation")
    print("Type exit to quit")

    while True:
        question = input("\nYou: ").strip()

        if contains_pii(question):
            print("PII detected. Please remove personal contact details.")
            break

        if question.lower() == "exit":
            break

        if question.lower() == "/new":
            thread_number += 1
            previous_count = 0
            print("New conversation started.")
            continue

        if not question:
            continue

        config = {
            "configurable": {
                "thread_id": f"workshop-{thread_number}"
            }
        }

        try:
            result = agent.invoke(
                {
                    "messages": [
                        {"role": "user", "content": question}
                    ]
                },
                config=config
            )

            messages = result["messages"]
            new_messages = messages[previous_count:]
            previous_count = len(messages)

            for message in new_messages:

                if isinstance(message, AIMessage):
                    for call in message.tool_calls:
                        print("[TOOL]:", call["name"])
                        print("Arguments:", call["args"])

                if isinstance(message, ToolMessage):
                    print("[TOOL RESULT]:", message.content)

            print("\nAgent:", clean_output(messages[-1].content))

        except Exception as error:
            print("Error:", error)
