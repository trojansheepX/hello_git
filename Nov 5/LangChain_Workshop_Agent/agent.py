"""A minimal LangChain tool-calling agent with thread-level chat memory."""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from tools import search_workshops, calculate_workshop_fee

load_dotenv(Path(__file__).resolve().parent / ".env")
if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("Missing OPENAI_API_KEY. Create .env using .env.example.")

model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
model = ChatOpenAI(model=model_name, temperature=0)

agent = create_agent(
    model=model,
    tools=[search_workshops, calculate_workshop_fee],
    system_prompt=(
        "You are a friendly workshop enquiry assistant. "
        "Use the available tools for all workshop details and fee calculations; "
        "never invent workshop data or prices. "
        "If the workshop ID or participant count is missing from the conversation, ask for it. "
        "Use past messages in this conversation to understand follow-up questions. "
        "You cannot book workshops or take payment. Keep responses concise."
    ),
    checkpointer=InMemorySaver(),
)

thread_number = 1
thread_id = f"workshop-demo-{thread_number}"
message_count = 0

print(f"Workshop Agent | Model: {model_name}")
print("Commands: /new to begin another conversation, exit to quit")
print("Try: 'I have 4 participants. Show me Python workshops.'")

while True:
    question = input("\nYou: ").strip()
    if question.lower() == "exit":
        break
    if question.lower() == "/new":
        thread_number += 1
        thread_id = f"workshop-demo-{thread_number}"
        message_count = 0
        print("Started new conversation. Previous thread's details are not shared.")
        continue
    if not question:
        continue

    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
        )
    except Exception as exc:
        print(f"Agent request failed: {exc}")
        continue

    # Print tool activity only from this turn (not earlier turns in memory).
    messages = result["messages"]
    new_messages = messages[message_count:]
    message_count = len(messages)

    for message in new_messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                print(f"[AGENT CHOSE TOOL] {call['name']} | Arguments: {call['args']}")
        elif isinstance(message, ToolMessage):
            print(f"[TOOL RESULT] {str(message.content)[:500]}")

    print("\nAgent:", messages[-1].content)


