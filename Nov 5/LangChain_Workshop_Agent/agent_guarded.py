"""Workshop-only LangChain agent: domain guardrails + contextual number replies.

Place this file alongside tools.py, workshops.json, and .env.
Run: python agent_guarded.py
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from tools import search_workshops, calculate_workshop_fee


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY is missing from .env")

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
REFUSAL = "I can only help with available workshops and their fees."

model = ChatOpenAI(model=MODEL_NAME, temperature=0)

agent = create_agent(
    model=model,
    tools=[search_workshops, calculate_workshop_fee],
    system_prompt=(
        "You are ONLY an assistant for workshops in workshops.json. "
        "You may search available workshops, describe their details, calculate fees, "
        "and remember the workshop and participant count from this conversation. "
        "Use the tools for workshop facts and ALL fee calculations. "
        "Do not answer unrelated or general-knowledge questions. "
        f"For off-topic requests reply exactly: {REFUSAL} "
        "When the user replies with a number to your question about participants, "
        "treat that number as the participant count, and use the fee calculation "
        "tool for the previously discussed workshop. "
        "Ask a clarifying question if the workshop ID or participant count is unknown. "
        "Do not invent workshop names or prices."
    ),
    checkpointer=InMemorySaver(),
)


def is_short_reply(question: str) -> bool:
    return " ".join(question.lower().split()) in {
        "yes", "yeah", "yep", "sure", "no", "no thanks", "okay", "ok",
        "please do", "go ahead", "thank you", "thanks",
    }


def is_memory_question(question: str) -> bool:
    return bool(re.search(
        r"\b(what workshop|which workshop|how many participants|"
        r"what did we discuss|what were we discussing|"
        r"what did i say|remind me)\b", question.lower(),
    ))


def participant_count_reply(question: str) -> bool:
    """Accept simple counts ONLY if the agent requested this information."""
    q = " ".join(question.lower().strip().split())
    match = re.fullmatch(
        r"(?:(?:we have|there are|it is|make it)\s+)?"
        r"(\d{1,3})(?:\s+(?:participants?|attendees?|people|students?))?",
        q,
    )
    return bool(match and 1 <= int(match.group(1)) <= 100)


def asks_for_participant_count(answer: str) -> bool:
    """Track a specific question the assistant asked, not just any active chat."""
    text = answer.lower()
    return bool(re.search(
        r"(?:how many|number of)\s+(?:\w+\s+){0,3}"
        r"(?:participants?|attendees?|people|students?)\b",
        text,
    ) or re.search(
        r"\b(?:participants?|attendees?|people|students?)\b.{0,60}"
        r"(?:how many|number of)",
        text,
    )) and "?" in text


def is_workshop_request(
    question: str,
    conversation_active: bool,
    awaiting_participants: bool,
) -> bool:
    q = " ".join(question.lower().split())

    if not q or len(q) > 500:
        return False

    # A bare number is allowed only after the agent has asked for a count.
    if awaiting_participants and participant_count_reply(q):
        return True

    # Explicit workshop requests.
    if re.search(r"\b(workshops?|trainings?|courses?)\b", q):
        return True

    # E.g., 'Find Python' or 'Show Azure'.
    search_intent = re.search(r"\b(show|find|search|list|recommend|available|offer)\b", q)
    topic = re.search(r"\b(python|azure|cloud|automation|data science|ai|power bi)\b", q)
    if search_intent and topic:
        return True

    if conversation_active:
        if is_short_reply(q) or is_memory_question(q):
            return True
        if re.search(r"\b(fees?|cost|price|total|participants?|attendees?|people|team)\b", q):
            return True
        if re.search(r"\b(make that|change|increase|decrease|instead|same)\b", q):
            return True

    return False


if __name__ == "__main__":
    thread_number = 1
    thread_id = f"workshop-demo-{thread_number}"
    shown_message_count = 0
    conversation_active = False
    awaiting_participants = False

    print(f"Workshop Agent (guarded) | Model: {MODEL_NAME}")
    print("Commands: /new for a new conversation, exit to quit")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if question.lower() == "/new":
            thread_number += 1
            thread_id = f"workshop-demo-{thread_number}"
            shown_message_count = 0
            conversation_active = False
            awaiting_participants = False
            print("New conversation started.")
            continue

        if not question:
            continue

        was_awaiting_count = awaiting_participants
        if not is_workshop_request(question, conversation_active, awaiting_participants):
            print("[INPUT GUARDRAIL] Blocked: outside workshop domain")
            print("Agent:", REFUSAL)
            continue

        config = {"configurable": {"thread_id": thread_id}}

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config=config,
            )

            messages = result["messages"]
            new_messages = messages[shown_message_count:]
            shown_message_count = len(messages)

            tool_used = False
            for message in new_messages:
                if isinstance(message, AIMessage):
                    for call in message.tool_calls:
                        print(f"[AGENT CHOSE TOOL] {call['name']} | Arguments: {call['args']}")
                elif isinstance(message, ToolMessage):
                    tool_used = True
                    print("[TOOL RESULT]", str(message.content)[:1000])

            answer = str(messages[-1].content)
            # 'yes', memory questions, and genuine clarifying questions may
            # produce no tool call. A calculated fee must still use a tool.
            safe_clarification = asks_for_participant_count(answer)
            safe_follow_up = conversation_active and (
                is_short_reply(question) or is_memory_question(question)
            )

            if tool_used or safe_clarification or safe_follow_up:
                print("\nAgent:", answer)
                conversation_active = True
                awaiting_participants = safe_clarification
            else:
                # If the model failed to use a tool for the count, do NOT show
                # an unverified fee. Keep the pending slot so user can retry.
                if was_awaiting_count and participant_count_reply(question):
                    print("[OUTPUT GUARDRAIL] Fee must be verified with the tool")
                    print("Agent: I couldn't verify the fee. Please try the participant count again.")
                    awaiting_participants = True
                else:
                    print("[OUTPUT GUARDRAIL] No tool evidence for this response")
                    print("Agent:", REFUSAL)

        except Exception as error:
            print("Agent error:", error)
