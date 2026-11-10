
import os

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

from tools import search_workshops, calculate_fee


def create_agents():

    # Shared model client
    client = OpenAIChatClient(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    )

    # Agent 1: Find workshops
    search_agent = Agent(
        client=client,
        name="SearchAgent",
        instructions=(
            "You are a workshop search specialist. "
            "Use search_workshops to find workshops "
            "matching the user's requested topic. "
            "List the matching workshop IDs and details. "
            "Do not calculate total group fees."
        ),
        tools=[search_workshops]
    )

    # Agent 2: Calculate fees
    pricing_agent = Agent(
        client=client,
        name="PricingAgent",
        instructions=(
            "You are a workshop pricing specialist. "
            "Read the previous search results and user request. "
            "Use calculate_fee for each matching workshop "
            "using the requested number of participants. "
            "Report the verified totals. "
            "Never invent prices."
        ),
        tools=[calculate_fee]
    )

    # Agent 3: Recommend
    recommendation_agent = Agent(
        client=client,
        name="RecommendationAgent",
        instructions=(
            "You are a workshop recommendation specialist. "
            "Read the search and pricing results. "
            "Recommend the best matching workshop "
            "within the user's budget. "
            "Explain your choice briefly. "
            "Use only verified workshop data and totals."
        )
    )

    return search_agent, pricing_agent, recommendation_agent
