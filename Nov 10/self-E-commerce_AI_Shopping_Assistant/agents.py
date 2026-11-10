
import os

from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient

from tools import search_products, calculate_order_total


def create_agents():

    # Shared model client.
    # Use Chat Completions (stateless) — OpenRouter's Responses endpoint
    # does not support previous_response_id, which breaks tool calling.
    client = OpenAIChatCompletionClient(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=os.getenv(
            "OPENAI_CHAT_MODEL",
            "nvidia/nemotron-3-ultra-550b-a55b:free"
        ),
        base_url=os.getenv(
            "OPENAI_BASE_URL",
            "https://openrouter.ai/api/v1"
        ),
    )

    # Agent 1: Find workshops
    search_agent = Agent(
        client=client,
        name="InventoryAgent",
        instructions=(
            "You are a E-commerce search specialist. "
            "Use search_products to find products "
            "matching the user's requested topic. "
            "List the matching product IDs and details. "
            "Do not calculate total order price."
        ),
        tools=[search_products]
    )

    # Agent 2: Calculate fees
    pricing_agent = Agent(
        client=client,
        name="PricingAgent",
        instructions=(
            "You are a products pricing specialist. "
            "Read the previous search results and user request. "
            "Use calcualate_order_total for each matching product "
            "using the requested number of quantity. "
            "Report the verified totals. "
            "Never invent prices."
        ),
        tools=[calculate_order_total]
    )

    # Agent 3: Recommend
    recommendation_agent = Agent(
        client=client,
        name="RecommendationAgent",
        instructions=(
            "You are a E-commerce products recommendation specialist. "
            "Read the search and pricing results. "
            "Recommend the best matching product "
            "within the user's budget. "
            "Explain your choice briefly. "
            "Use only verified product data and totals."
        )
    )

    return search_agent, pricing_agent, recommendation_agent
