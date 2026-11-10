
import asyncio

from dotenv import load_dotenv
from agent_framework import AgentResponse
from agent_framework.orchestrations import SequentialBuilder

from agents import create_agents


load_dotenv()


async def main():

    # Step 1: Create agents
    search_agent, pricing_agent, recommendation_agent = create_agents()

    # Step 2: Build sequential workflow
    workflow = SequentialBuilder(
        participants=[
            search_agent,
            pricing_agent,
            recommendation_agent
        ],
        output_from="all"
    ).build()

    # Step 3: Get user request
    question = input("You: ").strip()

    if not question:
        return

    # Step 4: Execute workflow
    result = await workflow.run(question)

    # Step 5: Display each agent's response
    for output in result.get_outputs():

        if isinstance(output, AgentResponse):
            for message in output.messages:
                print(f"\n--- {message.author_name or 'Agent'} ---")
                print(message.text)


if __name__ == "__main__":
    asyncio.run(main())
