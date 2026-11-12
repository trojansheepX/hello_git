
import asyncio
import json
import os
from pathlib import Path
from typing import Never

from dotenv import load_dotenv
from openai import AsyncOpenAI

from agent_framework import (
    Agent,
    WorkflowBuilder,
    WorkflowContext,
    executor,
)
from agent_framework.openai import OpenAIChatClient


# =====================================================
# 1. LOAD ENVIRONMENT VARIABLES
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_DIR / ".env")


def read_setting(name: str) -> str:
    value = os.getenv(name, "").strip()

    if (
        not value
        or value.startswith("PASTE_")
        or value.startswith("YOUR_")
    ):
        raise ValueError(
            f"Set {name} correctly in your .env file"
        )

    return value


# =====================================================
# 2. BUILD MULTI-AGENT WORKFLOW
# =====================================================

def build_workflow(chat_client: OpenAIChatClient):

    # -------------------------------------------------
    # AGENT 1: TRIAGE AGENT
    # -------------------------------------------------

    triage_agent = Agent(
        client=chat_client,
        name="TriageAgent",
        instructions=(
            "You are an IT incident triage agent. "
            "Classify the incident severity. "

            "Return HIGH if the incident affects "
            "multiple employees, causes a company-wide "
            "outage, or impacts a critical business "
            "application. "

            "Return LOW if the incident is a routine "
            "issue affecting one employee. "

            "Reply with exactly one word: HIGH or LOW."
        ),
    )

    # -------------------------------------------------
    # AGENT 2: ESCALATION AGENT
    # -------------------------------------------------

    escalation_agent = Agent(
        client=chat_client,
        name="EscalationAgent",
        instructions=(
            "You are an IT escalation specialist. "

            "You handle HIGH-severity IT incidents. "

            "For every incident, provide: "
            "1. Incident summary. "
            "2. Responsible support team. "
            "3. Immediate containment actions. "
            "4. Escalation procedure. "
            "5. Communication recommendations. "

            "Keep the response short and practical. "
            "Do not claim actions have already "
            "been completed."
        ),
    )

    # -------------------------------------------------
    # AGENT 3: RESOLUTION AGENT
    # -------------------------------------------------

    resolution_agent = Agent(
        client=chat_client,
        name="ResolutionAgent",
        instructions=(
            "You are an IT help desk support agent. "

            "You handle LOW-severity IT incidents. "

            "Provide 3 to 5 simple troubleshooting "
            "steps that an employee can follow. "

            "Include when the issue should be "
            "escalated to the help desk. "

            "Keep the explanation easy to understand. "
            "Do not claim the issue is already fixed."
        ),
    )

    # =================================================
    # 3. DEFINE WORKFLOW EXECUTORS
    # =================================================

    # -------------------------------------------------
    # EXECUTOR 1: TRIAGE
    # -------------------------------------------------

    @executor(id="triage")
    async def triage(
        incident: dict,
        ctx: WorkflowContext[dict],
    ) -> None:

        print("\n[Triage Agent] Analyzing incident...")

        result = await triage_agent.run(
            "Classify this IT incident. "
            "Reply HIGH or LOW only:\n"
            + json.dumps(
                incident,
                ensure_ascii=False,
            )
        )

        severity = (
            result.text
            .strip()
            .upper()
            .strip(". ")
        )

        if severity not in {"HIGH", "LOW"}:
            raise ValueError(
                "Unexpected TriageAgent response: "
                f"{result.text!r}"
            )

        print(
            f"[Triage Agent] Severity: {severity}"
        )

        # Forward incident and severity to the graph
        await ctx.send_message({
            **incident,
            "severity": severity,
        })

    # -------------------------------------------------
    # EXECUTOR 2: ESCALATION
    # -------------------------------------------------

    @executor(id="escalation")
    async def escalation(
        incident: dict,
        ctx: WorkflowContext[Never, str],
    ) -> None:

        print(
            "\n[Escalation Agent] "
            "Handling HIGH-severity incident..."
        )

        result = await escalation_agent.run(
            "Prepare an IT escalation plan for:\n"
            + json.dumps(
                incident,
                ensure_ascii=False,
            )
        )

        output = (
            f"\nIncident ID: {incident['id']}\n"
            f"Severity: HIGH\n"
            f"Assigned Agent: EscalationAgent\n\n"
            f"{result.text.strip()}"
        )

        await ctx.yield_output(output)

    # -------------------------------------------------
    # EXECUTOR 3: RESOLUTION
    # -------------------------------------------------

    @executor(id="resolution")
    async def resolution(
        incident: dict,
        ctx: WorkflowContext[Never, str],
    ) -> None:

        print(
            "\n[Resolution Agent] "
            "Handling LOW-severity incident..."
        )

        result = await resolution_agent.run(
            "Prepare troubleshooting steps for:\n"
            + json.dumps(
                incident,
                ensure_ascii=False,
            )
        )

        output = (
            f"\nIncident ID: {incident['id']}\n"
            f"Severity: LOW\n"
            f"Assigned Agent: ResolutionAgent\n\n"
            f"{result.text.strip()}"
        )

        await ctx.yield_output(output)

    # =================================================
    # 4. CREATE GRAPH WITH CONDITIONAL ROUTING
    # =================================================

    workflow = (
        WorkflowBuilder(start_executor=triage)

        # Route HIGH incidents to EscalationAgent
        .add_edge(
            triage,
            escalation,
            condition=lambda msg:
                msg["severity"] == "HIGH",
        )

        # Route LOW incidents to ResolutionAgent
        .add_edge(
            triage,
            resolution,
            condition=lambda msg:
                msg["severity"] == "LOW",
        )

        .build()
    )

    return workflow


# =====================================================
# 5. MAIN APPLICATION
# =====================================================

async def main() -> None:

    print("=" * 60)
    print("MICROSOFT AGENT FRAMEWORK")
    print("IT INCIDENT ROUTING SYSTEM")
    print("Authentication: Azure Foundry API Key")
    print("=" * 60)

    # Load Azure Foundry credentials

    api_key = read_setting(
        "AZURE_OPENAI_API_KEY"
    )

    base_url = read_setting(
        "AZURE_OPENAI_BASE_URL"
    )

    deployment = read_setting(
        "AZURE_OPENAI_MODEL"
    )

    print("\nAzure Foundry configuration loaded.")
    print(f"Endpoint: {base_url}")
    print(f"Deployment: {deployment}")

    # -------------------------------------------------
    # LOAD INCIDENT DATA
    # -------------------------------------------------

    incidents_file = (
        PROJECT_DIR / "incidents.json"
    )

    with incidents_file.open(
        encoding="utf-8"
    ) as file:
        incidents = json.load(file)

    print(
        f"\nTotal incidents loaded: {len(incidents)}"
    )

    # -------------------------------------------------
    # CONNECT TO AZURE FOUNDRY
    # -------------------------------------------------

    async with AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    ) as sdk_client:

        chat_client = OpenAIChatClient(
            model=deployment,
            async_client=sdk_client,
        )

        print(
            "\nAzure Foundry client initialized."
        )

        # ---------------------------------------------
        # BUILD WORKFLOW
        # ---------------------------------------------

        workflow = build_workflow(chat_client)

        print(
            "Graph workflow built successfully."
        )

        # ---------------------------------------------
        # PROCESS INCIDENTS
        # ---------------------------------------------

        for incident in incidents:

            print("\n" + "=" * 60)

            print(
                f"Processing Incident: {incident['id']}"
            )

            print(
                f"Description: {incident['description']}"
            )

            print("=" * 60)

            # Run the workflow
            result = await workflow.run(incident)

            # Display outputs
            for output in result.get_outputs():

                print("\n" + "-" * 60)
                print("FINAL INCIDENT RESPONSE")
                print("-" * 60)

                print(output)

                print("-" * 60)

        print("\n" + "=" * 60)
        print("ALL INCIDENTS PROCESSED")
        print("=" * 60)


# =====================================================
# 6. APPLICATION ENTRY POINT
# =====================================================

if __name__ == "__main__":
    asyncio.run(main())
