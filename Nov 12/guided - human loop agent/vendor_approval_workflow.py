
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Never

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agent_framework import (
    Agent,
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    executor,
    handler,
    response_handler,
)
from agent_framework.openai import OpenAIChatClient


# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


@dataclass
class ManagerApprovalRequest:
    purchase: dict
    prompt: str


# =====================================================
# 2. HUMAN-IN-THE-LOOP APPROVAL GATEWAY
# =====================================================

class ApprovalGateway(Executor):

    def __init__(self):
        super().__init__(id="approval_gateway")

    @handler
    async def review(
        self,
        purchase: dict,
        ctx: WorkflowContext[dict],
    ) -> None:

        needs_approval = (
            purchase["amount_inr"] >= 100_000
            or purchase["risk"] == "HIGH"
        )

        if not needs_approval:

            purchase["decision"] = "AUTO_APPROVED"
            purchase["decision_by"] = "Demo approval policy"

            print("  ApprovalGateway: automatic approval")

            await ctx.send_message(purchase)
            return

        # PAUSE THE WORKFLOW FOR HUMAN APPROVAL
        print("  ApprovalGateway: PAUSED - awaiting manager")

        await ctx.request_info(
            request_data=ManagerApprovalRequest(
                purchase=purchase,
                prompt=(
                    f"Review {purchase['id']} | "
                    f"{purchase['item']} | "
                    f"INR {purchase['amount_inr']:,} | "
                    f"Risk: {purchase['risk']}"
                ),
            ),
            response_type=str,
        )

    @response_handler
    async def on_manager_decision(
        self,
        original_request: ManagerApprovalRequest,
        response: str,
        ctx: WorkflowContext[dict],
    ) -> None:

        purchase = dict(original_request.purchase)

        decision = response.strip().lower()

        if decision not in {"approve", "reject"}:
            decision = "reject"

        purchase["decision"] = (
            "MANAGER_APPROVED"
            if decision == "approve"
            else "MANAGER_REJECTED"
        )

        purchase["decision_by"] = "Human manager"

        # RESUME THE WORKFLOW
        await ctx.send_message(purchase)


# =====================================================
# 3. CREATE AGENTS AND WORKFLOW
# =====================================================

def build_workflow(chat_client: OpenAIChatClient):

    # AGENT 1 - PROCUREMENT

    procurement_agent = Agent(
        client=chat_client,
        name="ProcurementAgent",
        instructions=(
            "You review equipment purchase requests. "
            "Provide a concise business justification "
            "review based only on supplied facts. "
            "Do not invent prices or vendor background. "
            "Respond in 2 sentences."
        ),
    )

    # AGENT 2 - RISK ASSESSMENT

    risk_agent = Agent(
        client=chat_client,
        name="RiskAgent",
        instructions=(
            "You assess procurement risk using only "
            "the supplied data. "
            "Mark HIGH when vendor_status is new, "
            "or when there is a critical missing detail. "
            "Otherwise mark LOW. "
            "Reply with one word only: HIGH or LOW."
        ),
    )

    # AGENT 3 - DECISION SUMMARY

    summary_agent = Agent(
        client=chat_client,
        name="SummaryAgent",
        instructions=(
            "Create a short procurement decision record. "
            "Respect the exact decision supplied by the "
            "workflow; you cannot override it. "
            "Say this is a demo record, not a placed order. "
            "Include request ID, amount, risk, "
            "human or automatic decision, and "
            "brief next steps."
        ),
    )

    # -------------------------------------------------
    # EXECUTOR 1 - PROCUREMENT REVIEW
    # -------------------------------------------------

    @executor(id="procurement_review")
    async def procurement_review(
        purchase: dict,
        ctx: WorkflowContext[dict],
    ) -> None:

        print("  ProcurementAgent: checking justification")

        result = await procurement_agent.run(
            json.dumps(purchase)
        )

        await ctx.send_message({
            **purchase,
            "review": result.text.strip(),
        })

    # -------------------------------------------------
    # EXECUTOR 2 - RISK REVIEW
    # -------------------------------------------------

    @executor(id="risk_review")
    async def risk_review(
        purchase: dict,
        ctx: WorkflowContext[dict],
    ) -> None:

        print("  RiskAgent: assessing purchase")

        result = await risk_agent.run(
            json.dumps(purchase)
        )

        risk = result.text.strip().upper().strip(". ")

        if risk not in {"HIGH", "LOW"}:
            risk = "HIGH"

        # Deterministic safeguard for new vendors
        if purchase.get(
            "vendor_status", ""
        ).lower() != "approved":
            risk = "HIGH"

        print(f"  RiskAgent: {risk}")

        await ctx.send_message({
            **purchase,
            "risk": risk,
        })

    # -------------------------------------------------
    # EXECUTOR 3 - FINAL DECISION
    # -------------------------------------------------

    @executor(id="final_summary")
    async def final_summary(
        purchase: dict,
        ctx: WorkflowContext[Never, str],
    ) -> None:

        print("  SummaryAgent: preparing decision record")

        result = await summary_agent.run(
            json.dumps(purchase)
        )

        await ctx.yield_output(
            f"{purchase['id']} -> "
            f"{purchase['decision']}\n"
            f"{result.text.strip()}"
        )

    # -------------------------------------------------
    # CREATE GRAPH
    # -------------------------------------------------

    gateway = ApprovalGateway()

    return (
        WorkflowBuilder(
            start_executor=procurement_review
        )
        .add_edge(
            procurement_review,
            risk_review,
        )
        .add_edge(
            risk_review,
            gateway,
        )
        .add_edge(
            gateway,
            final_summary,
        )
        .build()
    )


# =====================================================
# 4. EXECUTE WORKFLOW WITH HUMAN APPROVAL
# =====================================================

async def run_one(workflow, purchase: dict) -> None:

    responses = None
    first_run = True

    while True:

        # Initial execution or resumed execution
        if first_run:
            stream = workflow.run(
                purchase,
                stream=True,
            )
            first_run = False
        else:
            stream = workflow.run(
                stream=True,
                responses=responses,
            )

        pending = {}
        completed = False

        async for event in stream:

            if event.type == "request_info":
                pending[event.request_id] = event.data

            elif event.type == "output":
                print(
                    "\nFINAL RECORD\n"
                    + str(event.data)
                )
                completed = True

        if completed:
            return

        if not pending:
            raise RuntimeError(
                "Workflow stopped without output "
                "or approval request"
            )

        responses = {}

        # Ask the human manager for a decision
        for request_id, request in pending.items():

            print("\n" + "=" * 50)
            print("MANAGER APPROVAL REQUIRED")
            print(request.prompt)

            print(
                "Procurement review:",
                request.purchase["review"],
            )

            while True:

                answer = input(
                    "Type approve or reject: "
                ).strip().lower()

                if answer in {"approve", "reject"}:
                    break

                print("Please type approve or reject.")

            responses[request_id] = answer


# =====================================================
# 5. MAIN APPLICATION
# =====================================================

async def main() -> None:

    api_key = os.environ[
        "AZURE_OPENAI_API_KEY"
    ]

    base_url = os.environ[
        "AZURE_OPENAI_BASE_URL"
    ]

    model = os.environ[
        "AZURE_OPENAI_MODEL"
    ]

    # Load purchase requests
    with (
        BASE_DIR / "purchases.json"
    ).open(encoding="utf-8") as file:

        purchases = json.load(file)

    # Azure Foundry API key authentication
    async with AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    ) as client:

        chat_client = OpenAIChatClient(
            model=model,
            async_client=client,
        )

        # Process requests sequentially
        for purchase in purchases:

            print("\n" + "=" * 60)

            print(
                f"REQUEST {purchase['id']}: "
                f"{purchase['item']}"
            )

            # Fresh workflow state for every request
            workflow = build_workflow(chat_client)

            await run_one(workflow, purchase)


# =====================================================
# 6. ENTRY POINT
# =====================================================

if __name__ == "__main__":
    asyncio.run(main())
