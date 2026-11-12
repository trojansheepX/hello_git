"""
E-commerce Returns & Refund Processing
Graph-based multi-agent workflow (Microsoft Agent Framework + Azure AI Foundry)

Graph:
    Return Request -> Eligibility Agent -> Risk Agent -> Approval Gateway
                   -> Customer Response Agent -> Final Output

SIMULATION ONLY: no payments are processed, no messages are sent and no
replacements are initiated.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Never

from dotenv import load_dotenv
from openai import APIError, AsyncOpenAI
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

REQUIRED_SETTINGS = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_BASE_URL",
    "AZURE_OPENAI_MODEL",
)

# Demonstration return policy thresholds
MAX_AUTO_AMOUNT_INR = 25_000
MAX_RETURN_DAYS = 30
MAX_PREVIOUS_RETURNS = 3          # must be FEWER than this
VALID_DECISIONS = {"approve", "reject"}
RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")

REQUIRED_FIELDS = {
    "request_id": str,
    "order_id": str,
    "product": str,
    "product_category": str,
    "request_type": str,
    "amount_inr": (int, float),
    "days_since_delivery": int,
    "item_condition": str,
    "return_reason": str,
    "invoice_available": bool,
    "photo_evidence_available": bool,
    "previous_returns_90_days": int,
}


class ConfigurationError(Exception):
    """Raised when Azure AI Foundry settings are missing or placeholders."""


def load_settings(env_file: Path) -> dict:
    """Load and validate the Azure AI Foundry settings.

    Reports *all* missing settings at once instead of failing on the first.
    """
    if env_file.exists():
        load_dotenv(env_file)

    settings, missing = {}, []
    for name in REQUIRED_SETTINGS:
        value = os.getenv(name, "").strip()
        if (
            not value
            or value.upper().startswith(("PASTE_", "YOUR_", "<"))
        ):
            missing.append(name)
        else:
            settings[name] = value

    if missing:
        raise ConfigurationError(
            "Missing or placeholder Azure AI Foundry settings: "
            + ", ".join(missing)
            + f"\n  Looked in: {env_file}"
            + "\n  Copy .env.example to .env and fill in real values."
        )
    return settings


# =====================================================
# 2. TASK 1 - LOAD CUSTOMER REQUESTS
# =====================================================

def load_requests(path: Path) -> list[dict]:
    """Load return requests dynamically and validate their structure."""
    with path.open(encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(f"{path.name} must contain a JSON array")

    for index, record in enumerate(records, start=1):
        for field, expected in REQUIRED_FIELDS.items():
            if field not in record:
                raise ValueError(f"Record {index}: missing field '{field}'")
            value = record[field]
            # bool is a subclass of int - reject it for numeric fields
            if not isinstance(value, expected) or (
                expected is not bool and isinstance(value, bool)
            ):
                raise ValueError(
                    f"Record {index} ({record.get('request_id')}): "
                    f"field '{field}' has invalid value {value!r}"
                )
    return records


# =====================================================
# 3. TASK 3 - DETERMINISTIC BUSINESS RULES
# =====================================================

def check_policy_rules(req: dict) -> list[dict]:
    """Rules 1-5 of the demo return policy (pure Python, no AI)."""
    condition = req["item_condition"].upper()
    condition_ok = condition == "UNOPENED" or (
        condition == "DAMAGED" and req["photo_evidence_available"]
    )

    return [
        {
            "rule": "1. Amount below INR 25,000",
            "passed": req["amount_inr"] < MAX_AUTO_AMOUNT_INR,
            "detail": f"INR {req['amount_inr']:,}",
        },
        {
            "rule": "2. Within 30 days of delivery",
            "passed": req["days_since_delivery"] <= MAX_RETURN_DAYS,
            "detail": f"{req['days_since_delivery']} days since delivery",
        },
        {
            "rule": "3. Invoice available",
            "passed": req["invoice_available"],
            "detail": "invoice "
            + ("available" if req["invoice_available"] else "missing"),
        },
        {
            "rule": "4. Unopened, or damaged with photos",
            "passed": condition_ok,
            "detail": f"condition={condition}, photos="
            + ("yes" if req["photo_evidence_available"] else "no"),
        },
        {
            "rule": "5. Fewer than 3 returns in 90 days",
            "passed": req["previous_returns_90_days"] < MAX_PREVIOUS_RETURNS,
            "detail": f"{req['previous_returns_90_days']} previous returns",
        },
    ]


def deterministic_risk(req: dict) -> tuple[str, list[str]]:
    """Minimum risk level derived from hard facts.

    The Risk Agent may raise this level but can never lower it.
    """
    flags_high, flags_medium = [], []

    if req["previous_returns_90_days"] >= MAX_PREVIOUS_RETURNS:
        flags_high.append("unusual return frequency")
    if req["amount_inr"] >= MAX_AUTO_AMOUNT_INR:
        flags_high.append("high-value request")
    if req["item_condition"].upper() == "DAMAGED" and not req[
        "photo_evidence_available"
    ]:
        flags_high.append("damage claimed without photo evidence")
    if not req["invoice_available"]:
        flags_high.append("missing invoice")

    if req["item_condition"].upper() == "USED":
        flags_medium.append("item has been used")
    if req["days_since_delivery"] > MAX_RETURN_DAYS:
        flags_medium.append("outside the return window")

    if flags_high:
        return "HIGH", flags_high + flags_medium
    if flags_medium:
        return "MEDIUM", flags_medium
    return "LOW", []


def max_risk(a: str, b: str) -> str:
    return a if RISK_LEVELS.index(a) >= RISK_LEVELS.index(b) else b


def parse_risk_reply(text: str) -> tuple[str, str]:
    """Parse 'RISK_LEVEL: X / ASSESSMENT: ...'. Unparseable -> HIGH (fail safe)."""
    level_match = re.search(r"RISK_LEVEL\s*:\s*(LOW|MEDIUM|HIGH)", text, re.I)
    assess_match = re.search(r"ASSESSMENT\s*:\s*(.+)", text, re.I | re.S)
    level = level_match.group(1).upper() if level_match else "HIGH"
    assessment = (assess_match.group(1) if assess_match else text).strip()
    return level, assessment


# Phrases that would wrongly claim the refund/replacement is already done
COMPLETION_CLAIMS = re.compile(
    r"\b(has|have) been (refunded|credited|processed|issued|completed|"
    r"shipped|dispatched|sent|initiated|replaced)\b"
    r"|\brefund (is|was) (complete|completed|processed|issued)\b"
    r"|\bwe (have )?(refunded|credited|shipped|dispatched|replaced)\b",
    re.I,
)


def fallback_customer_message(record: dict) -> str:
    """Safe template used if the AI draft violates communication rules."""
    kind = record["request_type"].lower()
    if record["final_decision"] == "MANAGER_REJECTED":
        return (
            f"Dear Customer,\n\nThank you for contacting us about your "
            f"{kind} request {record['request_id']} for order "
            f"{record['order_id']} ({record['product']}). After review, we "
            f"are unable to approve this request under our return policy. "
            f"Please reply to this message if you have additional "
            f"information you would like us to consider.\n\n"
            f"Customer Support Team"
        )
    return (
        f"Dear Customer,\n\nYour {kind} request {record['request_id']} for "
        f"order {record['order_id']} ({record['product']}) has been "
        f"approved. Our team will contact you with the next steps for the "
        f"{kind}; nothing has been processed yet.\n\nCustomer Support Team"
    )


# =====================================================
# 4. TASK 5 - HUMAN-IN-THE-LOOP APPROVAL GATEWAY
# =====================================================

@dataclass
class ManagerApprovalRequest:
    record: dict          # full request context, preserved across the pause
    prompt: str
    attempt: int = 1


class ApprovalGateway(Executor):
    """Applies all 6 policy rules; pauses for a manager when any fail."""

    def __init__(self):
        super().__init__(id="approval_gateway")

    @handler
    async def review(
        self,
        record: dict,
        ctx: WorkflowContext[dict],
    ) -> None:
        # Rule 6 - Risk Agent has not identified a HIGH-risk concern
        rule6 = {
            "rule": "6. No HIGH-risk concern",
            "passed": record["risk_level"] != "HIGH",
            "detail": f"risk level {record['risk_level']}",
        }
        rules = record["policy_checks"] + [rule6]
        failed = [r for r in rules if not r["passed"]]

        record = {
            **record,
            "policy_checks": rules,
            "eligibility_issues": [
                f"{r['rule']} - FAILED ({r['detail']})" for r in failed
            ],
        }

        if not failed:
            print("  [ApprovalGateway] all 6 rules passed -> AUTO_APPROVED")
            await ctx.send_message({
                **record,
                "approval_route": "Automatic",
                "final_decision": "AUTO_APPROVED",
                "decided_by": "Deterministic return policy",
            })
            return

        print(
            f"  [ApprovalGateway] {len(failed)} rule(s) failed "
            "-> PAUSED, awaiting manager"
        )
        await ctx.request_info(
            request_data=ManagerApprovalRequest(
                record=record,
                prompt=self._prompt(record),
            ),
            response_type=str,
        )

    @staticmethod
    def _prompt(record: dict) -> str:
        issues = "\n".join(f"    - {i}" for i in record["eligibility_issues"])
        return (
            f"  Request ID : {record['request_id']}  (order {record['order_id']})\n"
            f"  Item       : {record['product']} [{record['product_category']}]\n"
            f"  Type       : {record['request_type']}\n"
            f"  Amount     : INR {record['amount_inr']:,}\n"
            f"  Reason     : {record['return_reason']}\n"
            f"  Eligibility issues:\n{issues}\n"
            f"  Risk level : {record['risk_level']}\n"
            f"  Risk assessment: {record['risk_assessment']}"
        )

    @response_handler
    async def on_manager_decision(
        self,
        original_request: ManagerApprovalRequest,
        response: str,
        ctx: WorkflowContext[dict],
    ) -> None:
        record = dict(original_request.record)
        decision = (response or "").strip().lower()

        # Defence in depth: never guess on invalid input - stay paused
        if decision not in VALID_DECISIONS:
            print(
                f"  [ApprovalGateway] invalid decision {response!r} received; "
                "re-requesting manager input"
            )
            await ctx.request_info(
                request_data=ManagerApprovalRequest(
                    record=record,
                    prompt=original_request.prompt,
                    attempt=original_request.attempt + 1,
                ),
                response_type=str,
            )
            return

        print(f"  [ApprovalGateway] RESUMED with manager decision: {decision}")
        await ctx.send_message({
            **record,
            "approval_route": "Human Review",
            "final_decision": (
                "MANAGER_APPROVED" if decision == "approve"
                else "MANAGER_REJECTED"
            ),
            "decided_by": "Human manager",
        })


# =====================================================
# 5. TASK 2 + 4 - AGENTS AND GRAPH WORKFLOW
# =====================================================

def build_workflow(chat_client: OpenAIChatClient):

    # ---------------- AGENT 1: ELIGIBILITY ----------------
    eligibility_agent = Agent(
        client=chat_client,
        name="EligibilityAgent",
        instructions=(
            "You are an e-commerce returns eligibility reviewer. "
            "The return policy is: amount below INR 25,000; request within "
            "30 days of delivery; invoice available; item unopened, or "
            "damaged with supporting photographs; fewer than 3 previous "
            "returns in 90 days. "
            "You receive the request and the deterministic rule results, "
            "which are authoritative - never contradict them. "
            "In 2-3 sentences state whether the request meets eligibility "
            "requirements and explain each exception. Use only supplied facts."
        ),
    )

    # ---------------- AGENT 2: RISK ----------------
    risk_agent = Agent(
        client=chat_client,
        name="RiskAgent",
        instructions=(
            "You assess return/refund risk using only the supplied data. "
            "Consider unusual return frequency, missing invoice or photo "
            "evidence, high-value requests (INR 25,000 or more), used items "
            "and claims inconsistent with the stated condition. "
            "Use HIGH for serious fraud/abuse or financial concerns, MEDIUM "
            "for moderate concerns, LOW otherwise. "
            "Reply in exactly this format:\n"
            "RISK_LEVEL: <LOW|MEDIUM|HIGH>\n"
            "ASSESSMENT: <one or two sentences>"
        ),
    )

    # ---------------- AGENT 3: CUSTOMER RESPONSE ----------------
    response_agent = Agent(
        client=chat_client,
        name="CustomerResponseAgent",
        instructions=(
            "You write short, polite customer emails for an e-commerce "
            "returns team. The workflow's final_decision is binding and you "
            "cannot change it: AUTO_APPROVED or MANAGER_APPROVED means the "
            "request is approved; MANAGER_REJECTED means it is not approved. "
            "Never claim that a refund, credit, replacement or shipment has "
            "already been processed, issued or completed - say the next "
            "steps will follow. For rejections, explain using only the "
            "supplied policy_reasons, if any. Do not mention internal risk scores, rule "
            "numbers or the manager. Do not invent dates, amounts or "
            "policies. Include the request ID and order ID. Keep it under "
            "120 words. Sign as 'Customer Support Team'."
        ),
    )

    # ------------- EXECUTOR 0: RETURN REQUEST (intake) -------------
    @executor(id="return_request")
    async def return_request(req: dict, ctx: WorkflowContext[dict]) -> None:
        print(f"  [ReturnRequest] received {req['request_id']} "
              f"({req['request_type']}, INR {req['amount_inr']:,})")
        await ctx.send_message(dict(req))

    # ------------- EXECUTOR 1: ELIGIBILITY -------------
    @executor(id="eligibility_review")
    async def eligibility_review(req: dict, ctx: WorkflowContext[dict]) -> None:
        checks = check_policy_rules(req)
        passed = sum(c["passed"] for c in checks)
        print(f"  [EligibilityAgent] deterministic rules 1-5: "
              f"{passed}/{len(checks)} passed")

        result = await eligibility_agent.run(
            "Return request:\n" + json.dumps(req, ensure_ascii=False)
            + "\n\nDeterministic rule results (authoritative):\n"
            + json.dumps(checks, ensure_ascii=False)
        )
        await ctx.send_message({
            **req,
            "policy_checks": checks,
            "eligibility_findings": result.text.strip(),
        })

    # ------------- EXECUTOR 2: RISK -------------
    @executor(id="risk_review")
    async def risk_review(record: dict, ctx: WorkflowContext[dict]) -> None:
        facts = {k: record[k] for k in REQUIRED_FIELDS}
        result = await risk_agent.run(
            "Assess this return request:\n"
            + json.dumps(facts, ensure_ascii=False)
        )
        ai_level, ai_assessment = parse_risk_reply(result.text)
        floor_level, flags = deterministic_risk(record)
        level = max_risk(ai_level, floor_level)

        print(f"  [RiskAgent] AI={ai_level}, rule floor={floor_level} "
              f"-> {level}")

        await ctx.send_message({
            **record,
            "risk_level": level,
            "risk_assessment": ai_assessment,
            "risk_flags": flags,
        })

    # ------------- EXECUTOR 4: CUSTOMER RESPONSE -------------
    @executor(id="customer_response")
    async def customer_response(record: dict, ctx: WorkflowContext[dict]) -> None:
        print(f"  [CustomerResponseAgent] drafting message for "
              f"{record['final_decision']}")
        context = {
            "request_id": record["request_id"],
            "order_id": record["order_id"],
            "product": record["product"],
            "request_type": record["request_type"],
            "return_reason": record["return_reason"],
            "final_decision": record["final_decision"],
        }
        if record["final_decision"] == "MANAGER_REJECTED":
            # Customer-safe policy reasons (no risk scores / rule numbers)
            context["policy_reasons"] = [
                c["rule"].split(". ", 1)[1] + f" ({c['detail']})"
                for c in record["policy_checks"]
                if not c["passed"] and not c["rule"].startswith("6.")
            ]
        result = await response_agent.run(
            "Draft the customer message for:\n"
            + json.dumps(context, ensure_ascii=False)
        )
        draft = result.text.strip()

        # Guardrails: AI may not claim completion or soften a rejection
        rejected = record["final_decision"] == "MANAGER_REJECTED"
        rejection_words = re.search(
            r"unable|not (been )?approved|cannot|can't|declin|reject|"
            r"not eligible|regret",
            draft, re.I,
        )
        if COMPLETION_CLAIMS.search(draft) or (rejected and not rejection_words):
            print("  [CustomerResponseAgent] draft failed guardrail -> "
                  "using safe template")
            draft = fallback_customer_message(record)

        await ctx.send_message({**record, "customer_message": draft})

    # ------------- EXECUTOR 5: FINAL OUTPUT -------------
    @executor(id="final_output")
    async def final_output(record: dict, ctx: WorkflowContext[Never, dict]) -> None:
        await ctx.yield_output({
            "request_id": record["request_id"],
            "order_id": record["order_id"],
            "product": record["product"],
            "request_type": record["request_type"],
            "requested_amount_inr": record["amount_inr"],
            "eligibility_findings": record["eligibility_findings"],
            "eligibility_issues": record["eligibility_issues"],
            "risk_assessment": {
                "level": record["risk_level"],
                "summary": record["risk_assessment"],
                "flags": record["risk_flags"],
            },
            "approval_route": record["approval_route"],
            "final_decision": record["final_decision"],
            "decided_by": record["decided_by"],
            "customer_communication_draft": record["customer_message"],
            "note": "SIMULATION - no payment, message or replacement executed",
        })

    # ------------- GRAPH -------------
    gateway = ApprovalGateway()

    return (
        WorkflowBuilder(start_executor=return_request)
        .add_edge(return_request, eligibility_review)
        .add_edge(eligibility_review, risk_review)
        .add_edge(risk_review, gateway)
        .add_edge(gateway, customer_response)
        .add_edge(customer_response, final_output)
        .build()
    )


# =====================================================
# 6. RUN ONE REQUEST (WITH PAUSE / RESUME)
# =====================================================

def ask_manager(request: ManagerApprovalRequest) -> str:
    """Prompt until a valid 'approve' / 'reject' is entered."""
    print("\n" + "=" * 64)
    print("  MANAGER APPROVAL REQUIRED (workflow paused)")
    print("=" * 64)
    print(request.prompt)
    print("-" * 64)

    while True:
        try:
            # strip a UTF-8 BOM that some shells add to piped input
            raw = input("  Type approve or reject: ").lstrip("﻿")
        except EOFError:
            raise SystemExit(
                "\nNo manager input available (stdin closed). "
                "Workflow left paused; exiting."
            )
        # Echo when input is piped so console logs show what was entered
        if not sys.stdin.isatty():
            print(raw)
        answer = raw.strip().lower()
        if answer in VALID_DECISIONS:
            return answer
        print(f"  Invalid input {raw.strip()!r}. "
              "Please type exactly 'approve' or 'reject'.")


async def run_one(workflow, req: dict) -> dict:
    stream = workflow.run(req, stream=True)

    while True:
        pending, output = {}, None

        async for event in stream:
            if event.type == "request_info":
                pending[event.request_id] = event.data
            elif event.type == "output":
                output = event.data

        if output is not None:
            return output
        if not pending:
            raise RuntimeError("Workflow stopped without output or approval request")

        responses = {rid: ask_manager(r) for rid, r in pending.items()}
        print("  Resuming workflow ...")
        stream = workflow.run(stream=True, responses=responses)


def print_result(result: dict) -> None:
    print("\n" + "-" * 64)
    print(f"  FINAL OUTPUT  {result['request_id']} -> {result['final_decision']}")
    print("-" * 64)
    print(json.dumps(result, indent=2, ensure_ascii=False))


# =====================================================
# 7. MAIN APPLICATION
# =====================================================

async def main() -> int:
    parser = argparse.ArgumentParser(description="Returns & refund workflow")
    parser.add_argument("--input", default=str(BASE_DIR / "returns.json"),
                        help="path to returns JSON file")
    parser.add_argument("--env-file", default=str(BASE_DIR / ".env"),
                        help="path to .env configuration file")
    parser.add_argument("--ids", default="",
                        help="comma-separated request IDs to process (default: all)")
    parser.add_argument("--output", default=str(BASE_DIR / "returns_results.json"),
                        help="where to save the structured results")
    args = parser.parse_args()

    print("=" * 64)
    print("  E-COMMERCE RETURNS & REFUND PROCESSING  (SIMULATION)")
    print("  Microsoft Agent Framework + Azure AI Foundry")
    print("=" * 64)

    # --- configuration (Task 7: missing API configuration) ---
    try:
        settings = load_settings(Path(args.env_file))
    except ConfigurationError as exc:
        print(f"\nCONFIGURATION ERROR: {exc}")
        print("No requests were processed.")
        return 2

    # --- data (Task 1) ---
    try:
        requests = load_requests(Path(args.input))
    except (OSError, ValueError) as exc:
        print(f"\nINPUT ERROR: {exc}")
        return 3

    if args.ids:
        wanted = {i.strip().upper() for i in args.ids.split(",") if i.strip()}
        requests = [r for r in requests if r["request_id"].upper() in wanted]

    print(f"\nLoaded {len(requests)} return request(s) from {Path(args.input).name}")
    print(f"Deployment: {settings['AZURE_OPENAI_MODEL']}")

    results = []
    async with AsyncOpenAI(
        api_key=settings["AZURE_OPENAI_API_KEY"],
        base_url=settings["AZURE_OPENAI_BASE_URL"],
    ) as sdk_client:
        chat_client = OpenAIChatClient(
            model=settings["AZURE_OPENAI_MODEL"],
            async_client=sdk_client,
        )

        for req in requests:
            print("\n" + "=" * 64)
            print(f"  REQUEST {req['request_id']}: {req['product']}")
            print("=" * 64)

            workflow = build_workflow(chat_client)  # fresh state per request
            try:
                result = await run_one(workflow, req)
            except APIError as exc:
                print(f"\nAZURE AI FOUNDRY ERROR: {exc.__class__.__name__}: {exc}")
                print("Check AZURE_OPENAI_API_KEY / BASE_URL / MODEL in .env.")
                return 4
            print_result(result)
            results.append(result)

    # --- summary ---
    Path(args.output).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n" + "=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    for r in results:
        print(f"  {r['request_id']:<8} {r['product']:<22} "
              f"INR {r['requested_amount_inr']:>7,}  "
              f"{r['approval_route']:<13} {r['final_decision']}")
    print(f"\nStructured results saved to {Path(args.output).name}")
    print("SIMULATION ONLY - no payments, messages or replacements were executed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
