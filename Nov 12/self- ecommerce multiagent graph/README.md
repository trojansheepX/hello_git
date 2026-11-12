# E-commerce Returns & Refund Processing

A graph-based multi-agent workflow built with **Microsoft Agent Framework** and **Azure AI Foundry**.
It reviews customer return, refund, and replacement requests automatically. Requests that are
high-value, risky, or exceptional go to a human manager for approval.

> **Simulation only.** It processes no payments, sends no messages, and starts no replacements.

## Files

| File | Purpose |
|---|---|
| `ecommerce_graph_workflow.py` | Complete application |
| `returns.json` | Input data (10 customer requests) |
| `requirements.txt` | Python dependencies |
| `.env.example` | Configuration template (no real credentials) |
| `output_ecommerce_returns_workflow.txt` | Console output: auto approval, human approval, human rejection, invalid input |
| `output_missing_config.txt` | Console output: missing API configuration |
| `returns_results.json` | Structured final output for every request (generated) |

## Workflow graph

```
return_request ─▶ eligibility_review ─▶ risk_review ─▶ approval_gateway ─▶ customer_response ─▶ final_output
 (intake)         (Eligibility Agent)   (Risk Agent)    (rules 1-6 + HITL)   (Customer Response    (structured
                                                                              Agent)                record)
```

| Executor | Responsibility |
|---|---|
| `return_request` | Entry point. Receives one validated request record. |
| `eligibility_review` | Runs **deterministic Python checks for rules 1–5**. The **Eligibility Agent** then explains the findings. Its explanation cannot change the rule results. |
| `risk_review` | The **Risk Agent** returns `LOW/MEDIUM/HIGH` plus a short assessment. A deterministic *risk floor* (≥3 returns, ≥ ₹25,000, damage without photos, or a missing invoice gives HIGH; a used item or a request outside the window gives MEDIUM) means the AI can raise the risk level but never lower it. A reply that can't be parsed counts as HIGH. |
| `approval_gateway` | Applies rule 6 (no HIGH risk). If all six rules pass, the result is `AUTO_APPROVED`. Otherwise it **pauses** with `ctx.request_info(...)` and resumes in a `@response_handler` with the manager's decision. |
| `customer_response` | The **Customer Response Agent** drafts the email from the binding decision. Guardrails reject drafts that claim a refund or replacement is already done, or that soften a rejection. A rejected draft is replaced with a safe template. |
| `final_output` | Yields the structured record (Task 6). |

All request context travels with the message dict through every edge. The `ManagerApprovalRequest`
carries the full record, so the context survives the pause and resume.

## Return policy (deterministic)

A request is auto-approved only when **all** of these are true:

1. Amount < ₹25,000
2. Within 30 days of delivery
3. Invoice available
4. Unopened, **or** damaged with photos
5. Fewer than 3 returns in the last 90 days
6. Risk Agent did not report HIGH risk

Anything else is escalated for human review.

## Human-in-the-loop

- The gateway calls `ctx.request_info(ManagerApprovalRequest, response_type=str)`. The workflow emits a `request_info` event and stops.
- The console shows the request ID, item, amount, eligibility issues, and risk assessment, then asks for `approve` or `reject`.
- Any other input is refused and asked for again. The response handler also checks the value: if it is invalid, the handler issues a new request and the workflow stays paused instead of guessing.
- `workflow.run(responses={request_id: decision}, stream=True)` resumes the paused workflow.
- A manager rejection becomes `MANAGER_REJECTED`. The AI cannot override it: the response agent only drafts text, and the guardrail enforces rejection wording.

## Final output (per request)

`request_id`, `order_id`, `product`, `requested_amount_inr`, `eligibility_findings`,
`eligibility_issues`, `risk_assessment`, `approval_route` (`Automatic` or `Human Review`),
`final_decision` (`AUTO_APPROVED`, `MANAGER_APPROVED` or `MANAGER_REJECTED`), and `customer_communication_draft`.

## Setup and run

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then edit .env with real values

python ecommerce_graph_workflow.py                    # all 10 requests, interactive
python ecommerce_graph_workflow.py --ids RET202,RET204 # a subset
python ecommerce_graph_workflow.py --env-file missing.env   # demo: missing configuration
```

Manager decisions can also be piped in for a repeatable demo:

```powershell
"maybe`n`nreject`napprove`nreject`nreject`napprove`nreject`n" | python ecommerce_graph_workflow.py
```

If the configuration is missing, the app lists every missing or placeholder setting and exits with code 2
before it makes any API call. Azure API errors (bad key, endpoint, or deployment) produce a clear message and exit code 4.

## Test results (from `returns.json`)

| Request | Route | Decision |
|---|---|---|
| RET201 Wireless Keyboard | Automatic | AUTO_APPROVED |
| RET202 Gaming Laptop (₹79,999, used) | Human Review | MANAGER_REJECTED (after 2 invalid inputs) |
| RET203 Ceramic Dinner Set | Automatic | AUTO_APPROVED |
| RET204 Travel Suitcase (damaged, no photo) | Human Review | MANAGER_APPROVED |
| RET205 Yoga Mat (44 days) | Human Review | MANAGER_REJECTED |
| RET206 Desk Lamp (4 prior returns) | Human Review | MANAGER_REJECTED |
| RET207 Coffee Grinder (no invoice) | Human Review | MANAGER_APPROVED |
| RET208 Sketchbook Set | Automatic | AUTO_APPROVED |
| RET209 Smartwatch (used) | Human Review | MANAGER_REJECTED |
| RET210 Floor Mirror (damaged + photos) | Automatic | AUTO_APPROVED |

Note: if your shell sets `PORT=0`, unset it before running. The Azure agent-server telemetry
package installed with `agent-framework` rejects that value.
