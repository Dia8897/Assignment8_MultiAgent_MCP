import json
import re

from agent.llm.ai_model import llm
from agent.state.agent_state import AgentState
from tools.product_recall_tool import product_recall_checker


def _json_object(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Recall query extraction did not return JSON")
    return json.loads(cleaned[start:end + 1])


def _format_recalls(data: dict) -> str:
    if data.get("status") == "error":
        return data.get("message", "The openFDA lookup failed.")
    recalls = data.get("recalls", [])
    if not recalls:
        return f'{data.get("message", "No matching reports were found.")}\n\n{data["disclaimer"]}'

    lines = [
        f'Found {data.get("matches", len(recalls))} openFDA report(s) matching '
        f'"{data.get("query", "the search term")}".',
        data.get("interpretation", ""),
    ]
    for recall in recalls:
        lines.extend([
            "",
            f'**{recall.get("product") or "Unnamed product"}**',
            f'- Company: {recall.get("company") or "Not provided"}',
            f'- Match basis: {recall.get("match_basis") or "openFDA text search"}',
            f'- Classification: {recall.get("classification") or "Not provided"}',
            f'- Status: {recall.get("status") or "Not provided"}',
            f'- Reason: {recall.get("reason") or "Not provided"}',
            f'- Report date: {recall.get("report_date") or "Not provided"}',
            f'- Recall number: {recall.get("recall_number") or "Not provided"}',
        ])
    lines.extend(["", data["disclaimer"], "Source: openFDA food enforcement reports."])
    return "\n".join(lines)


async def recall_agent(state: AgentState):
    task = state["remaining_task"] or state["message"]
    prompt = f"""
        Convert the recall request into arguments for an openFDA food recall search.
        Return only JSON with: query, limit, classification, status, date_from,
        date_to. query must contain only the product, company, contaminant, or recall
        subject—not the full question. Use null for absent filters and YYYY-MM-DD dates.

        Request: {task}
    """
    parsed_response = llm.invoke(prompt)
    parsed_text = str(
        getattr(parsed_response, "text", "")
        or getattr(parsed_response, "content", "")
    )
    try:
        arguments = _json_object(parsed_text)
        allowed_arguments = {
            "query",
            "limit",
            "classification",
            "status",
            "date_from",
            "date_to",
        }
        arguments = {
            key: value
            for key, value in arguments.items()
            if key in allowed_arguments and value is not None
        }
        arguments.setdefault("query", task)
        arguments.setdefault("limit", 5)
        data = await product_recall_checker.ainvoke(arguments)
        result = _format_recalls(data)
    except Exception as exc:
        result = f"The recall request could not be prepared reliably: {exc}"

    return {
        "response": result,
        "selected_agent": "recall_agent",
        "specialist_results": state["specialist_results"] + [result],
        "completed_agents": state["completed_agents"] + ["recall_agent"],
        "remaining_task": "",
    }
