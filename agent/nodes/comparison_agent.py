import json
import re

from agent.llm.ai_model import llm
from agent.state.agent_state import AgentState
from tools.product_decision_tool import product_decision_analyzer


def _json_object(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Product extraction did not return JSON")
    return json.loads(cleaned[start:end + 1])


def _format_analysis(analysis: dict) -> str:
    if analysis.get("status") != "completed":
        missing = analysis.get("missing_required_metrics", [])
        suffix = f" Missing criteria: {', '.join(missing)}." if missing else ""
        return f'{analysis.get("message", "The evidence is insufficient.")}{suffix}'

    lines = [
        f'**Recommended product: {analysis["recommended_product"]}**',
        "",
        "Ranking:",
    ]
    for position, product in enumerate(analysis["ranking"], start=1):
        price = product.get("current_price")
        currency = product.get("currency") or ""
        price_text = f"{price:.2f} {currency}" if price is not None else "price unavailable"
        details = [f'score {product["score"]:.2f}']
        if analysis["weights"].get("total_price", 0) > 0:
            details.append(price_text)
        if analysis["weights"].get("unit_price", 0) > 0:
            details.append(f'unit price: {product["unit_price_label"]}')
        lines.append(
            f'{position}. **{product["name"]}** — ' + "; ".join(details)
        )
        discount = product.get("computed_discount_percent")
        if discount is not None:
            lines.append(f"   - Calculated discount: {discount:.1f}%")
        for flag in product.get("flags", []):
            lines.append(f"   - Warning: {flag}")

    for warning in analysis.get("warnings", []):
        lines.append(f"- Data note: {warning}")
    lines.append("\nRanking uses only the retrieved RAG evidence.")
    return "\n".join(lines)


async def comparison_agent(state: AgentState):
    evidence = state["retrieved_evidence"]
    if not evidence:
        result = "Product comparison could not run because no RAG evidence was retrieved."
    else:
        prompt = f"""
            Extract structured product records from RAG evidence for deterministic
            comparison. Use only explicit evidence and null for missing values.
            Never invent prices, sizes, currencies, ingredients, or exchange rates.
            Keep facts attached to their exact product variant. Count ingredients
            only when an explicit ingredient list is present.

            Interpret the user's requested weights. Valid keys are unit_price,
            total_price, ingredients, and size. Include explicit 0 values for
            criteria the user says to ignore.

            Return exactly this JSON shape:
            {{
              "products": [{{
                "name": "string",
                "current_price": null,
                "original_price": null,
                "currency": null,
                "size_value": null,
                "size_unit": null,
                "ingredients": [],
                "ingredient_count": null,
                "claimed_discount_percent": null,
                "source": "short evidence reference"
              }}],
              "weights": {{
                "unit_price": 0.0,
                "total_price": 0.0,
                "ingredients": 0.0,
                "size": 0.0
              }}
            }}

            Comparison request:
            {state["remaining_task"] or state["message"]}

            RAG evidence:
            {evidence[:12000]}
        """
        extraction = llm.invoke(prompt)
        extraction_text = str(
            getattr(extraction, "text", "")
            or getattr(extraction, "content", "")
        )
        try:
            structured = _json_object(extraction_text)
            analysis = await product_decision_analyzer.ainvoke(
                {
                    "products": structured.get("products", []),
                    "weights": structured.get("weights") or None,
                }
            )
            result = _format_analysis(analysis)
        except Exception as exc:
            result = f"Product evidence could not be structured reliably: {exc}"

    return {
        "response": result,
        "selected_agent": "comparison_agent",
        "specialist_results": state["specialist_results"] + [result],
        "completed_agents": state["completed_agents"] + ["comparison_agent"],
        "remaining_task": "",
    }
