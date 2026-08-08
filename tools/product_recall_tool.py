import os
import re
from datetime import datetime
from typing import Any

import requests
from langchain_core.tools import tool


OPENFDA_FOOD_ENFORCEMENT_URL = "https://api.fda.gov/food/enforcement.json"
VALID_CLASSIFICATIONS = {"Class I", "Class II", "Class III"}


def _openfda_phrase(value: str) -> str:
    cleaned = " ".join(value.strip().split())
    cleaned = cleaned.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{cleaned}"'


def _openfda_date(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%Y%m%d")
    except ValueError as exc:
        raise ValueError("Dates must use YYYY-MM-DD format") from exc


def _display_date(value: str | None) -> str | None:
    if not value or not re.fullmatch(r"\d{8}", value):
        return value
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def check_product_recalls(
    query: str,
    limit: int = 5,
    classification: str | None = None,
    status: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Search official openFDA food recall enforcement reports."""
    query = " ".join(query.strip().split())
    if not query:
        return {"status": "error", "message": "A product or company query is required."}

    limit = max(1, min(int(limit), 10))
    if classification and classification not in VALID_CLASSIFICATIONS:
        return {
            "status": "error",
            "message": "classification must be Class I, Class II, or Class III",
        }

    start_date = _openfda_date(date_from)
    end_date = _openfda_date(date_to)
    if start_date and end_date and start_date > end_date:
        return {"status": "error", "message": "date_from must not be after date_to"}

    phrase = _openfda_phrase(query)
    search_parts = [
        f"(product_description:{phrase} OR recalling_firm:{phrase})"
    ]
    if classification:
        search_parts.append(f"classification:{_openfda_phrase(classification)}")
    if status:
        search_parts.append(f"status:{_openfda_phrase(status)}")
    if start_date or end_date:
        lower = start_date or "19000101"
        upper = end_date or "now"
        search_parts.append(f"report_date:[{lower} TO {upper}]")

    params = {
        "search": " AND ".join(search_parts),
        "limit": limit,
        "sort": "report_date:desc",
    }
    api_key = os.getenv("OPENFDA_API_KEY")
    if api_key:
        params["api_key"] = api_key

    try:
        response = requests.get(
            OPENFDA_FOOD_ENFORCEMENT_URL,
            params=params,
            timeout=12,
        )
    except requests.RequestException as exc:
        return {
            "status": "error",
            "message": f"The openFDA request failed: {type(exc).__name__}",
        }

    if response.status_code == 404:
        return {
            "status": "completed",
            "query": query,
            "matches": 0,
            "recalls": [],
            "message": "No matching openFDA food enforcement reports were found.",
            "disclaimer": (
                "No match does not prove that a product is safe or has never "
                "been recalled. This search covers openFDA food enforcement records."
            ),
            "source": "openFDA",
        }

    if not response.ok:
        return {
            "status": "error",
            "message": f"openFDA returned HTTP {response.status_code}",
        }

    payload = response.json()
    recalls = []
    for report in payload.get("results", []):
        product_description = report.get("product_description") or ""
        recalling_firm = report.get("recalling_firm") or ""
        query_lower = query.lower()
        if query_lower in recalling_firm.lower():
            match_basis = "recalling company name"
        elif query_lower in product_description.lower():
            match_basis = "product description mention"
        else:
            match_basis = "openFDA text search"
        recalls.append(
            {
                "product": product_description,
                "company": recalling_firm,
                "match_basis": match_basis,
                "classification": report.get("classification"),
                "status": report.get("status"),
                "reason": report.get("reason_for_recall"),
                "recall_initiation_date": _display_date(
                    report.get("recall_initiation_date")
                ),
                "report_date": _display_date(report.get("report_date")),
                "distribution": report.get("distribution_pattern"),
                "product_codes_or_lots": report.get("code_info"),
                "quantity": report.get("product_quantity"),
                "location": {
                    "city": report.get("city"),
                    "state": report.get("state"),
                    "country": report.get("country"),
                },
                "recall_number": report.get("recall_number"),
            }
        )

    return {
        "status": "completed",
        "query": query,
        "matches": payload.get("meta", {}).get("results", {}).get("total", len(recalls)),
        "returned": len(recalls),
        "recalls": recalls,
        "interpretation": (
            "A matching report may describe a product that contains or mentions "
            "the search term. It does not necessarily represent a recall initiated "
            "by that brand or manufacturer."
        ),
        "disclaimer": (
            "openFDA data is informational and should not be treated as medical "
            "advice or proof that an unmatched product is safe."
        ),
        "source": "openFDA food enforcement reports",
    }


@tool
def product_recall_checker(
    query: str,
    limit: int = 5,
    classification: str | None = None,
    status: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Check official openFDA food recall enforcement reports.

    Use this for current or historical food recall questions by product name or
    recalling company. Optional filters support recall classification, status,
    and report-date range. This tool uses live external data and does not use RAG.
    """
    return check_product_recalls(
        query=query,
        limit=limit,
        classification=classification,
        status=status,
        date_from=date_from,
        date_to=date_to,
    )
