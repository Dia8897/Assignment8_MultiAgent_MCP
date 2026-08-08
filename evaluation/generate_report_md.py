import csv
import json
from datetime import datetime
from pathlib import Path


EVALUATION_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = EVALUATION_DIR / "evaluation_report.md"

RAGAS_SUMMARY_PATH = EVALUATION_DIR / "ragas_summary.json"
RAGAS_RESULTS_PATH = EVALUATION_DIR / "ragas_results.csv"
AGENT_RESULTS_PATH = EVALUATION_DIR / "evaluation_results.csv"

RAGAS_METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "llm_context_precision_with_reference": "Context Precision",
    "context_recall": "Context Recall",
}

RAGAS_RESULT_COLUMNS = [
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "context_recall",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _truncate(text: str, limit: int = 90) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _escape_md_cell(text: str) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", " ")


def _format_score(value) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _load_ragas_summary() -> dict[str, float]:
    if not RAGAS_SUMMARY_PATH.exists():
        return {}
    return json.loads(RAGAS_SUMMARY_PATH.read_text(encoding="utf-8"))


def _agent_result_status(row: dict[str, str]) -> str:
    actual_route = row.get("actual_route", "")
    response = row.get("response", "")

    if actual_route == "ERROR":
        return "Error"

    if "blocked" in response.lower() or row.get("input_classification") == "UNSAFE":
        return "Pass (blocked)"

    if "insufficient_data" in response.lower() or "cannot be calculated" in response.lower():
        return "Partial"

    if row.get("expected_route") != actual_route:
        return "Route mismatch"

    return "Pass"


def _agent_route_match(row: dict[str, str]) -> bool:
    return row.get("expected_route") == row.get("actual_route")


def build_report() -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    ragas_summary = _load_ragas_summary()
    ragas_rows = _read_csv(RAGAS_RESULTS_PATH)
    agent_rows = _read_csv(AGENT_RESULTS_PATH)

    lines = [
        "# Evaluation Report",
        "",
        f"Generated: {generated_at}",
        "",
        "Combined summary of RAGAS (RAG layer) and end-to-end multi-agent evaluation.",
        "",
    ]

    lines.extend(["## Overview", ""])
    if ragas_summary:
        ragas_count = len(ragas_rows) if ragas_rows else "—"
        lines.append(f"- **RAGAS test cases:** {ragas_count}")
    else:
        lines.append("- **RAGAS:** no summary file found")

    if agent_rows:
        route_matches = sum(1 for row in agent_rows if _agent_route_match(row))
        errors = sum(1 for row in agent_rows if row.get("actual_route") == "ERROR")
        partials = sum(
            1 for row in agent_rows if _agent_result_status(row) == "Partial"
        )
        passes = sum(1 for row in agent_rows if _agent_result_status(row).startswith("Pass"))
        lines.extend(
            [
                f"- **End-to-end test cases:** {len(agent_rows)}",
                f"- **Routing matches:** {route_matches}/{len(agent_rows)}",
                f"- **Passing responses:** {passes}/{len(agent_rows)}",
                f"- **Partial responses:** {partials}/{len(agent_rows)}",
                f"- **Execution errors:** {errors}/{len(agent_rows)}",
            ]
        )
    else:
        lines.append("- **End-to-end:** no results file found")

    lines.extend(["", "## RAGAS Summary (RAG / MCP)", ""])
    if ragas_summary:
        lines.extend(
            [
                "| Metric | Score |",
                "|--------|-------|",
            ]
        )
        for key, label in RAGAS_METRIC_LABELS.items():
            if key in ragas_summary:
                lines.append(f"| {label} | {_format_score(ragas_summary[key])} |")
    else:
        lines.append("_Run `python -m evaluation.run_ragas_evaluation` first._")

    lines.extend(["", "## RAGAS Per Question", ""])
    if ragas_rows:
        lines.extend(
            [
                "| Question | Faithfulness | Relevancy | Precision | Recall |",
                "|----------|--------------|-----------|-----------|--------|",
            ]
        )
        for row in ragas_rows:
            question = _escape_md_cell(_truncate(row.get("user_input", ""), 55))
            scores = [_format_score(row.get(column, "")) for column in RAGAS_RESULT_COLUMNS]
            lines.append(f"| {question} | {' | '.join(scores)} |")
    else:
        lines.append("_No RAGAS detail file found._")

    lines.extend(["", "## End-to-End Agent Evaluation", ""])
    if agent_rows:
        lines.extend(
            [
                "| # | Category | Expected Route | Actual Route | Status | Response Preview |",
                "|---|----------|----------------|--------------|--------|------------------|",
            ]
        )
        for index, row in enumerate(agent_rows, start=1):
            status = _agent_result_status(row)
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(index),
                        _escape_md_cell(row.get("category", "")),
                        _escape_md_cell(row.get("expected_route", "")),
                        _escape_md_cell(row.get("actual_route", "")),
                        status,
                        _escape_md_cell(_truncate(row.get("response", ""), 80)),
                    ]
                )
                + " |"
            )
    else:
        lines.append("_Run `python -m evaluation.run_evaluation` first._")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- **RAGAS** evaluates MCP `ask_rag` answer quality only.",
            "- **End-to-end** evaluates routing, guardrails, retrieval, comparison, recall, and general agents.",
            "- **Partial** means the route was correct but the answer was incomplete or refused due to missing evidence.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    report = build_report()
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"Report written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
