import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from agent.mcp.client import create_mcp_client


EVALUATION_DIR = Path(__file__).resolve().parent
TESTSET_PATH = EVALUATION_DIR / "ragas_testset.json"
COLLECTED_PATH = EVALUATION_DIR / "ragas_collected_dataset.json"
RESULTS_PATH = EVALUATION_DIR / "ragas_results.csv"
SUMMARY_PATH = EVALUATION_DIR / "ragas_summary.json"


def _content_blocks_text(tool_result: Any) -> str:
    if isinstance(tool_result, str):
        return tool_result
    if isinstance(tool_result, list):
        for block in tool_result:
            if isinstance(block, dict) and "text" in block:
                return str(block["text"])
    raise ValueError("MCP returned an unsupported content-block format")


def _context_text(document: Any) -> str:
    if isinstance(document, str):
        return document
    if not isinstance(document, dict):
        return str(document)

    for key in ("page_content", "content", "text", "document"):
        value = document.get(key)
        if isinstance(value, str) and value.strip():
            metadata = document.get("metadata")
            if metadata:
                return f"{value}\nMetadata: {json.dumps(metadata, default=str)}"
            return value
    return json.dumps(document, ensure_ascii=False, default=str)


async def collect_rag_samples(test_cases: list[dict]) -> list[dict]:
    client = create_mcp_client()
    tools = await client.get_tools()
    ask_rag = next(tool for tool in tools if tool.name == "ask_rag")
    samples = []

    for index, test_case in enumerate(test_cases, start=1):
        question = test_case["user_input"]
        print(f"Collecting RAG sample {index}/{len(test_cases)}: {question}")
        raw_result = await ask_rag.ainvoke({"query": question})
        payload = json.loads(_content_blocks_text(raw_result))
        documents = payload.get("documents") or payload.get("contexts") or []
        contexts = [_context_text(document) for document in documents]
        if not contexts:
            raise ValueError(f"RAG returned no contexts for: {question}")

        samples.append(
            {
                "user_input": question,
                "response": str(payload["answer"]),
                "retrieved_contexts": contexts,
                "reference": test_case["reference"],
            }
        )
    return samples


def evaluator_components(api_key: str):
    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=api_key,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )
    )
    return evaluator_llm, evaluator_embeddings


def aggregate_scores(results_frame) -> dict[str, float]:
    metric_columns = [
        "faithfulness",
        "answer_relevancy",
        "llm_context_precision_with_reference",
        "context_recall",
    ]
    return {
        column: round(float(results_frame[column].mean()), 4)
        for column in metric_columns
        if column in results_frame.columns
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MCP/RAG with RAGAS")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N test cases.",
    )
    parser.add_argument(
        "--reuse-collected",
        action="store_true",
        help="Skip MCP retrieval and evaluate the previously collected dataset.",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required for RAGAS evaluation")

    test_cases = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    if args.limit is not None:
        test_cases = test_cases[:max(1, args.limit)]

    if args.reuse_collected:
        if not COLLECTED_PATH.exists():
            raise FileNotFoundError(
                "No collected dataset exists. Run once without --reuse-collected."
            )
        samples = json.loads(COLLECTED_PATH.read_text(encoding="utf-8"))
        if args.limit is not None:
            samples = samples[:max(1, args.limit)]
        print(f"Reusing {len(samples)} previously collected RAG sample(s).")
    else:
        samples = asyncio.run(collect_rag_samples(test_cases))
        COLLECTED_PATH.write_text(
            json.dumps(samples, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    dataset = EvaluationDataset.from_list(samples)
    evaluator_llm, evaluator_embeddings = evaluator_components(api_key)
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            strictness=1,
        ),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
    ]
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=RunConfig(
            timeout=120,
            max_retries=2,
            max_wait=20,
            max_workers=2,
        ),
        raise_exceptions=False,
    )
    results_frame = result.to_pandas()
    results_frame.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")

    summary = aggregate_scores(results_frame)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("\nAverage RAGAS scores:")
    for metric, score in summary.items():
        print(f"- {metric}: {score:.4f}")
    print(f"\nDetailed results: {RESULTS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
