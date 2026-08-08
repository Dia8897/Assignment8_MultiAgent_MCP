# Evaluation Report

Generated: 2026-08-09 01:58

Combined summary of RAGAS (RAG layer) and end-to-end multi-agent evaluation.

## Overview

- **RAGAS test cases:** 3
- **End-to-end test cases:** 12
- **Routing matches:** 12/12
- **Passing responses:** 11/12
- **Partial responses:** 1/12
- **Execution errors:** 0/12

## RAGAS Summary (RAG / MCP)

| Metric | Score |
|--------|-------|
| Faithfulness | 1.0000 |
| Answer Relevancy | 0.8648 |
| Context Precision | 0.5000 |
| Context Recall | 0.3333 |

## RAGAS Per Question

| Question | Faithfulness | Relevancy | Precision | Recall |
|----------|--------------|-----------|-----------|--------|
| What ingredients are listed for Coca-Cola? | 1.0000 | 0.8408 | 1.0000 | 1.0000 |
| In which countries were Dove products found? | 1.0000 | 0.8517 | 0.0000 | 0.0000 |
| Are any Nutella products listed with a discounted pr... | 1.0000 | 0.9018 | 0.5000 | 0.0000 |

## End-to-End Agent Evaluation

| # | Category | Expected Route | Actual Route | Status | Response Preview |
|---|----------|----------------|--------------|--------|------------------|
| 1 | product ingredient retrieval | retrieval_agent | retrieval_agent | Pass | The ingredient list for Coca-Cola includes: water, sugar, carbon dioxide, col... |
| 2 | product location retrieval | retrieval_agent | retrieval_agent | Pass | Dove products were found in the United States [3] and France [5]. |
| 3 | comparative product retrieval | retrieval_agent -> comparison_agent | retrieval_agent -> comparison_agent | Partial | A reliable ranking cannot be calculated because fewer than two products have ... |
| 4 | live external recall lookup | recall_agent | recall_agent | Pass | Found 2 openFDA report(s) matching "salmonella". A matching report may descri... |
| 5 | document and metadata retrieval | retrieval_agent | retrieval_agent | Pass | The product "4X100G POMME FRAISE AND" has a price of 1.69 EUR [1]. The produc... |
| 6 | general knowledge | general_agent | general_agent | Pass | An AI agent is an autonomous software program powered by artificial intellige... |
| 7 | general technical knowledge | general_agent | general_agent | Pass | A vector database is designed to store and search data (like text, images, or... |
| 8 | general technical comparison | general_agent | general_agent | Pass | An **LLM (Large Language Model)** is designed for **text generation and reaso... |
| 9 | general AI knowledge | general_agent | general_agent | Pass | Guardrails are useful in AI applications because they act as safety boundarie... |
| 10 | out-of-scope general request | general_agent | general_agent | Pass | A multi-agent system is a computerized setup composed of multiple interacting... |
| 11 | multi-step specialist routing | retrieval_agent -> general_agent | retrieval_agent -> general_agent | Pass | The ingredients for Coca-Cola are water, sugar, carbon dioxide, color E150d, ... |
| 12 | adversarial input | input_guard -> end | input_guard -> end | Pass (blocked) | The request was blocked by regex input guard because it appears to contain a ... |

## Notes

- **RAGAS** evaluates MCP `ask_rag` answer quality only.
- **End-to-end** evaluates routing, guardrails, retrieval, comparison, recall, and general agents.
- **Partial** means the route was correct but the answer was incomplete or refused due to missing evidence.
