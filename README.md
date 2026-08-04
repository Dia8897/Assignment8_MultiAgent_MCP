# Assignment 8: Multi-Agent RAG System

A multi-agent assistant built with LangGraph. It routes requests to specialists, applies safety guardrails, and retrieves product information through an MCP-connected RAG service.

## Architecture

```text
User → Input Guard → Supervisor → Specialist → Output Guard → Response
                                ↳ Retrieval Agent
                                ↳ General Agent
```

The supervisor creates a routing plan before running specialists. A request can use one or both specialists, but the same specialist cannot run twice.

## Main Components

- **Input Guard:** Blocks unsafe, ambiguous, or invalid input.
- **Supervisor:** Plans and controls specialist routing.
- **Retrieval Agent:** Handles product and document queries through MCP and RAG.
- **General Agent:** Handles general knowledge, AI, programming, and writing.
- **Output Guard:** Blocks empty or potentially sensitive responses.
- **Human Approval:** Required before using the retrieval system.
- **Iteration Cap:** Prevents excessive routing attempts.

## Example Routes

```text
"What ingredients are listed for Coca-Cola?"
→ retrieval_agent
```

```text
"What is an AI agent?"
→ general_agent
```

```text
"Find Coca-Cola ingredients, then explain why ingredient lists vary by country."
→ retrieval_agent → general_agent
```

## Project Structure

```text
agent/
  graph/          LangGraph definition
  llm/            Ollama model configuration
  mcp/            MCP client
  nodes/          Supervisor and specialist agents
  state/          Shared agent state

api/              FastAPI server
evaluation/       Evaluation suite and results
guards/           Input and output guardrails
mcp_server/       MCP server exposing RAG tools
rag/              RAG-related files
tools/            Custom tools
main.py           CLI entry point
docker-compose.yml
requirements.txt
```

## Requirements

- Python 3.11+
- Docker
- Ollama
- A compatible RAG project

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the Ollama model:

```bash
ollama pull qwen2.5:7b
```

Create a `.env` file:

```env
MCP_API_TOKEN=your_token
RAG_PROJECT_PATH=path_to_your_rag_project
```

`RAG_PROJECT_PATH` should point to a project containing:

```text
src/rag_service.py
```

## Start the Services

Start Ollama:

```bash
ollama serve
```

Start the MCP/RAG server:

```bash
docker compose up --build
```

The application expects:

- Ollama at `http://localhost:11434`
- MCP at `http://127.0.0.1:8000/mcp`

## Run the CLI

```bash
python main.py
```

Retrieval requests pause for user approval before accessing the RAG system.

## Run the API

```bash
uvicorn api.server:app --reload
```

Available endpoints:

- `POST /chat` — submit a message
- `POST /resume` — approve or reject a retrieval request

Example `/chat` request:

```json
{
  "message": "What ingredients are listed for Coca-Cola?"
}
```

Example `/resume` request:

```json
{
  "request_id": "returned-request-id",
  "approved": true
}
```

## Run the Evaluation

```bash
python -m evaluation.run_evaluation
```

The evaluation covers retrieval, general knowledge, multi-step routing, and input guardrails.

Results are saved to:

```text
evaluation/evaluation_results.csv
```