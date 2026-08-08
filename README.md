# Assignment 8: Multi-Agent RAG System

A multi-agent assistant built with LangGraph. It routes requests to specialists, applies safety guardrails, retrieves product information through an MCP-connected RAG service, compares products with local tools, and checks food recalls through openFDA.

## Architecture

```text
User → Input Guard → Supervisor → Specialist(s) → Output Guard → Response
                                ↳ Retrieval Agent   (MCP / RAG)
                                ↳ General Agent     (Gemini)
                                ↳ Comparison Agent  (local ranking tool)
                                ↳ Recall Agent      (openFDA API)
```

The supervisor plans the full route before running specialists. A request may use multiple specialists in order, but the same specialist cannot run twice. Comparison always runs after retrieval.

## Main Components

- **Input Guard:** Blocks unsafe, ambiguous, or invalid input.
- **Supervisor:** Plans and controls specialist routing.
- **Retrieval Agent:** Handles product and document queries through MCP (`ask_rag`, `retrieve_rag`).
- **Comparison Agent:** Ranks products using retrieved evidence and the local `product_decision_analyzer` tool.
- **Recall Agent:** Searches live openFDA food enforcement reports via `product_recall_checker`.
- **General Agent:** Handles general knowledge, AI, programming, and writing.
- **Output Guard:** Blocks empty or potentially sensitive responses.
- **Human Approval:** Required before using the RAG system.
- **Iteration Cap:** Prevents excessive routing attempts (max 4).

## Tools

| Tool | Location | Used by |
|------|----------|---------|
| `ask_rag`, `retrieve_rag` | MCP server (`mcp_server/server.py`) | Retrieval agent |
| `product_decision_analyzer` | `tools/product_decision_tool.py` | Comparison agent |
| `product_recall_checker` | `tools/product_recall_tool.py` | Recall agent |

MCP exposes only the RAG tools. Product decision and recall tools run locally inside the agent process.

## Example Routes

```text
"What ingredients are listed for Coca-Cola?"
→ retrieval_agent
```

```text
"Compare the available Coca-Cola product records."
→ retrieval_agent → comparison_agent
```

```text
"Check openFDA for food recalls involving salmonella."
→ recall_agent
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
  llm/            Gemini model configuration
  mcp/            MCP client
  nodes/          Supervisor and specialist agents
  state/          Shared agent state

api/              FastAPI server and MySQL chat history
evaluation/       Evaluation scripts and results
guards/           Input and output guardrails
mcp_server/       MCP server exposing RAG tools
tools/            Local LangChain tools (decision, recall)
main.py           CLI entry point
docker-compose.yml
requirements.txt
```

## Requirements

- Python 3.11+
- Docker
- A compatible RAG project (with Qdrant running locally)
- MySQL (for Open WebUI chat history via the API)
- Google AI API key (Gemini)

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
pip install fastapi uvicorn
```

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key
MCP_API_TOKEN=your_token
RAG_PROJECT_PATH=path_to_your_rag_project
HF_CACHE_PATH=path_to_huggingface_cache


# MySQL (Open WebUI chat history)
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=assignment8
MYSQL_PASSWORD=assignment8
MYSQL_DATABASE=assignment8_chat
```

`RAG_PROJECT_PATH` should point to a project containing:

```text
src/rag_service.py
```

## Start the Services

Start Qdrant (required by the RAG project) and the MCP/RAG server:

```bash
docker compose up --build
```

The application expects:

- MCP at `http://127.0.0.1:8000/mcp`
- Qdrant at `http://localhost:6333` (via `host.docker.internal` inside the MCP container)

## Run the CLI

```bash
python main.py
```

Retrieval requests pause for user approval before accessing the RAG system.

## Run the API

```bash
uvicorn api.server:app --reload --port 8080
```

### Native endpoints

- `POST /chat`: submit a message
- `POST /resume`: approve or reject a retrieval request
- `GET /history/{conversation_id}`: inspect stored chat history

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

### Open WebUI (OpenAI-compatible)

The API also exposes:

- `GET /v1/models`
- `POST /v1/chat/completions`

In Open WebUI, add a connection pointing to:

```text
http://127.0.0.1:8080/v1
```

Use model ID: `assignment8-agent`


