from fastmcp import FastMCP
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

rag_project_path = Path(
    os.getenv("RAG_PROJECT_PATH", "")
).resolve()
if not rag_project_path.exists():
    raise FileNotFoundError(
        f"RAG project not found: {rag_project_path}"
    )

sys.path.insert(0, str(rag_project_path))
# adds the old RAG project to Python’s import search path


from src.rag_service import RAGService


mcp = FastMCP("RAG Server")
rag_service = RAGService()

# @mcp.tool
# def hello(name: str) -> str:
#     """
#     Returns a greeting for the provided name.
#     """
#     return f"Hello, {name}!"

@mcp.tool
def ask_rag(query: str) -> dict:
    """
    Answer a user question using the RAG system
    Returns the generated answer together with the retrieved evidence
    """
    return rag_service.ask_rag(query)

@mcp.tool
def retrieve_rag(query:str, top_k:int=5)->list[dict]:
    """
    Retrieve relevant evidence from the RAG system without generating a final answer
    Use this tool when a client needs the underlying retrieved documents,
    metadata, scores, and dataset information for further reasoning
    """
    result=rag_service.retrieve_rag(query,top_k=top_k)
    return result["documents"]

@mcp.resource("rag://datasets")
def get_rag_datasets()->dict:
    """
    Describe the datasets and collections available through the RAG server

    Clients can read this resource to understand which data sources
    the retrieval tools can search
    """

    return {
        "datasets": {
            "food": "Open Food Facts product information",
            "beauty": "Open Beauty Facts product information",
            "prices": "Retail product price observations",
            "documents": "User-uploaded documents",
        }
    }

# if __name__ == "__main__":
#     mcp.run()