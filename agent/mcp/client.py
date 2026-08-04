from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
from dotenv import load_dotenv

load_dotenv()
MCP_API_TOKEN = os.getenv("MCP_API_TOKEN")  

PROJECT_ROOT=Path(__file__).resolve().parents[2]
SERVER_PATH = PROJECT_ROOT / "mcp_server" / "server.py"

# create the mcp client used by LangChain or LangGraph
def create_mcp_client()->MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            # "rag_server":{
            #     "transport": "stdio",
            #     # means the client launches the server as a local subprocess and communicates through standard input/output


            #     "command": "fastmcp",
            #     "args": [
            #         "run",
            #         str(SERVER_PATH),
            #         "--no-banner",
            #     ],
            # }

            "rag_server": {
                "transport": "http",
                "url": "http://127.0.0.1:8000/mcp",
                "headers": {
                    "Authorization": f"Bearer {MCP_API_TOKEN}"
                },
            }
        }
    )



import asyncio


async def main():
    client = create_mcp_client()

    tools = await client.get_tools()

    print("Available tools:")
    for tool in tools:
        print("-", tool.name)


if __name__ == "__main__":
    asyncio.run(main())