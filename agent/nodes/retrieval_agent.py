import json
from agent.mcp.client import create_mcp_client
from agent.state.agent_state import AgentState
from langgraph.types import interrupt


async def retrieval_agent(state:AgentState):

    approval=interrupt(
        "This action will query the RAG system through the MCP server. Approve?"
    )
    if not approval:
        result = "The retrieval request was cancelled by the user."
        updated_results = state["specialist_results"] + [result]
        return{
            "response": result,
            "selected_agent": "retrieval_agent",
            "specialist_results": updated_results,
        }

    client=create_mcp_client() #create the connection object

    tools=await client.get_tools()

    # search the tools list and get only the rag tool
    ask_rag_tool=next(
        tool for tool in tools
        if tool.name=="ask_rag"
    )


    # that message goes through the MCP client
    # the client sends:
    # ask_rag
    # 'query'=...
    # and returns: 
    # 'answer':...
    # 'documents':...
    tool_result = await ask_rag_tool.ainvoke(
        {"query": state["message"]}
    )
    # ainvoke bc asynchronous -> while waiting, the event loop can run other asynchronous tasks
    # bc it takes time

    # MCP returns a list of content blocks
    text_result = tool_result[0]["text"]

    # Convert the JSON string into a Python dictionary
    parsed_result = json.loads(text_result)

    # Extract only the final RAG answer
    result = parsed_result["answer"]

    updated_results = state["specialist_results"] + [result]

    return {
        "response": result,
        "selected_agent": "retrieval_agent",
        "specialist_results": updated_results,
    }

