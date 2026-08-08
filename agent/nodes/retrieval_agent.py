import json
import asyncio
from agent.mcp.client import create_mcp_client
from agent.state.agent_state import AgentState
from langgraph.types import interrupt


def retrieval_failure(state: AgentState, error: Exception) -> dict:
    result = f"The RAG request failed safely: {type(error).__name__}."
    return {
        "response": result,
        "selected_agent": "retrieval_agent",
        "retrieved_evidence": "",
        "specialist_results": state["specialist_results"] + [result],
        "completed_agents": state["completed_agents"] + ["retrieval_agent"],
        "remaining_task": "",
    }


async def retrieval_agent(state:AgentState):
    task = state["remaining_task"] or state["message"]
    approval=interrupt(
        "This action will query the RAG system through the MCP server. Approve?"
    )
    if not approval:
        result = "The retrieval request was cancelled by the user."
        updated_results = state["specialist_results"] + [result]
        updated_completed_agents = (
            state["completed_agents"] + ["retrieval_agent"]
        )
        return {
            "response": result,
            "selected_agent": "retrieval_agent",
            "specialist_results": updated_results,
            "completed_agents": updated_completed_agents,
            "remaining_task": "",
        }

    client=create_mcp_client() #create the connection object

    try:
        tools=await client.get_tools()
    except Exception as error:
        return retrieval_failure(state, error)

    if "comparison_agent" in state["required_agents"]:
        retrieve_tool = next(
            tool for tool in tools
            if tool.name == "retrieve_rag"
        )
        price_query = (
            f"{task}\nFocus on exact product variant, current price, original "
            "price, currency, package size, size unit, and discount fields."
        )
        ingredient_query = (
            f"{task}\nFocus on exact product variant and its complete ingredient "
            "list. Keep every ingredient list attached to the correct variant."
        )
        try:
            price_result, ingredient_result = await asyncio.gather(
                retrieve_tool.ainvoke({"query": price_query, "top_k": 8}),
                retrieve_tool.ainvoke({"query": ingredient_query, "top_k": 8}),
            )
        except Exception as error:
            return retrieval_failure(state, error)
        evidence = json.dumps(
            {
                "price_and_package_evidence": price_result,
                "ingredient_evidence": ingredient_result,
            },
            ensure_ascii=False,
            default=str,
        )
        return {
            "response": "Product evidence retrieved for comparison.",
            "selected_agent": "retrieval_agent",
            "retrieved_evidence": evidence,
            "completed_agents": state["completed_agents"] + ["retrieval_agent"],
            "remaining_task": "",
        }

    # search the tools list and get only the RAG answer tool
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
    try:
        tool_result = await ask_rag_tool.ainvoke(
            {"query": task}
        )
    # ainvoke bc asynchronous -> while waiting, the event loop can run other asynchronous tasks
    # bc it takes time

    # MCP returns a list of content blocks
        text_result = tool_result[0]["text"]

    # Convert the JSON string into a Python dictionary
        parsed_result = json.loads(text_result)

    # Extract only the final RAG answer
        result = parsed_result["answer"]
    except Exception as error:
        return retrieval_failure(state, error)

    updated_results = state["specialist_results"] + [result]
    updated_completed_agents = (
        state["completed_agents"] + ["retrieval_agent"]
    )

    return {
        "response": result,
        "selected_agent": "retrieval_agent",
        "specialist_results": updated_results,
        "completed_agents": updated_completed_agents,
        "remaining_task": "",
    }

