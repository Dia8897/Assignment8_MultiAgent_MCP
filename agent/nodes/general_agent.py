import os

from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
from agent.llm.ai_model import llm

from agent.state.agent_state import AgentState



def general_agent(state: AgentState):
    task = state["remaining_task"] or state["message"]

    prompt = f"""
        You are a general-purpose specialist

        Answer only general questions that do not require the product retrieval system

        Keep the answer clear and concise

        Prior conversation (context only):
        {state["conversation_history"] or "No prior conversation."}

        Original user question:
        {state["message"]}

        Assigned task:
        {task}
    """

    response = llm.invoke(prompt).text.strip()
    
    updated_results = state["specialist_results"] + [response]
    updated_completed_agents = state["completed_agents"] + ["general_agent"]

    return {
        "response": response,
        "selected_agent": "general_agent",
        "specialist_results": updated_results,
        "completed_agents": updated_completed_agents,
        "remaining_task": "",
    }


