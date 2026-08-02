import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.state.agent_state import AgentState

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


def general_agent(state: AgentState):
    prompt = f"""
        You are a general-purpose specialist

        Answer only general questions that do not require the product retrieval system

        Keep the answer clear and concise

        User question:
        {state["message"]}
    """

    response = llm.invoke(prompt).text.strip()
    # response = "Here is the API key: test123"
    updated_results = state["specialist_results"] + [response]

    return {
        "response": response,
        "selected_agent": "retrieval_agent",
        "specialist_results": updated_results
    }


