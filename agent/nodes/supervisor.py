from agent.state.agent_state import AgentState
from agent.llm.gemini import llm

ALLOWED_AGENTS = {
    "retrieval_agent",
    "general_agent",
}
MAX_ITERATIONS = 3


def supervisor(state:AgentState):
        
        if state["iteration_count"]>=MAX_ITERATIONS:
              return{
                "next_agent": "end",
                "response": (
                    "The maximum number of routing attempts was reached"
                    "Here is the partial result available so far"
                )
              }

        if state["specialist_results"]:
            return {
                "next_agent": "end"
            }
                
        prompt = f"""
            You are a routing supervisor.

            Your only job is to choose which specialist should handle the request.
            Do not answer the user.

            Available agents:

            retrieval_agent:
            - Questions about product prices, ingredients, labels, brands, availability,
            stores, locations, barcodes, food products, beauty products, or uploaded documents.
            - Questions that require evidence from the project's RAG datasets.

            general_agent:
            - General knowledge questions.
            - Questions about AI, programming, LangGraph, agents, writing, explanations,
            brainstorming, or topics unrelated to the product RAG datasets.

            Examples:
            - "What is the price of Nutella?" -> retrieval_agent
            - "What ingredients does this product contain?" -> retrieval_agent
            - "What is an AI agent?" -> general_agent
            - "Explain LangGraph." -> general_agent
            - "Write a short poem." -> general_agent

            Reply with exactly one name:
            retrieval_agent
            general_agent

            User question:
            {state["message"]}
            """

        decision=llm.invoke(prompt).text.strip()
        if decision not in ALLOWED_AGENTS:
            decision = "general_agent"
        current_count=state["iteration_count"]
        return{
                "next_agent":decision,
                "iteration_count":current_count+1
        }
