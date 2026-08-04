from agent.state.agent_state import AgentState
from agent.llm.ai_model import llm
import json

ALLOWED_AGENTS = {
    "retrieval_agent",
    "general_agent",
}
MAX_ITERATIONS = 3

def combine_results(results: list[str]) -> str:
    return "\n\n".join(
        result.strip()
        for result in results
        if result and result.strip()
    )

def supervisor(state:AgentState):

    completed_agents = set(state["completed_agents"])
    required_agents = state["required_agents"]
    agent_tasks = state["agent_tasks"]
    combined_response = combine_results(state["specialist_results"])

   
        
        
    if not required_agents:
        prompt = f"""
            You are a routing planner.

            Create the complete specialist plan before any specialist runs.

            retrieval_agent handles:
            - Product prices, ingredients, labels, brands, availability, stores,
            locations, barcodes, product records, uploaded documents, and
            questions requiring evidence from the RAG datasets.

            general_agent handles:
            - General knowledge, AI, programming, explanations, writing,
            reasoning, and information not found through product retrieval.

            Rules:
            - Use exactly one specialist when that specialist can answer the
            complete request.
            - Use both specialists only when the user explicitly asks for
            separate retrieval and general-knowledge tasks.
            - Do not add a second specialist merely to expand, rewrite, verify,
            summarize, or comment on a complete answer.
            - Preserve the order requested by the user.
            - Each task must contain only the portion assigned to that specialist.

            Examples:

            "What ingredients are listed for Coca-Cola?"
            -> retrieval_agent only

            "What is an AI agent?"
            -> general_agent only

            "What is the difference between an LLM and an embedding model?"
            -> general_agent only

            "Find the available Coca-Cola ingredients, then explain why ingredient
            lists may vary between countries."
            -> retrieval_agent, then general_agent

            Reply with exactly one JSON object:
            {{
            "required_agents": ["retrieval_agent"],
            "agent_tasks": {{
                "retrieval_agent": "focused task"
            }}
            }}

            Use a two-item required_agents list only for a genuine multi-part request.

            User question:
            {state["message"]}
        """

        raw_plan = llm.invoke(prompt).text.strip()

        try:
            parsed_plan = json.loads(raw_plan)
            proposed_agents = parsed_plan.get("required_agents", [])
            proposed_tasks = parsed_plan.get("agent_tasks", {})
        except (json.JSONDecodeError, AttributeError):
            proposed_agents = []
            proposed_tasks = {}

        validated_agents = []

        if isinstance(proposed_agents, list):
            for agent_name in proposed_agents:
                if (
                    agent_name in ALLOWED_AGENTS
                    and agent_name not in validated_agents
                ):
                    validated_agents.append(agent_name)

        # Safe fallback: preserve the original general-agent fallback.
        if not validated_agents:
            validated_agents = ["general_agent"]
            proposed_tasks = {
                "general_agent": state["message"],
            }

        # There are only two specialists; reject oversized plans.
        validated_agents = validated_agents[:2]

        if not isinstance(proposed_tasks, dict):
            proposed_tasks = {}

        validated_tasks = {
            agent_name: (
                proposed_tasks.get(agent_name, state["message"])
                if isinstance(proposed_tasks.get(agent_name), str)
                else state["message"]
            )
            for agent_name in validated_agents
        }

        required_agents = validated_agents
        agent_tasks = validated_tasks

    pending_agents = [
        agent_name
        for agent_name in required_agents
        if agent_name not in completed_agents
    ]

    if not pending_agents:
        return {
            "next_agent": "end",
            "response": combined_response,
            "remaining_task": "",
        }

    if state["iteration_count"] >= MAX_ITERATIONS:
        response = combined_response

        if response:
            response += "\n\n"

        response += "The maximum number of routing attempts was reached."

        return {
            "next_agent": "end",
            "response": response,
            "remaining_task": "",
        }

    decision = pending_agents[0]
    remaining_task = agent_tasks.get(decision, state["message"])

    if not isinstance(remaining_task, str) or not remaining_task.strip():
        remaining_task = state["message"]

    return {
        "next_agent": decision,
        "required_agents": required_agents,
        "agent_tasks": agent_tasks,
        "remaining_task": remaining_task.strip(),
        "iteration_count": state["iteration_count"] + 1,
    }
