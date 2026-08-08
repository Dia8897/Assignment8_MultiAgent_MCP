from agent.state.agent_state import AgentState
from agent.llm.ai_model import llm
import json

ALLOWED_AGENTS = {
    "retrieval_agent",
    "general_agent",
    "comparison_agent",
    "recall_agent",
}
MAX_ITERATIONS = 4

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
            Plan the complete route using only these specialists:
            - general_agent: general knowledge, programming, writing, explanations.
            - retrieval_agent: product/document facts from the local RAG system.
            - comparison_agent: product ranking, unit price, weighted comparison,
              or discount analysis; it must follow retrieval_agent.
            - recall_agent: live or historical openFDA food recall searches; no RAG.

            Required routes:
            - Product fact question -> ["retrieval_agent"]
            - General question -> ["general_agent"]
            - Product comparison -> ["retrieval_agent", "comparison_agent"]
            - FDA recall question -> ["recall_agent"]
            Use additional agents only for explicitly separate tasks. Preserve order.

            Return exactly one JSON object and no markdown:
            {{
            "required_agents": ["retrieval_agent"],
            "agent_tasks": {{
                "retrieval_agent": "focused task"
            }}
            }}

            Prior conversation for resolving references:
            {state["conversation_history"] or "No prior conversation."}

            User question:
            {state["message"]}
        """

        raw_plan = llm.invoke(prompt).text.strip()
        if raw_plan.startswith("```"):
            raw_plan = raw_plan.removeprefix("```json").removeprefix("```")
            raw_plan = raw_plan.removesuffix("```").strip()

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

        # Structural invariant: comparison consumes retrieval evidence.
        if "comparison_agent" in validated_agents:
            without_comparison = [
                agent_name
                for agent_name in validated_agents
                if agent_name not in {"retrieval_agent", "comparison_agent"}
            ]
            validated_agents = [
                "retrieval_agent",
                "comparison_agent",
                *without_comparison,
            ]

        # Safe fallback: preserve the original general-agent fallback.
        if not validated_agents:
            validated_agents = ["general_agent"]
            proposed_tasks = {
                "general_agent": state["message"],
            }

        # Keep plans within the graph's iteration cap.
        validated_agents = validated_agents[:MAX_ITERATIONS]

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
