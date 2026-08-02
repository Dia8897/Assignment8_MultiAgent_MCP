from langgraph.graph import StateGraph, START, END
from agent.state.agent_state import AgentState
from agent.nodes.supervisor import supervisor
from agent.nodes.general_agent import general_agent
from agent.nodes.retrieval_agent import retrieval_agent
from guards.input_guard import input_guard
from guards.output_guard import output_guard
from langgraph.checkpoint.memory import InMemorySaver


def route_to_specialist(state: AgentState) -> str:
    return state["next_agent"]


def route_after_input_guard(state: AgentState) -> str:
    if state["input_safe"]:
        return "supervisor"

    return "end"


# create the graph
graph_builder=StateGraph(AgentState)

graph_builder.add_node("input_guard", input_guard)
graph_builder.add_node("output_guard", output_guard)
graph_builder.add_node("supervisor",supervisor)
graph_builder.add_node("general_agent",general_agent)
graph_builder.add_node("retrieval_agent", retrieval_agent)

graph_builder.add_conditional_edges(
    "supervisor",
    route_to_specialist,
    {
        "general_agent": "general_agent",
        "retrieval_agent": "retrieval_agent",
        "end": "output_guard",
    },
)

graph_builder.add_conditional_edges(
    "input_guard",
    route_after_input_guard,
    {
        "supervisor": "supervisor",
        "end": END,
    },
)

graph_builder.add_edge(START, "input_guard")
graph_builder.add_edge("general_agent", "supervisor")
graph_builder.add_edge("retrieval_agent", "supervisor")
graph_builder.add_edge("output_guard", END)


memory = InMemorySaver()

# Compile the graph
graph = graph_builder.compile(checkpointer=memory)