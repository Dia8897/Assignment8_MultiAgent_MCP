import asyncio
from agent.graph.agent_graph import graph
from uuid import uuid4
from langgraph.types import Command



def create_initial_state(
    user_message: str,
    conversation_history: str = "",
) -> dict:
    return {
        "message": user_message,
        "conversation_history": conversation_history,
        "retrieved_evidence": "",
        "response": "",
        "next_agent": "",
        "selected_agent": "",
        "iteration_count": 0,
        "specialist_results": [],
        "input_safe": True,
        "output_safe": True,
        "guard_message": "",
        "input_classification": "",
        "completed_agents": [],
        "required_agents": [],
        "agent_tasks": {},
        "remaining_task": user_message,
    }

async def run_agent(user_message: str, conversation_history: str = ""):
    initial_state = create_initial_state(user_message, conversation_history)

    config = {
            "configurable": {
                "thread_id": str(uuid4())
            }
        }

    final_state = await graph.ainvoke(initial_state, config=config)

    return final_state,config

async def main():

    user_message=input("You: ")

    final_state,config = await run_agent(user_message)



    if "__interrupt__" in final_state:
        interrupt_data = final_state["__interrupt__"][0]
        print(interrupt_data.value)

        user_approval = input("Approve? (yes/no): ").strip().lower()
        approved = user_approval in {"yes", "y"}

        final_state = await graph.ainvoke(
            Command(resume=approved),
            config=config,
        )

        
    print("Handled by:", final_state["selected_agent"])
    print("Final route:", final_state["next_agent"])
    print("Response:", final_state["response"])


if __name__ == "__main__":
    asyncio.run(main())
