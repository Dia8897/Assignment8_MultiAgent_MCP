import asyncio
from agent.graph.agent_graph import graph
from uuid import uuid4
from langgraph.types import Command


async def main():

    user_message=input("You: ")
    initial_state={
        "message":user_message,
        "response":"",
        "next_agent":"",
        "selected_agent": "",
        "iteration_count": 0,
        "specialist_results": [],
        "input_safe": True,
        "output_safe": True,
        "guard_message": "",   
        "input_classification": "",     
    }

    config = {
        "configurable": {
            "thread_id": str(uuid4())
        }
    }

    result=await graph.ainvoke(initial_state,
                               config=config,
                               )

    if "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0]
        print(interrupt_data.value)

        user_approval = input("Approve? (yes/no): ").strip().lower()
        approved = user_approval in {"yes", "y"}

        result = await graph.ainvoke(
            Command(resume=approved),
            config=config,
        )

        
    print("Handled by:", result["selected_agent"])
    print("Final route:", result["next_agent"])
    print("Response:", result["response"])


if __name__ == "__main__":
    asyncio.run(main())