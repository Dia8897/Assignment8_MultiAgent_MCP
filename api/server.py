from fastapi import FastAPI
from pydantic import BaseModel

from agent.agent import run_agent
from uuid import uuid4
from agent.graph.agent_graph import graph
from langgraph.types import Command

pending_requests={}

app=FastAPI(title="Assignment 8  Agent API")

class ChatRequest(BaseModel):
    message:str

class ResumeRequest(BaseModel):
    request_id:str
    approved:bool



@app.post("/chat")
async def chat(request:ChatRequest):
    final_state, config =await run_agent(request.message)


    if "__interrupt__" in final_state:
        request_id = str(uuid4())

        pending_requests[request_id] = {
            "config": config
        }

        interrupt_data = final_state["__interrupt__"][0]

        return {
            "status": "approval_required",
            "request_id": request_id,
            "message": interrupt_data.value,
        }   

    
    return {
        "status": "completed",
        "response": final_state["response"],
        "handled_by": final_state["selected_agent"],
    }


@app.post("/resume")
async def resume(request:ResumeRequest):
    pending_request = pending_requests.get(request.request_id) #get() returns 'None' if the ID doesnt exist instead of crashing
    if pending_request is None:
        return {
            "status": "error",
            "message": "Unknown request ID."
        }

    config = pending_request["config"]

    final_state = await graph.ainvoke(
        Command(resume=request.approved),
        config=config,
    )

    pending_requests.pop(request.request_id, None)

    return {
        "status": "completed",
        "response": final_state["response"],
        "handled_by": final_state["selected_agent"],
    }

