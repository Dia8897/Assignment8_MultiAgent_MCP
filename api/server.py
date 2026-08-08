from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent.agent import run_agent
from uuid import uuid4
from agent.graph.agent_graph import graph
from agent.llm.ai_model import llm
from langgraph.types import Command
import json
import re
import time
from typing import Any
from fastapi.responses import StreamingResponse
from api.database import (
    find_conversation_by_assistant_content,
    get_conversation_history,
    get_conversation_summary,
    save_conversation_summary,
    save_message,
)

pending_requests={}

app=FastAPI(title="Assignment 8  Agent API")

class ChatRequest(BaseModel):
    message:str

class ResumeRequest(BaseModel):
    request_id:str
    approved:bool


class OpenAIMessage(BaseModel):
    role: str
    content: Any


class OpenAIChatRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    stream: bool = False
    chat_id: str | None = None
    id: str | None = None
    message_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)



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

MODEL_ID = "assignment8-agent"
SUMMARY_MESSAGE_THRESHOLD = 8
SUMMARY_CHARACTER_THRESHOLD = 3000
RECENT_MESSAGES_TO_KEEP = 4


def request_identifiers(request: OpenAIChatRequest) -> tuple[str, str]:
    conversation_id = (
        request.chat_id
        or request.metadata.get("chat_id")
        or request.session_id
        or request.metadata.get("session_id")
    )

    # Open WebUI may remove chat_id before forwarding an OpenAI-compatible
    # request. Its request still contains the full message history, so use the
    # immediately preceding assistant reply to reconnect to the saved chat.
    if not conversation_id and len(request.messages) >= 2:
        previous_message = request.messages[-2]
        if previous_message.role == "assistant":
            conversation_id = find_conversation_by_assistant_content(
                str(previous_message.content)
            )

    if not conversation_id:
        conversation_id = str(uuid4())

    message_id = (
        request.id
        or request.message_id
        or request.metadata.get("message_id")
        or str(uuid4())
    )
    return str(conversation_id), str(message_id)


def persist_exchange_message(
    conversation_id: str,
    message_id: str,
    model: str,
    role: str,
    content: str,
    message_type: str = "chat",
    enabled: bool = True,
) -> None:
    if not enabled:
        return

    save_message(
        conversation_external_id=conversation_id,
        message_external_id=f"{message_id}:{role}",
        model=model,
        role=role,
        content=content,
        message_type=message_type,
    )


def mysql_conversation_context(conversation_id: str) -> str:
    history = get_conversation_history(conversation_id)

    # The current user message was just saved; it is already passed separately
    # to the graph and must not be duplicated in the history context.
    prior_messages = [
        message
        for message in history[:-1]
        if message["message_type"] != "error"
    ]
    summary_record = get_conversation_summary(conversation_id)
    existing_summary = summary_record["summary"] if summary_record else ""
    summarized_through = (
        int(summary_record["through_message_id"])
        if summary_record
        else 0
    )
    unsummarized = [
        message
        for message in prior_messages
        if int(message["id"]) > summarized_through
    ]

    unsummarized_characters = sum(
        len(str(message["content"])) for message in unsummarized
    )
    should_summarize = (
        len(unsummarized) > SUMMARY_MESSAGE_THRESHOLD
        or unsummarized_characters > SUMMARY_CHARACTER_THRESHOLD
    )

    if should_summarize and len(unsummarized) > RECENT_MESSAGES_TO_KEEP:
        messages_to_summarize = unsummarized[:-RECENT_MESSAGES_TO_KEEP]
        transcript = "\n".join(
            f'{message["role"].capitalize()}: {message["content"]}'
            for message in messages_to_summarize
        )
        prompt = f"""
            You summarize conversation memory for another AI agent.

            Produce a concise factual summary that preserves:
            - user-provided names, preferences, and personal details;
            - questions, decisions, approvals, and completed actions;
            - product names and retrieval results needed for follow-up questions;
            - unresolved requests or important constraints.

            Do not invent information. Do not include UUIDs or approval markers.

            Existing summary:
            {existing_summary or "No existing summary."}

            New conversation turns to merge into the summary:
            {transcript}
        """
        summary_response = llm.invoke(prompt)
        existing_summary = str(
            getattr(summary_response, "text", "")
            or getattr(summary_response, "content", "")
        ).strip()
        summarized_through = int(messages_to_summarize[-1]["id"])
        save_conversation_summary(
            conversation_id,
            existing_summary,
            summarized_through,
        )
        unsummarized = [
            message
            for message in prior_messages
            if int(message["id"]) > summarized_through
        ]

    recent_context = "\n".join(
        f'{message["role"].capitalize()}: {message["content"]}'
        for message in unsummarized
    )
    context_sections = []
    if existing_summary:
        context_sections.append(f"Conversation summary:\n{existing_summary}")
    if recent_context:
        context_sections.append(f"Recent conversation:\n{recent_context}")

    return "\n\n".join(context_sections)


def openai_response(content: str, model: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
    }


def stream_response(content: str, model: str):
    completion_id = f"chatcmpl-{uuid4()}"
    created = int(time.time())

    first_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": None,
            }
        ],
    }

    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }

    yield f"data: {json.dumps(first_chunk)}\n\n"
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


def send_openai_response(content: str, model: str, stream: bool):
    if stream:
        return StreamingResponse(
            stream_response(content, model),
            media_type="text/event-stream",
        )

    return openai_response(content, model)


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "assignment8",
            }
        ],
    }


@app.get("/history/{conversation_id}")
async def conversation_history(conversation_id: str):
    return {
        "conversation_id": conversation_id,
        "summary": get_conversation_summary(conversation_id),
        "messages": get_conversation_history(conversation_id),
    }


@app.post("/v1/chat/completions")
async def openai_chat(request: OpenAIChatRequest):
    if not request.messages:
        return send_openai_response(
            "No message was provided.",
            request.model,
            request.stream,
        )

    latest_message = request.messages[-1]
    user_message = str(latest_message.content)
    persist_history = not user_message.lstrip().startswith("### Task:")
    conversation_id, message_id = request_identifiers(request)

    persist_exchange_message(
        conversation_id,
        message_id,
        request.model,
        "user",
        user_message,
        enabled=persist_history,
    )
    conversation_context = (
        mysql_conversation_context(conversation_id)
        if persist_history
        else ""
    )

    # Handle approval when the previous assistant message contains a request ID.
    if len(request.messages) >= 2:
        previous_message = request.messages[-2]

        if previous_message.role == "assistant":
            match = re.search(
                r"\[APPROVAL_REQUIRED:([a-f0-9-]+)\]",
                str(previous_message.content),
            )

            if match:
                request_id = match.group(1)
                pending_request = pending_requests.get(request_id)

                if pending_request is None:
                    content = "That approval request has expired. Please ask the question again."
                    persist_exchange_message(
                        conversation_id,
                        message_id,
                        request.model,
                        "assistant",
                        content,
                        "error",
                        persist_history,
                    )
                    return send_openai_response(
                        content,
                        request.model,
                        request.stream,
                    )

                answer = user_message.strip().lower()

                if answer not in {"yes", "y", "no", "n"}:
                    content = "Please answer yes or no to the retrieval request."
                    persist_exchange_message(
                        conversation_id,
                        message_id,
                        request.model,
                        "assistant",
                        content,
                        "approval",
                        persist_history,
                    )
                    return send_openai_response(
                        content,
                        request.model,
                        request.stream,
                    )

                approved = answer in {"yes", "y"}

                final_state = await graph.ainvoke(
                    Command(resume=approved),
                    config=pending_request["config"],
                )

                pending_requests.pop(request_id, None)
                content = final_state["response"]

                persist_exchange_message(
                    conversation_id,
                    message_id,
                    request.model,
                    "assistant",
                    content,
                    "approval_result",
                    persist_history,
                )

                return send_openai_response(
                    content,
                    request.model,
                    request.stream,
                )

    # Start a new agent request.
    final_state, config = await run_agent(
        user_message,
        conversation_context,
    )

    if "__interrupt__" in final_state:
        request_id = str(uuid4())
        pending_requests[request_id] = {"config": config}

        interrupt_data = final_state["__interrupt__"][0]

        content = (
            f"{interrupt_data.value}\n\n"
            "Reply **yes** to approve or **no** to reject.\n\n"
            f"[APPROVAL_REQUIRED:{request_id}]"
        )

        persist_exchange_message(
            conversation_id,
            message_id,
            request.model,
            "assistant",
            content,
            "approval_request",
            persist_history,
        )

        return send_openai_response(
            content,
            request.model,
            request.stream,
        )

    content = final_state["response"]
    persist_exchange_message(
        conversation_id,
        message_id,
        request.model,
        "assistant",
        content,
        enabled=persist_history,
    )

    return send_openai_response(
        content,
        request.model,
        request.stream,
    )

