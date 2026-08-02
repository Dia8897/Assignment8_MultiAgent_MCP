# hybrid approach:
# rules + LLM classifier

from agent.state.agent_state import AgentState
from guards.input_classifier import classify_input
MAX_INPUT_LENGTH = 1000

BLOCKED_PHRASES = [
    "ignore previous instructions",
    "forget previous instructions",
    "reveal the system prompt",
    "show the system prompt",
]

def input_guard(state:AgentState):
    message=state["message"].strip()
    message_lower = message.lower()

    if not message:
        guard_message="The input cannot be empty"
        return{
            "input_safe": False,
            "guard_message": guard_message,
            "response": guard_message,
            "next_agent": "end",
        }

    if len(message) > MAX_INPUT_LENGTH:
        guard_message = (
            f"The input is too long. The maximum allowed length is "
            f"{MAX_INPUT_LENGTH} characters."
        )

        return {
            "input_safe": False,
            "guard_message": guard_message,
            "response": guard_message,
            "next_agent": "end",
        }
    matched_phrase=next(
        (
            phrase
            for phrase in BLOCKED_PHRASES
            if phrase in message_lower
        ),
        None
    )  
    if matched_phrase:
        guard_message=(
            "The request was blocked because it appears to contain "
            "a prompt-injection attempt."            
        )  
        return {
            "input_safe": False,
            "guard_message": guard_message,
            "response": guard_message,
            "next_agent": "end",
        }


    classification = classify_input(message)
    if classification == "UNSAFE":
        guard_message = (
            "The request was blocked because it was classified as unsafe."
        )

        return {
            "input_safe": False,
            "input_classification": classification,
            "guard_message": guard_message,
            "response": guard_message,
            "next_agent": "end",
        } 

    if classification == "AMBIGUOUS":
        guard_message = (
            "The request is ambiguous. Please clarify what you would like to know."
        )

        return {
            "input_safe": False,
            "input_classification": classification,
            "guard_message": guard_message,
            "response": guard_message,
            "next_agent": "end",
        }  
     
    return {
        "input_safe": True,
        "input_classification": classification,
        "guard_message": "",
    }  
