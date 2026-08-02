from agent.state.agent_state import AgentState

BLOCKED_OUTPUT_PHRASES = [
    "system prompt",
    "api key",
    "secret key",
    "password",
]

def output_guard(state:AgentState):
    response=state["response"].strip()
    response_lower=response.lower()

    if not response:
        guard_message="The agent produced an empty response."
        return{
            "output_safe": False,
            "guard_message": guard_message,
            "response": guard_message,
            "next_agent": "end",            
        }

    matched_phrase=next(
        (
            phrase
            for phrase in BLOCKED_OUTPUT_PHRASES
            if phrase in response_lower
        ),
        None,
    )

    if matched_phrase:
        guard_message=(
            "The response was blocked because it may expose "
            "sensitive information."            
        )
        return {
            "output_safe": False,
            "guard_message": guard_message,
            "response": guard_message,
            "next_agent": "end",
        }


    return {
        "output_safe": True,
        "guard_message": "",
    }    
