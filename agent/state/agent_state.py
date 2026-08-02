from typing import TypedDict
class AgentState(TypedDict):
    message:str
    response:str
    next_agent:str 
    selected_agent:str #to avoid losing info abt who handled the req when "end"
    iteration_count:int
    specialist_results: list[str]
    input_safe:bool
    output_safe:bool
    guard_message:str
    input_classification:str
    