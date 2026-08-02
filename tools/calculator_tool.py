from langchain_core.tools import tool

@tool
def calculator(expression:str)->str:
    """Evaluate basic mathematical expression"""
    allowed_characters = set("0123456789+-*/(). ")
    if not set(expression).issubset(allowed_characters):
        return "Invalid Expression"
    try:
        result=eval(expression,{"__builtins__": {}}, {})
        return str(result)
    except Exception as error:
        return f"Calculation error: {error}"