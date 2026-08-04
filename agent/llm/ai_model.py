import os
from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI

# load_dotenv()

# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.6-flash",
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5:7b",
    base_url="http://localhost:11434",
    temperature=0,
)