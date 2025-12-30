from langchain.chat_models import init_chat_model
from os import getenv
from dotenv import load_dotenv

load_dotenv()

# Initialize the model with OpenRouter's base URL
model = init_chat_model(
    model="google/gemini-2.0-flash-lite-001",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
)

# Example usage
response = model.invoke("Who is albert einstein?")
print(response.content)
