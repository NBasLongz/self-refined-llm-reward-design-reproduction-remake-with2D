"""List free models on OpenRouter."""
from openai import OpenAI
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)
models = client.models.list()
free_models = [m for m in models.data if ":free" in m.id]
for m in sorted(free_models, key=lambda x: x.id):
    print(m.id)
