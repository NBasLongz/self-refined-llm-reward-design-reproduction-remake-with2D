import urllib.request
import json
import os
from config import GEMINI_API_KEY

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
req = urllib.request.Request(url)
with urllib.request.urlopen(req) as response:
    data = json.loads(response.read().decode("utf-8"))
    for model in data.get("models", []):
        if "generateContent" in model.get("supportedGenerationMethods", []):
            print(model["name"])
