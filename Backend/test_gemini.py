
from pathlib import Path
from dotenv import load_dotenv
from google import genai
import os

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print("Connecting to Gemini...", flush=True)

try:
    
    interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input="Explain UPI fraud detection in two simple sentences."
    )

    print(interaction.output_text)


except Exception as e:
    print("\nGemini request failed:")
    print(repr(e))
