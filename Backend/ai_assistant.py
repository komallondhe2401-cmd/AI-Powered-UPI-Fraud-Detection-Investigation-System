
import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Load environment variables from backend/.env
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

gemini_client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"


def build_prompt(transaction):
    # Only send relevant, non-identifying fields to the AI.
    allowed_fields = [
        "amount_inr",
        "risk_score",
        "risk_level",
        "is_suspicious",
        "is_new_device",
        "device_change",
        "iqr_outlier",
        "anomaly",
        "time_anomaly",
        "anomaly_pattern",
        "amount_to_balance_ratio",
        "amount_to_hist_avg_ratio",
        "amount_zscore",
    ]

    safe_data = {
        key: transaction[key]
        for key in allowed_fields
        if key in transaction
    }

    return f"""
You are a UPI fraud investigation assistant.

Analyze the supplied transaction information.

Rules:
1. Explain the supplied risk score and risk level.
2. Use only evidence present in the supplied fields.
3. Do not invent missing information.
4. Suggest practical checks an analyst can perform.
5. A suspicious transaction is not automatically confirmed fraud.
6. Do not make the final fraud decision yourself.

Transaction information:
{json.dumps(safe_data, default=str)}

Respond with:
- Risk summary
- Evidence available
- Recommended investigation steps
- Important limitations
"""


def explain_with_gemini(transaction):
    if gemini_client is None:
        raise RuntimeError(
            "Gemini API key is missing. Configure backend/.env."
        )

    prompt = build_prompt(transaction)

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    return response.text or "Gemini returned no explanation."


def explain_with_local_ai(transaction):
    prompt = build_prompt(transaction)

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
        },
        timeout=180,
    )

    response.raise_for_status()

    return response.json()["message"]["content"]
