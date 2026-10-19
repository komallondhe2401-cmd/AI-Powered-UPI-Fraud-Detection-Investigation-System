from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ai_assistant import (
    explain_with_gemini,
    explain_with_local_ai,
)

app = FastAPI(
    title="UPI Fraud Investigation Assistant",
    description="AI explanations for suspicious transactions",
    version="1.0.0",
)


class InvestigationRequest(BaseModel):
    transaction: dict[str, Any]
    engine: Literal["gemini", "local"] = "local"


# Column names in upi_transactions_analyzed.csv differ from the names the
# API uses, so map them here. This lets a CSV row be sent as-is.
ALIASES = {
    "final_risk_score": "risk_score",
    "final_risk_level": "risk_level",
    "is_anomaly": "anomaly",
}


def normalize_transaction(transaction: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(transaction)
    for old_name, new_name in ALIASES.items():
        if new_name not in cleaned and old_name in cleaned:
            cleaned[new_name] = cleaned[old_name]
    return cleaned


@app.get("/")
def home():
    return {
        "message": "UPI Fraud Investigation API is running"
    }


@app.post("/investigate")
def investigate(request: InvestigationRequest):
    transaction = normalize_transaction(request.transaction)

    # This first version explains an existing risk result.
    if "risk_score" not in transaction:
        raise HTTPException(
            status_code=400,
            detail="Transaction must include risk_score."
        )

    if "risk_level" not in transaction:
        raise HTTPException(
            status_code=400,
            detail="Transaction must include risk_level."
        )

    try:
        if request.engine == "gemini":
            explanation = explain_with_gemini(transaction)
        else:
            explanation = explain_with_local_ai(transaction)

    except Exception as exc:
        # Avoid returning API keys or internal details to the caller.
        raise HTTPException(
            status_code=502,
            detail=(
                "AI request failed. Check your API key, "
                "model availability, or local AI server."
            ),
        ) from exc

    return {
        "risk_score": transaction["risk_score"],
        "risk_level": transaction["risk_level"],
        "ai_engine": request.engine,
        "investigation_report": explanation,
        "warning": (
            "AI-generated guidance only. "
            "This is not confirmation of fraud."
        ),
    }