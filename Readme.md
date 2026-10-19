# 🛡️ AI-Powered UPI Fraud Detection & Investigation System

A fraud detection project that detects suspicious UPI transactions using statistical and machine-learning methods, then uses an LLM (Google Gemini or a local Ollama model) to explain *why* a transaction was flagged and what an analyst should check next.

> **Note:** The system flags suspicious transactions. It does **not** confirm fraud. All AI output is guidance for a human investigator.

---

## Features

- **Three-signal risk scoring**: IQR outlier detection, Isolation Forest anomaly detection and a time-series spike check, combined into a risk score and level.
- **Interactive Streamlit app**: enter a transaction, get an instant risk result with the model's anomaly score.
- **AI investigation reports**: a FastAPI backend sends non-identifying evidence to Gemini or a local Ollama model and returns a risk summary, evidence, recommended checks and limitations.
- **Privacy-minded prompts**: only a whitelist of non-identifying fields (no VPAs, IDs or names) is sent to the AI.

---

## Project structure

```
├── Backend/
│   ├── main.py            # FastAPI app (/investigate endpoint)
│   ├── ai_assistant.py    # Gemini / Ollama integration and prompt builder
│   └─ test_gemini.py     # Quick Gemini connection test
|    └─ .env                   # API keys (never commit this)
|    
├── Dataset/
│   ├── upi_transaction.csv             # Raw data (10,000 transactions)
│   └── upi_transactions_analyzed.csv   # Data with risk scores added
├── Deployment/
│   └── app.py             # Streamlit app
├── Model/
│   └── isolation_forest.pkl            # Trained Isolation Forest
├── Notebook/
│   └── upi_transaction_file.ipynb      # Analysis and model training
├── Requirement.txt
└── venv
├── .editorconfig/
|── .gitignore         
|---Readme.md
|─ Requirement.txt


```

---

## How it works

```
Raw data → Feature engineering → 3 detectors combined → Saved model + scored CSV
                                                              ↓
              AI investigation report ← FastAPI backend ← Streamlit app
```

### 1. Feature engineering (notebook)
- Per-sender average amount, deviation from it, and amount ratio
- User and recipient transaction counts
- Device-change indicator (device differs from the sender's previous transaction)

### 2. Detection methods
| Signal | Method | Flags |
|---|---|---|
| `iqr_outlier` | IQR rule on transaction amount | Unusually large or small amounts |
| `is_anomaly` | Isolation Forest (100 trees, 5% contamination) | Multi-feature anomalies |
| `time_anomaly` | 7-day rolling mean + 2 standard deviations | Abnormal daily transaction spikes |

### 3. Risk score and level
The three binary signals are summed into `risk_score`:

| Score | Risk level |
|---|---|
| 0 | Normal |
| 1 | Review |
| 2 or more | High Risk |

### 4. Streamlit app
The app takes a few inputs (amount, hour, distance from home, new device, SIM swap, etc.), fills the remaining model features with sensible defaults, and computes the risk score from the IQR and Isolation Forest signals. The time-spike signal is always 0 in the app, because it needs a full day of transaction history, so the app's maximum score is 2.

### 5. AI investigation (FastAPI)
`POST /investigate` accepts a scored transaction and an engine (`gemini` or `local`), builds a restricted prompt, and returns:

- Risk summary
- Evidence available
- Recommended investigation steps
- Important limitations

---

## Setup

### Prerequisites
- Python 3.10 or newer
- A Google Gemini API key (from [Google AI Studio](https://aistudio.google.com)), *or* [Ollama](https://ollama.com) for the local engine

### 1. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

If PowerShell blocks the activation script, run this once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 2. Install dependencies
```bash
python -m pip install -r Requirement.txt
```

### 3. Configure environment variables
Create a `.env` file inside the `Backend/` folder:

```
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=your_gemini_model_name
```

Use a valid Gemini model name for your key. **Never commit `.env` to version control.**

### 4. (Optional) Local AI with Ollama
```bash
ollama pull llama3.2
ollama serve
```

---

## Running the project

Both servers must run at the same time, so open two terminals.

**Terminal 1: backend API**
```powershell
cd Backend
..\venv\Scripts\python.exe -m uvicorn main:app --reload
```
The API runs at `http://127.0.0.1:8000`, and the interactive docs are at `http://127.0.0.1:8000/docs`.

**Terminal 2: Streamlit app** (from the project root)
```powershell
.\venv\Scripts\python.exe -m streamlit run Deployment/app.py
```
The app opens at `http://localhost:8501`.

> Using `python -m ...` with the venv's Python avoids "command not recognized" errors.

---

## Usage

1. In the Streamlit app, click **Load suspicious example** (or enter your own values).
2. Click **Check transaction** to see the risk level and scores.
3. Click **Explain with AI** to get an investigation report from the backend.

### Calling the API directly

```json
POST /investigate
{
  "transaction": {
    "amount_inr": 95000,
    "risk_score": 2,
    "risk_level": "High Risk",
    "is_new_device": 1,
    "iqr_outlier": 1,
    "anomaly": 1
  },
  "engine": "gemini"
}
```

The API also accepts the column names from `upi_transactions_analyzed.csv` (`final_risk_score`, `final_risk_level`, `is_anomaly`), so a CSV row can be sent as-is.

### Re-training the model
Open `Notebook/upi_transaction_file.ipynb`, update the hard-coded file paths to match your machine, and run all cells. This regenerates `isolation_forest.pkl` and the analyzed CSV.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named uvicorn` / `streamlit` | Install into the venv: `.\venv\Scripts\python.exe -m pip install -r Requirement.txt` |
| Pylance "import could not be resolved" | Select the venv interpreter: `Ctrl+Shift+P` → Python: Select Interpreter |
| `422` in Swagger | Clear the request body box and paste only the JSON |
| `502` "AI request failed" | Check the API key, the `GEMINI_MODEL` name and that `.env` is in `Backend/`; restart uvicorn after editing `.env` |
| "Cannot reach the backend" in Streamlit | Start the backend in Terminal 1 |
| Model fails to load | Use the same scikit-learn version the model was trained with |

---

## Limitations

- The Isolation Forest is **unsupervised** and there are no confirmed fraud labels, so it finds unusual transactions, not proven fraud.
- The dataset is synthetic, so results do not reflect real-world fraud rates.
- The Streamlit app fills many features with default values, so it is a demonstration and not a production scorer.
- AI-generated reports can be incomplete or wrong, and a human analyst must make every final decision.

---

## Tech stack

Python · pandas · scikit-learn · Isolation Forest · Streamlit · FastAPI · Uvicorn · Google Gemini API · Ollama (llama3.2)

---

## Security

- Keep API keys in `.env` only, and never commit them.
- Rotate any key that has been shared or uploaded by accident.
- `.env`, `venv/` and `__pycache__/` are already listed in `.gitignore`.