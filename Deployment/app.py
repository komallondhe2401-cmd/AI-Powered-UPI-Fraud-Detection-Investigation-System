from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="UPI Fraud Detection", page_icon="🛡️", layout="centered")

# Relative path so it works locally and on Streamlit Cloud
MODEL_PATH = Path(__file__).resolve().parent.parent / "Model" / "isolation_forest.pkl"
DATA_PATH = Path(__file__).resolve().parent.parent / "Dataset" / "upi_transaction.csv"
API_URL = "http://127.0.0.1:8000/investigate"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_iqr_bounds():
    """Same IQR rule as the notebook, computed on the raw amounts."""
    try:
        amounts = pd.read_csv(DATA_PATH, usecols=["amount_inr"])["amount_inr"]
    except Exception:
        return None
    q1, q3 = amounts.quantile(0.25), amounts.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def risk_level_from_score(score):
    # Same mapping as the notebook: 0 Normal, 1 Review, 2+ High Risk
    if score == 0:
        return "Normal"
    if score == 1:
        return "Review"
    return "High Risk"


model = load_model()
FEATURES = list(model.feature_names_in_)

# Defaults for every feature the user does not enter directly.
# Replace these with medians from your training data for better realism.
BASE = {
    "amount_inr": 450, "mcc": 5411, "latitude": 19.07, "longitude": 72.87,
    "distance_from_home_km": 3.2, "device_age_hours": 4000, "is_new_device": 0,
    "is_rooted_or_emulator": 0, "sim_swap_last_7d": 0, "hour_of_day": 14,
    "day_of_month": 12, "is_weekend": 0, "is_night_txn": 0,
    "usual_hour_start": 9, "usual_hour_end": 22, "is_unusual_hour": 0,
    "sender_account_age_days": 900, "hist_avg_amount": 500, "hist_std_amount": 150,
    "hist_avg_daily_txns": 2, "balance_before_txn": 15000,
    "amount_to_balance_ratio": 0.03, "amount_to_hist_avg_ratio": 0.9,
    "amount_zscore": -0.3, "txn_count_last_1h": 0, "txn_count_last_24h": 1,
    "amount_sum_last_24h": 450, "mins_since_prev_txn": 600, "km_from_prev_txn": 1.5,
    "speed_kmph_from_prev": 0.2, "receiver_account_age_days": 700,
    "is_first_time_receiver": 0, "same_receiver_txn_count_24h": 0,
    "same_receiver_txn_count_7d": 2, "receiver_unique_senders_7d": 15,
    "sender_avg_amount": 500, "amount_deviation": -50, "amount_ratio": 0.9,
    "user_transaction_count": 120, "recipient_transaction_count": 300,
    "device_change": 0,
}

NORMAL = dict(
    amount=450.0, hour=14, distance=3.2, new_device=False, rooted=False,
    sim_swap=False, first_receiver=False, txn_1h=0, receiver_age=700,
    balance=15000.0, hist_avg=500.0, hist_std=150.0,
)
SUSPICIOUS = dict(
    amount=95000.0, hour=3, distance=1150.0, new_device=True, rooted=True,
    sim_swap=True, first_receiver=True, txn_1h=8, receiver_age=1,
    balance=100000.0, hist_avg=500.0, hist_std=150.0,
)

for k, v in NORMAL.items():
    st.session_state.setdefault(k, v)


def set_preset(preset):
    st.session_state.update(preset)


def build_payload(v):
    p = BASE.copy()
    avg = max(v["hist_avg"], 1.0)
    std = max(v["hist_std"], 1.0)
    bal = max(v["balance"], 1.0)
    new_dev = int(v["new_device"])
    first = int(v["first_receiver"])
    hour = int(v["hour"])

    p.update(
        amount_inr=v["amount"],
        hour_of_day=hour,
        distance_from_home_km=v["distance"],
        is_new_device=new_dev,
        device_change=new_dev,
        device_age_hours=2 if new_dev else 4000,
        is_rooted_or_emulator=int(v["rooted"]),
        sim_swap_last_7d=int(v["sim_swap"]),
        is_first_time_receiver=first,
        same_receiver_txn_count_7d=0 if first else 2,
        txn_count_last_1h=int(v["txn_1h"]),
        txn_count_last_24h=max(1, int(v["txn_1h"])),
        receiver_account_age_days=v["receiver_age"],
        balance_before_txn=v["balance"],
        hist_avg_amount=v["hist_avg"],
        sender_avg_amount=v["hist_avg"],
        hist_std_amount=v["hist_std"],
        amount_sum_last_24h=v["amount"],
        # derived features
        amount_to_balance_ratio=v["amount"] / bal,
        amount_to_hist_avg_ratio=v["amount"] / avg,
        amount_ratio=v["amount"] / avg,
        amount_zscore=(v["amount"] - avg) / std,
        amount_deviation=v["amount"] - avg,
        is_night_txn=int(hour < 6),
        is_unusual_hour=int(hour < p["usual_hour_start"] or hour > p["usual_hour_end"]),
    )
    return p


st.title("🛡️ UPI Fraud Detection")
st.caption("Isolation Forest anomaly detection on UPI transaction features.")

c1, c2 = st.columns(2)
c1.button("Load normal example", on_click=set_preset, args=(NORMAL,), use_container_width=True)
c2.button("Load suspicious example", on_click=set_preset, args=(SUSPICIOUS,), use_container_width=True)

st.subheader("Transaction")
a, b = st.columns(2)
a.number_input("Amount (₹)", min_value=1.0, step=100.0, key="amount")
b.slider("Hour of day", 0, 23, key="hour")
a.number_input("Distance from home (km)", min_value=0.0, step=1.0, key="distance")
b.number_input("Transactions in last 1 hour", min_value=0, step=1, key="txn_1h")
a.number_input("Receiver account age (days)", min_value=0, step=1, key="receiver_age")
b.number_input("Balance before transaction (₹)", min_value=1.0, step=500.0, key="balance")
a.number_input("Sender's usual average amount (₹)", min_value=1.0, step=50.0, key="hist_avg")
b.number_input("Sender's usual amount std dev (₹)", min_value=1.0, step=10.0, key="hist_std")

st.subheader("Risk signals")
d, e, f, g = st.columns(4)
d.checkbox("New device", key="new_device")
e.checkbox("Rooted / emulator", key="rooted")
f.checkbox("SIM swap (7d)", key="sim_swap")
g.checkbox("First-time receiver", key="first_receiver")

if st.button("Check transaction", type="primary", use_container_width=True):
    values = {k: st.session_state[k] for k in NORMAL}
    payload = build_payload(values)
    X = pd.DataFrame([payload])[FEATURES]  # enforce training column order

    pred = model.predict(X)[0]
    score = float(model.decision_function(X)[0])

    # Combine signals like the notebook: IQR outlier + Isolation Forest.
    # time_anomaly needs daily volume history, so it is 0 for one transaction.
    bounds = load_iqr_bounds()
    iqr_outlier = int(bounds is not None and not (bounds[0] <= values["amount"] <= bounds[1]))
    anomaly = int(pred == -1)
    risk_score = iqr_outlier + anomaly
    risk_level = risk_level_from_score(risk_score)

    # Keep the result in session_state so it survives the rerun that
    # happens when the "Explain with AI" button is clicked.
    st.session_state["result"] = {
        "features": X,
        "anomaly_score": score,
        "transaction": {
            "amount_inr": values["amount"],
            "risk_score": risk_score,
            "risk_level": risk_level,
            "is_suspicious": int(risk_score > 0),
            "is_new_device": payload["is_new_device"],
            "device_change": payload["device_change"],
            "iqr_outlier": iqr_outlier,
            "anomaly": anomaly,
            "time_anomaly": 0,
            "amount_to_balance_ratio": payload["amount_to_balance_ratio"],
            "amount_to_hist_avg_ratio": payload["amount_to_hist_avg_ratio"],
            "amount_zscore": payload["amount_zscore"],
        },
    }
    st.session_state.pop("report", None)  # clear any old AI report

result = st.session_state.get("result")

if result:
    tx = result["transaction"]

    if tx["risk_level"] == "High Risk":
        st.error("🚨 High Risk")
    elif tx["risk_level"] == "Review":
        st.warning("⚠️ Review: one risk signal triggered")
    else:
        st.success("✅ Normal")

    m1, m2, m3 = st.columns(3)
    m1.metric("Risk score (0-2)", tx["risk_score"])
    m2.metric("Anomaly score", f"{result['anomaly_score']:.4f}",
              help="Below 0 is anomalous. Lower means more suspicious.")
    m3.metric("IQR outlier", "Yes" if tx["iqr_outlier"] else "No")

    with st.expander("Features sent to the model"):
        st.dataframe(result["features"].T.rename(columns={0: "value"}))

    st.subheader("AI investigation")
    engine = st.radio("AI engine", ["gemini", "local"], horizontal=True,
                      help="'local' needs Ollama running with llama3.2.")

    if st.button("Explain with AI", use_container_width=True):
        try:
            with st.spinner("Asking the AI investigator..."):
                resp = requests.post(
                    API_URL,
                    json={"transaction": tx, "engine": engine},
                    timeout=200,
                )
            if resp.ok:
                st.session_state["report"] = resp.json()
            else:
                detail = resp.json().get("detail", resp.text)
                st.error(f"API error {resp.status_code}: {detail}")
        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot reach the backend. Start it first:\n\n"
                "`cd Backend` then `python -m uvicorn main:app --reload`"
            )
        except Exception as exc:
            st.error(f"Request failed: {exc}")

    report = st.session_state.get("report")
    if report:
        st.markdown(report["investigation_report"])
        st.caption(report["warning"])