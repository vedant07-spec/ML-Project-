# frontend.py
# Streamlit app to load vectorizer.pkl + model.pkl and predict RainTomorrow.
# Run: streamlit run frontend.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

st.set_page_config(page_title="Rain Prediction", page_icon="🌧️", layout="centered")

st.title("🌧️ Rain Tomorrow — Prediction")
st.caption("Loads the saved best model (model.pkl) and preprocessor (vectorizer.pkl).")

@st.cache_resource
def load_artifacts():
    with open("vectorizer.pkl", "rb") as f:
        preproc = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return preproc, model

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering as in ML.py (must match!)."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Month"] = df["Date"].dt.month
    df["DayOfYear"] = df["Date"].dt.dayofyear
  
    df["Sin_Month"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["Cos_Month"] = np.cos(2 * np.pi * df["Month"] / 12.0)
    df["Sin_DOY"] = np.sin(2 * np.pi * df["DayOfYear"] / 365.0)
    df["Cos_DOY"] = np.cos(2 * np.pi * df["DayOfYear"] / 365.0)

    df["TempRange"] = df["MaxTemp"] - df["MinTemp"]
    df["TempDelta_3pm_9am"] = df["Temp3pm"] - df["Temp9am"]
    df["PressureDelta"] = df["Pressure3pm"] - df["Pressure9am"]

    # rolling features need historical context; set to NaN for single-row predictions
    for col in ["Rainfall","Humidity","Pressure"]:
        df[f"{col}_prev3_mean"] = np.nan
        df[f"{col}_prev7_mean"] = np.nan
    df["RainToday_prev3_sum"] = np.nan

    return df

# expected columns (same as ML.py)
NUMERIC_COLS = [
    "MinTemp","MaxTemp","Rainfall","Evaporation","Sunshine",
    "WindGustSpeed","WindSpeed","WindSpeed9am","WindSpeed3pm",
    "Humidity","Humidity9am","Humidity3pm",
    "Pressure","Pressure9am","Pressure3pm",
    "Cloud9am","Cloud3pm","Temp9am","Temp3pm","RainToday",
    "Month","DayOfYear","Sin_Month","Cos_Month","Sin_DOY","Cos_DOY",
    "TempRange","TempDelta_3pm_9am","PressureDelta",
    "Rainfall_prev3_mean","Rainfall_prev7_mean",
    "Humidity_prev3_mean","Humidity_prev7_mean",
    "Pressure_prev3_mean","Pressure_prev7_mean",
    "RainToday_prev3_sum"
]
CATEGORICAL_COLS = ["Location"]
ALL_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# ---- Sidebar: Load model ----
with st.sidebar:
    st.header("Artifacts")
    try:
        preproc, model = load_artifacts()
        st.success("Loaded model.pkl & vectorizer.pkl")
    except Exception as e:
        st.error("Artifacts not found. Run ML.py first.")
        st.stop()

# ---- Input form ----
st.subheader("Enter today's observations")
with st.form("input-form"):
    col1, col2 = st.columns(2)

    with col1:
        date_val = st.date_input("Date", value=datetime.today())
        location = st.text_input("Location", value="Mumbai")
        min_temp = st.number_input("MinTemp (°C)", value=24.0)
        max_temp = st.number_input("MaxTemp (°C)", value=32.0)
        rainfall = st.number_input("Rainfall (mm)", value=2.0, min_value=0.0)
        evap = st.number_input("Evaporation (mm)", value=5.0, min_value=0.0)
        sunshine = st.number_input("Sunshine (hours)", value=7.0, min_value=0.0)
        wind_gust_speed = st.number_input("WindGustSpeed (km/h)", value=35.0, min_value=0.0)

    with col2:
        windspeed = st.number_input("WindSpeed (km/h)", value=15.0, min_value=0.0)
        wind9 = st.number_input("WindSpeed9am (km/h)", value=10.0, min_value=0.0)
        wind3 = st.number_input("WindSpeed3pm (km/h)", value=18.0, min_value=0.0)
        hum9 = st.number_input("Humidity9am (%)", value=70.0, min_value=0.0, max_value=100.0)
        hum3 = st.number_input("Humidity3pm (%)", value=60.0, min_value=0.0, max_value=100.0)
        pres9 = st.number_input("Pressure9am (hPa)", value=1010.0)
        pres3 = st.number_input("Pressure3pm (hPa)", value=1008.0)
        cloud9 = st.number_input("Cloud9am (oktas)", value=4.0, min_value=0.0, max_value=8.0)
        cloud3 = st.number_input("Cloud3pm (oktas)", value=5.0, min_value=0.0, max_value=8.0)
        temp9 = st.number_input("Temp9am (°C)", value=28.0)
        temp3 = st.number_input("Temp3pm (°C)", value=31.0)
        rain_today = st.selectbox("RainToday", ["No", "Yes"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # prepare a single-row DataFrame with base fields
    base = pd.DataFrame([{
        "Date": pd.to_datetime(date_val),
        "Location": location,
        "MinTemp": min_temp,
        "MaxTemp": max_temp,
        "Rainfall": rainfall,
        "Evaporation": evap,
        "Sunshine": sunshine,
        "WindGustSpeed": wind_gust_speed,
        "WindSpeed": windspeed,
        "WindSpeed9am": wind9,
        "WindSpeed3pm": wind3,
        "Humidity": np.nan,          # (not collected; will be imputed)
        "Humidity9am": hum9,
        "Humidity3pm": hum3,
        "Pressure": np.nan,          # (not collected; will be imputed)
        "Pressure9am": pres9,
        "Pressure3pm": pres3,
        "Cloud9am": cloud9,
        "Cloud3pm": cloud3,
        "Temp9am": temp9,
        "Temp3pm": temp3,
        "RainToday": 1 if rain_today == "Yes" else 0
    }])

    fe = engineer_features(base)

    # ensure all expected columns exist (preprocessor will impute)
    for c in ALL_COLS:
        if c not in fe.columns and c != "Location":
            fe[c] = np.nan

    # keep only the columns the model expects
    X = fe[ALL_COLS]

    try:
        X_vec = preproc.transform(X)
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X_vec)[:, 1][0])
        elif hasattr(model, "decision_function"):
            # map decision function to [0,1] via logistic
            score = float(model.decision_function(X_vec)[0])
            proba = 1 / (1 + np.exp(-score))
        else:
            proba = float(model.predict(X_vec)[0])
        pred = 1 if proba >= 0.5 else 0
        st.success(f"Prediction: {'🌧️ Yes, it may rain tomorrow' if pred==1 else '☀️ No, likely no rain tomorrow'}")
        st.metric("Estimated probability of rain", f"{proba*100:.2f}%")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption("Tip: If you want batch predictions, add a file uploader and apply the same engineer_features() before transform.")
