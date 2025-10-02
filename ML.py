import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set matplotlib to non-GUI backend for deployment
plt.ioff()

df=pd.read_csv("spam.csv",encoding="latin-1")

print("\nHead:")
print(df.head())

# ---------- Parse Date & sort ----------
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values(["Location", "Date"]).reset_index(drop=True)

# ---------- Encode targets ----------
print("\nEncoding targets RainTomorrow/RainToday -> 1/0")
df["RainTomorrow"] = df["RainTomorrow"].map({"Yes": 1, "No": 0})
if "RainToday" in df.columns:
    df["RainToday"] = df["RainToday"].map({"Yes": 1, "No": 0})

# ---------- Ensure required columns exist ----------
expected_cols = [
    "MinTemp","MaxTemp","Rainfall","Evaporation","Sunshine",
    "WindGustSpeed","WindSpeed","WindSpeed9am","WindSpeed3pm",
    "Humidity","Humidity9am","Humidity3pm",
    "Pressure","Pressure9am","Pressure3pm",
    "Cloud9am","Cloud3pm","Temp9am","Temp3pm",
    "RainToday"
]
for c in expected_cols:
    if c not in df.columns:
        df[c] = np.nan

# ---------- Basic checks ----------
print("\nUnique Locations:", df["Location"].nunique())
print("Target unique values:", df["RainTomorrow"].dropna().unique())
print("\nMissing values per column:")
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
print("Dropping duplicates...")
df = df.drop_duplicates(keep="first")

print("\nFinal shape:", df.shape)
print(df.info())

# ---------- Pie chart: Class distribution ----------
print("\nTarget value counts")
vc = df["RainTomorrow"].value_counts(dropna=False)
print(vc)

if len(vc) > 0:
    labels = ["No","Yes"] if set(vc.index) >= {0,1} else [str(x) for x in vc.index]
    plt.figure(figsize=(5,5))
    plt.pie(vc, labels=labels, autopct="%0.2f%%", colors=["lightblue","lightcoral"], startangle=90)
    plt.title("RainTomorrow Distribution (No vs Yes)")
    plt.axis("equal")
    plt.show()

# ---------- Histograms ----------
plt.figure(figsize=(15,10))
df[expected_cols].hist(bins=20, figsize=(15,12), layout=(5,4))
plt.suptitle("Histograms of Weather Features", fontsize=16)
plt.tight_layout()
plt.show()

# ---------- Feature engineering ----------
print("\nFeature engineering...")

df["Month"] = df["Date"].dt.month
df["DayOfYear"] = df["Date"].dt.dayofyear

df["Sin_Month"] = np.sin(2 * np.pi * df["Month"] / 12.0)
df["Cos_Month"] = np.cos(2 * np.pi * df["Month"] / 12.0)
df["Sin_DOY"] = np.sin(2 * np.pi * df["DayOfYear"] / 365.0)
df["Cos_DOY"] = np.cos(2 * np.pi * df["DayOfYear"] / 365.0)

df["TempRange"] = df["MaxTemp"] - df["MinTemp"]
df["TempDelta_3pm_9am"] = df["Temp3pm"] - df["Temp9am"]
df["PressureDelta"] = df["Pressure3pm"] - df["Pressure9am"]

def add_rolling(g):
    g = g.copy()
    for col in ["Rainfall","Humidity","Pressure"]:
        if col in g.columns:
            g[f"{col}_prev3_mean"] = g[col].shift(1).rolling(3, min_periods=1).mean()
            g[f"{col}_prev7_mean"] = g[col].shift(1).rolling(7, min_periods=1).mean()
    if "RainToday" in g.columns:
        g["RainToday_prev3_sum"] = g["RainToday"].shift(1).rolling(3, min_periods=1).sum()
    return g

df = df.groupby("Location", group_keys=False).apply(add_rolling)

# ---------- Select features ----------
feature_cols_numeric = [
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
feature_cols_numeric = [c for c in feature_cols_numeric if c in df.columns]
categorical_cols = ["Location"]

# ---------- Drop rows without target ----------
df = df.dropna(subset=["RainTomorrow"]).reset_index(drop=True)
y = df["RainTomorrow"].astype(int).values
X_all = df[feature_cols_numeric + categorical_cols].copy()

# ---------- Correlation heatmap ----------
num_for_corr = df[["RainTomorrow"] + [c for c in feature_cols_numeric if c not in ["Sin_Month","Cos_Month","Sin_DOY","Cos_DOY"]]]
plt.figure(figsize=(10,4))
sns.heatmap(num_for_corr.corr().fillna(0).clip(-1,1), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap (numeric features)")
plt.tight_layout()
plt.show()

# ---------- Pairplot ----------
try:
    sns.pairplot(df[["RainTomorrow","MinTemp","MaxTemp","Rainfall","Humidity","Pressure"]].dropna(),
                 hue="RainTomorrow", palette="husl", diag_kind="kde")
    plt.suptitle("Pairplot of Selected Features vs RainTomorrow", y=1.02)
    plt.show()
except Exception as e:
    print("⚠️ Pairplot skipped due to data size or memory:", e)

# ---------- Preprocessing ----------
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, feature_cols_numeric),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

scaler = StandardScaler(with_mean=False)  # keep sparse compatibility
linear_preproc = Pipeline(steps=[
    ("ct", preprocessor),
    ("scaler", scaler)
])
tree_preproc = Pipeline(steps=[
    ("ct", preprocessor)
])

# ---------- Time-based split ----------
n = len(df)
test_size = max(1, int(0.2*n))
train_idx = np.arange(0, n-test_size)
test_idx = np.arange(n-test_size, n)

X_train_raw = X_all.iloc[train_idx]
X_test_raw  = X_all.iloc[test_idx]
y_train = y[train_idx]
y_test  = y[test_idx]

print(f"\nTrain size: {len(y_train)} | Test size: {len(y_test)}")

print("Fitting preprocessors...")
X_train_linear = linear_preproc.fit_transform(X_train_raw)
X_test_linear  = linear_preproc.transform(X_test_raw)
X_train_tree   = tree_preproc.fit_transform(X_train_raw)
X_test_tree    = tree_preproc.transform(X_test_raw)

def dense(X):
    return X.toarray() if hasattr(X, "toarray") else X

# ---------- Models ----------
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, StackingClassifier, BaggingClassifier
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score

def make_bagging():
    try:
        return BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), n_estimators=200, random_state=42)
    except TypeError:
        return BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=42), n_estimators=200, random_state=42)

models = {
    "GaussianNB":            ("linear", GaussianNB()),
    "LogisticRegression":    ("linear", LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=200)),
    "SVC":                   ("linear", SVC(probability=True, class_weight="balanced")),
    "DecisionTreeClassifier":("tree",   DecisionTreeClassifier(class_weight="balanced", random_state=42)),
    "KNeighborsClassifier":  ("linear", KNeighborsClassifier(n_neighbors=3)),
    "RandomForestClassifier":("tree",   RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=42)),
    "AdaBoostClassifier":    ("tree",   AdaBoostClassifier(n_estimators=300, random_state=42)),
    "BaggingClassifier":     ("tree",   make_bagging()),
    "ExtraTreesClassifier":  ("tree",   ExtraTreesClassifier(n_estimators=400, random_state=42)),
    "GradientBoostingClassifier": ("tree", GradientBoostingClassifier(n_estimators=300, random_state=42))
}

performance_data = []
conf_matrices = []  # store confusion matrices

print("\nTraining models...")
for name, (prep, model) in models.items():
    if prep == "linear":
        Xtr, Xte = dense(X_train_linear), dense(X_test_linear)
    else:
        Xtr, Xte = dense(X_train_tree), dense(X_test_tree)

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(Xte)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(Xte)
    else:
        y_score = y_pred.astype(float)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    try:    roc = roc_auc_score(y_test, y_score)
    except: roc = np.nan
    try:    pr_auc = average_precision_score(y_test, y_score)
    except: pr_auc = np.nan
    cm = confusion_matrix(y_test, y_pred)

    performance_data.append((name, acc, prec, rec, f1, roc, pr_auc, prep, model))
    conf_matrices.append((name, cm))

# ---------- Stacking ----------
stack_estimators = [
    ("lr", LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=200)),
    ("rf", RandomForestClassifier(n_estimators=300, random_state=42))
]
stack_final = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=200)
stack = StackingClassifier(estimators=stack_estimators, final_estimator=stack_final)

Xtr, Xte = dense(X_train_tree), dense(X_test_tree)
stack.fit(Xtr, y_train)
y_pred_stack  = stack.predict(Xte)
y_score_stack = stack.predict_proba(Xte)[:,1]

cm_stack = confusion_matrix(y_test, y_pred_stack)
acc_s  = accuracy_score(y_test, y_pred_stack)
prec_s = precision_score(y_test, y_pred_stack, zero_division=0)
rec_s  = recall_score(y_test, y_pred_stack, zero_division=0)
f1_s   = f1_score(y_test, y_pred_stack, zero_division=0)
roc_s  = roc_auc_score(y_test, y_score_stack) if len(np.unique(y_test)) > 1 else np.nan
pr_s   = average_precision_score(y_test, y_score_stack)

performance_data.append(("StackingClassifier", acc_s, prec_s, rec_s, f1_s, roc_s, pr_s, "tree", stack))
conf_matrices.append(("StackingClassifier", cm_stack))

# ---------- Plot all confusion matrices together ----------
n_models = len(conf_matrices)
cols = 3
rows = int(np.ceil(n_models / cols))

fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
axes = axes.flatten()

for ax, (name, cm) in zip(axes, conf_matrices):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Rain","Rain"], yticklabels=["No Rain","Rain"], ax=ax)
    ax.set_title(name)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")

for ax in axes[len(conf_matrices):]:
    ax.axis("off")

plt.tight_layout()
plt.show()

# ---------- Compare & plot ----------
perf_df = pd.DataFrame(performance_data, columns=["Algorithm","Accuracy","Precision","Recall","F1","ROC_AUC","PR_AUC","Prep","Model"])
perf_df.to_csv("metrics.csv", index=False)

perf_melt = perf_df[["Algorithm","Accuracy","Precision"]].melt(id_vars="Algorithm", var_name="Metric", value_name="Score")
sns.catplot(x="Algorithm", y="Score", hue="Metric", data=perf_melt, kind="bar", height=6)
plt.xticks(rotation=45, ha="right")
plt.ylim(0.0, 1.05)
plt.title("Model Comparison - Accuracy v/s Precision")
plt.tight_layout()
plt.show()

# ---------- Pie chart of model performance (F1-scores) ----------
f1_scores = perf_df.set_index("Algorithm")["F1"]
plt.figure(figsize=(7,7))
plt.pie(f1_scores, labels=f1_scores.index, autopct="%0.2f%%", startangle=90)
plt.title("Model Performance Share (F1-Score)")
plt.axis("equal")
plt.show()

# ---------- Save best model + preprocessor ----------
perf_sorted = perf_df.sort_values(["PR_AUC","F1","Accuracy"], ascending=False).reset_index(drop=True)
best = perf_sorted.iloc[0]
best_model = best["Model"]
best_prep  = linear_preproc if best["Prep"] == "linear" else tree_preproc

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(best_prep, f)
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\n✅ Saved best model to model.pkl and preprocessor to vectorizer.pkl")
print("Top models by PR_AUC/F1/Accuracy:\n", perf_sorted[["Algorithm","PR_AUC","F1","Accuracy"]].head(5))



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
