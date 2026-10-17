import time
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             calinski_harabasz_score, davies_bouldin_score, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import db
from dataset import (AREAS, CITIES, CRIMES, DAY_TYPES, TIMES, create_dataset)

st.set_page_config(page_title="Crime Rate Prediction System", layout="wide")

# ----------------------------------------------------------------------------
# Look and feel
# ----------------------------------------------------------------------------
INK, PAPER, CARD, RULE = "#25282b", "#f6f3ee", "#fffdf9", "#d9d2c5"
RED, TEAL, OCHRE, GREY = "#8c2f39", "#2f5d62", "#c08a3e", "#8a8f94"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600&family=IBM+Plex+Sans:wght@400;500&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; color: {INK}; }}
.stApp {{ background: {PAPER}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
h1, h2, h3, h4 {{ font-family: 'Source Serif 4', Georgia, serif !important; color: {INK}; letter-spacing: -0.01em; }}

section[data-testid="stSidebar"] {{ background: #1e2a32; border-right: none; }}
section[data-testid="stSidebar"] * {{ color: #d8dde0; }}
section[data-testid="stSidebar"] hr {{ border-color: #35444e; }}
section[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child {{ display: none; }}
section[data-testid="stSidebar"] label[data-baseweb="radio"] {{
    padding: 7px 12px; margin: 1px 0; border-left: 3px solid transparent; width: 100%; }}
section[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {{ background: #26363f; }}
section[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {{
    background: #2b3d47; border-left: 3px solid {OCHRE}; }}
.brand {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 1.15rem; color: #f1ece2;
          padding: 4px 4px 2px 4px; line-height: 1.25; }}
.brand small {{ display:block; font-family: 'IBM Plex Sans', sans-serif; font-size: .72rem;
          letter-spacing: .12em; text-transform: uppercase; color: #8fa0aa; margin-top: 6px; }}

.masthead {{ border-bottom: 2px solid {INK}; padding-bottom: 10px; margin-bottom: 26px; }}
.masthead .kicker {{ font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: {RED}; }}
.masthead .title {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 2rem; font-weight: 600; margin: 2px 0 0 0; }}
.masthead .sub {{ color: #5d6368; margin-top: 4px; font-size: .95rem; }}

div[data-testid="stMetric"] {{ background: {CARD}; border: 1px solid {RULE}; border-left: 4px solid {TEAL};
    padding: 14px 16px; border-radius: 2px; }}
div[data-testid="stMetric"] label p {{ font-size: .72rem !important; letter-spacing: .08em; text-transform: uppercase; color: #6b6f73; }}
div[data-testid="stMetricValue"] {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 1.7rem; }}

.verdict {{ background: {CARD}; border: 1px solid {RULE}; border-left: 5px solid {TEAL};
    padding: 14px 18px; margin: 12px 0; border-radius: 2px; }}
.verdict.high {{ border-left-color: {RED}; }}
.verdict.caution {{ border-left-color: {OCHRE}; }}
.verdict .lab {{ font-size: .7rem; letter-spacing: .12em; text-transform: uppercase; color: #6b6f73; }}
.verdict .val {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 1.5rem; margin-top: 2px; }}
.verdict .txt {{ color: #4a4f53; font-size: .92rem; margin-top: 2px; }}
.note {{ background: #efe9dd; border-left: 3px solid {GREY}; padding: 10px 14px; margin: 10px 0;
         font-size: .92rem; color: #3e4347; border-radius: 2px; }}
.note.warn {{ border-left-color: {OCHRE}; background: #f6ecd9; }}

div[data-testid="stDataFrame"] {{ border: 1px solid {RULE}; border-radius: 2px; }}
.stButton > button, .stDownloadButton > button {{ background: {INK}; color: #f6f3ee; border: none;
    border-radius: 2px; padding: 8px 18px; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ background: {RED}; color: #fff; }}
hr {{ border-color: {RULE}; }}
section[data-testid="stSidebar"] .stButton > button {{ background: transparent; border: 1px solid #55697a; color: #d8dde0; width: 100%; }}
section[data-testid="stSidebar"] .stButton > button:hover {{ background: #2b3d47; border-color: {OCHRE}; color: #fff; }}
.track {{ display: flex; margin: 14px 0 6px 0; }}
.track .step {{ flex: 1; text-align: center; position: relative; font-size: .76rem; color: #7a7f84; }}
.track .step::before {{ content: ""; position: absolute; top: 13px; left: -50%; width: 100%; height: 2px; background: {RULE}; z-index: 0; }}
.track .step:first-child::before {{ display: none; }}
.track .dot {{ width: 28px; height: 28px; line-height: 26px; margin: 0 auto 6px auto; border: 2px solid {RULE};
    background: {PAPER}; border-radius: 2px; position: relative; z-index: 1; font-size: .78rem; }}
.track .step.done .dot {{ background: {TEAL}; border-color: {TEAL}; color: #fff; }}
.track .step.done::before, .track .step.now::before {{ background: {TEAL}; }}
.track .step.now .dot {{ background: {RED}; border-color: {RED}; color: #fff; }}
.track .step.now {{ color: {INK}; font-weight: 500; }}
.chip {{ background: #26363f; border-left: 3px solid {OCHRE}; padding: 8px 12px; margin: 4px 0 10px 0; font-size: .85rem; line-height: 1.35; }}
.chip b {{ color: #f1ece2; font-weight: 500; }}
.chip span {{ color: #8fa0aa; font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; }}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": "#9a958a", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e0dacd", "grid.linewidth": .7, "axes.axisbelow": True,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.prop_cycle": plt.cycler(color=[RED, TEAL, OCHRE, GREY, "#5b4a6b", "#4f7a3d"]),
    "legend.frameon": False,
})


def masthead(kicker, title, sub=""):
    st.markdown(f"<div class='masthead'><div class='kicker'>{kicker}</div>"
                f"<div class='title'>{title}</div><div class='sub'>{sub}</div></div>",
                unsafe_allow_html=True)


def verdict(label, value, text="", tone=""):
    st.markdown(f"<div class='verdict {tone}'><div class='lab'>{label}</div>"
                f"<div class='val'>{value}</div><div class='txt'>{text}</div></div>",
                unsafe_allow_html=True)


def note(text, warn=False):
    st.markdown(f"<div class='note {'warn' if warn else ''}'>{text}</div>", unsafe_allow_html=True)


def section(text):
    st.markdown(f"#### {text}")


def tracker(status):
    """Horizontal progress bar for a report: done steps are ticked, the current step is highlighted."""
    idx = db.STATUSES.index(status) if status in db.STATUSES else 0
    steps = ""
    for i, name in enumerate(db.STATUSES):
        cls = "done" if i < idx else ("now" if i == idx else "")
        mark = "&#10003;" if i < idx else str(i + 1)
        steps += f"<div class='step {cls}'><div class='dot'>{mark}</div>{name}</div>"
    st.markdown(f"<div class='track'>{steps}</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data + models
# ----------------------------------------------------------------------------
CLS_FEATURES = ["Population_Density", "Unemployment_Rate", "Literacy_Rate",
                "Police_Per_100k", "CCTV_Coverage", "Street_Lighting", "Prior_Incidents"]
REG_FEATURES = ["Population_Density", "Unemployment_Rate", "Literacy_Rate",
                "Police_Per_100k", "CCTV_Coverage", "Street_Lighting", "Avg_Income_K"]
CLUSTER_FEATURES = ["Population_Density", "Unemployment_Rate", "Police_Per_100k",
                    "CCTV_Coverage", "Avg_Income_K", "Crime_Rate"]
ARM_COLUMNS = ["Area_Type", "Crime_Type", "Time_of_Day", "Day_Type", "City"]
ARM_SUPPORT, ARM_CONF = .05, .50
CM_INDEX = ["Actual Low Risk", "Actual High Risk"]
CM_COLUMNS = ["Predicted Low Risk", "Predicted High Risk"]

# (label, min, max, default, step) for the prediction forms
RANGES = {
    "Population_Density": ("Population density (per km²)", 500, 45000, 12000, 500),
    "Unemployment_Rate": ("Unemployment rate (%)", 2.0, 25.0, 8.0, 0.5),
    "Literacy_Rate": ("Literacy rate (%)", 45.0, 99.0, 85.0, 0.5),
    "Police_Per_100k": ("Police per 100k people", 30, 300, 130, 5),
    "CCTV_Coverage": ("CCTV coverage (%)", 0.0, 95.0, 40.0, 1.0),
    "Street_Lighting": ("Street lighting (%)", 25.0, 100.0, 75.0, 1.0),
    "Prior_Incidents": ("Incidents in past 12 months", 0, 60, 8, 1),
    "Avg_Income_K": ("Average income (₹ thousand / month)", 8.0, 150.0, 35.0, 1.0),
}


@st.cache_data
def load_data():
    return create_dataset(10000)


def preprocess(df):
    data = df.copy()
    for col in data.select_dtypes(include=np.number).columns:
        data[col] = data[col].fillna(data[col].median())
    for col in data.select_dtypes(exclude=np.number).columns:
        data[col] = data[col].fillna(data[col].mode()[0])
    return data.drop_duplicates()


def mine_rules(freq, n_rows):
    """Works with both old and new mlxtend versions of association_rules."""
    try:
        return association_rules(freq, metric="confidence", min_threshold=ARM_CONF)
    except TypeError:
        return association_rules(freq, num_itemsets=n_rows, metric="confidence", min_threshold=ARM_CONF)


def run_apriori(data):
    sample = data[ARM_COLUMNS].astype(str)
    encoded = pd.get_dummies(sample).astype(bool)
    freq = apriori(encoded, min_support=ARM_SUPPORT, use_colnames=True)
    return freq, len(encoded)


def split_classification(data):
    X = data[CLS_FEATURES]
    y = (data.Risk_Level == "High Risk").astype(int)
    return X, y, train_test_split(X, y, test_size=.2, random_state=42, stratify=y)


def cm_frame(cm):
    return pd.DataFrame(cm, index=CM_INDEX, columns=CM_COLUMNS)


def input_form(features, key):
    """Slider grid for the prediction panels. Returns values in feature order."""
    vals = []
    cols = st.columns(3)
    for i, f in enumerate(features):
        label, lo, hi, default, step = RANGES[f]
        with cols[i % 3]:
            vals.append(st.slider(label, lo, hi, default, step, key=f"{key}_{f}"))
    return vals


@st.cache_data
def compute_comparison(data):
    """Runs every algorithm with fixed settings so results match the individual modules."""
    res = {}

    # ---- classification: Decision Tree vs Naive Bayes
    X, y, (Xtr, Xte, ytr, yte) = split_classification(data)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {"Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
              "Naive Bayes": GaussianNB()}
    clf, cms, roc = {}, {}, {}
    for name, m in models.items():
        t0 = time.perf_counter(); m.fit(Xtr, ytr); fit_ms = (time.perf_counter() - t0) * 1000
        pred = m.predict(Xte); proba = m.predict_proba(Xte)[:, 1]
        cvs = cross_val_score(m, X, y, cv=cv, scoring="accuracy")
        fpr, tpr, _ = roc_curve(yte, proba)
        clf[name] = {"Accuracy": accuracy_score(yte, pred),
                     "Precision": precision_score(yte, pred, zero_division=0),
                     "Recall": recall_score(yte, pred, zero_division=0),
                     "F1-Score": f1_score(yte, pred, zero_division=0),
                     "ROC-AUC": roc_auc_score(yte, proba),
                     "CV Accuracy (5-fold)": cvs.mean(),
                     "CV Std Dev": cvs.std(),
                     "Training Time (ms)": fit_ms}
        cms[name] = confusion_matrix(yte, pred)
        roc[name] = (fpr, tpr)
    res["clf"], res["cms"], res["roc"] = clf, cms, roc
    res["baseline"] = max(float(yte.mean()), 1 - float(yte.mean()))
    res["test_n"] = int(len(yte))

    # ---- regression: Linear vs Decision Tree vs Random Forest
    Xr, yr = data[REG_FEATURES], data.Crime_Rate
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=.2, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    reg_models = {"Linear Regression": LinearRegression(),
                  "Decision Tree Regressor": DecisionTreeRegressor(max_depth=8, random_state=42),
                  "Random Forest Regressor": RandomForestRegressor(n_estimators=60, max_depth=12,
                                                                   random_state=42, n_jobs=-1)}
    reg, reg_pred = {}, {}
    for name, m in reg_models.items():
        t0 = time.perf_counter(); m.fit(Xtr, ytr); fit_ms = (time.perf_counter() - t0) * 1000
        p = m.predict(Xte)
        cvs = cross_val_score(m, Xr, yr, cv=kf, scoring="r2")
        reg[name] = {"R2": r2_score(yte, p), "MAE": mean_absolute_error(yte, p),
                     "RMSE": float(np.sqrt(mean_squared_error(yte, p))),
                     "CV R2 (5-fold)": cvs.mean(), "CV Std Dev": cvs.std(),
                     "Training Time (ms)": fit_ms}
        reg_pred[name] = p
    res["reg"], res["reg_pred"] = reg, reg_pred
    res["reg_actual"], res["reg_mean"] = yte.values, float(yr.mean())

    # ---- association rules (Apriori)
    freq, n_rows = run_apriori(data)
    ar = {"itemsets": int(len(freq)), "rules": 0, "max_lift": None,
          "avg_conf": None, "avg_lift": None, "top_rule": "-"}
    if len(freq):
        rules = mine_rules(freq, n_rows)
        if len(rules):
            rules = rules.sort_values("lift", ascending=False)
            top = rules.iloc[0]
            ar.update({"rules": int(len(rules)), "max_lift": float(rules["lift"].max()),
                       "avg_conf": float(rules["confidence"].mean()),
                       "avg_lift": float(rules["lift"].mean()),
                       "top_rule": ", ".join(sorted(top["antecedents"])) + "  →  "
                                   + ", ".join(sorted(top["consequents"]))})
    res["ar"] = ar

    # ---- clustering: K-Means vs Gaussian Mixture vs Agglomerative (same 3,000-row sample, k = 3)
    Xc = StandardScaler().fit_transform(data[CLUSTER_FEATURES])
    idx = np.random.RandomState(42).choice(len(Xc), min(3000, len(Xc)), replace=False)
    Xs = Xc[idx]
    high = (data.Risk_Level == "High Risk").astype(int).values[idx]
    algos = {"K-Means": KMeans(n_clusters=3, random_state=42, n_init=10),
             "Gaussian Mixture": GaussianMixture(n_components=3, random_state=42),
             "Agglomerative (Ward)": AgglomerativeClustering(n_clusters=3)}
    clu, labels_by = {}, {}
    for name, a in algos.items():
        t0 = time.perf_counter(); labels = a.fit_predict(Xs); fit_ms = (time.perf_counter() - t0) * 1000
        sizes = np.bincount(labels, minlength=3)
        rate = [float(high[labels == c].mean()) if sizes[c] else 0.0 for c in range(3)]
        clu[name] = {"Silhouette": float(silhouette_score(Xs, labels)),
                     "Davies-Bouldin": float(davies_bouldin_score(Xs, labels)),
                     "Calinski-Harabasz": float(calinski_harabasz_score(Xs, labels)),
                     "Size Balance": float(sizes.min() / sizes.max()) if sizes.max() else 0.0,
                     "Training Time (ms)": fit_ms,
                     "sizes": [int(v) for v in sizes], "high_rate": rate}
        labels_by[name] = labels
    res["clu"], res["clu_labels"] = clu, labels_by
    res["clu_xy"] = PCA(n_components=2, random_state=42).fit_transform(Xs)
    res["clu_n"] = int(len(Xs))
    return res


def best_of(vals, higher=True, digits=4):
    """Name of the best model for one metric, or 'Tie' if the top two are equal after rounding."""
    ordered = sorted(vals, key=lambda n: round(vals[n], digits), reverse=higher)
    if len(ordered) > 1 and round(vals[ordered[0]], digits) == round(vals[ordered[1]], digits):
        return "Tie"
    return ordered[0]


def mean_rank(table, spec):
    """Average rank of each model over the chosen metrics (1 = best). spec = {metric: higher_is_better}."""
    ranks = {m: [] for m in table}
    for metric, higher in spec.items():
        for pos, m in enumerate(sorted(table, key=lambda n: table[n][metric], reverse=higher), 1):
            ranks[m].append(pos)
    return {m: float(np.mean(v)) for m, v in ranks.items()}


def comparison_table(results, rows):
    """rows = [(metric, higher_is_better, format)]. Returns (dataframe, wins per model)."""
    names = list(results); wins = {n: 0 for n in names}; out = []
    for metric, higher, fmt in rows:
        vals = {n: results[n][metric] for n in names}
        w = best_of(vals, higher, digits=1 if "Time" in metric else 4)
        if w in wins and "Time" not in metric:
            wins[w] += 1
        line = {"Metric": metric + ("" if higher else " (lower is better)")}
        line.update({n: fmt.format(vals[n]) for n in names}); line["Better"] = w
        out.append(line)
    return pd.DataFrame(out), wins



ALL_NUM = ["Population_Density", "Unemployment_Rate", "Literacy_Rate", "Police_Per_100k",
           "CCTV_Coverage", "Street_Lighting", "Avg_Income_K", "Prior_Incidents"]
ADMIN_MENU = ["Dashboard", "Dataset", "Preprocessing", "Classification", "Naive Bayes",
              "Regression", "Association Rules", "Clustering", "Comparison",
              "Registered Crimes", "Users", "Login History"]
USER_MENU = ["Dashboard", "Dataset", "Risk Check", "Register Crime", "My Reports"]


@st.cache_resource
def labeler(base):
    """Models trained on the original dataset only; they label newly registered crimes."""
    data = preprocess(base)
    y = (data.Risk_Level == "High Risk").astype(int)
    clf = DecisionTreeClassifier(max_depth=6, random_state=42).fit(data[CLS_FEATURES], y)
    reg = LinearRegression().fit(data[REG_FEATURES], data.Crime_Rate)
    return clf, reg


def live_dataset():
    """Original records plus every crime registered by users (re-read on each run)."""
    base = load_data().copy()
    base["Source"] = "Base dataset"
    reports = db.get_reports()
    if len(reports):
        extra = reports[db.DATA_COLUMNS].copy()
        extra["Source"] = "User report"
        base = pd.concat([base, extra], ignore_index=True)
    return base


# ----------------------------------------------------------------------------
# Authentication
# ----------------------------------------------------------------------------
db.init_db()


def sign_in(username, password, role):
    user = db.authenticate(username, password, role)
    if user:
        db.log_event(user["username"], role, "Login")
        st.session_state["auth"] = user
        st.rerun()
    db.log_event(username or "(blank)", role, "Failed Login")
    note("Incorrect username or password for this account type.", warn=True)


def login_screen():
    st.markdown("<style>section[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl']"
                "{display:none !important}</style>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:
        masthead("Restricted access", "Crime Rate Prediction System",
                 "Administrators and registered users sign in separately")
        tab_admin, tab_user = st.tabs(["Administrator", "End user"])
        with tab_admin:
            with st.form("admin_login"):
                u = st.text_input("Administrator username")
                p = st.text_input("Password", type="password")
                go = st.form_submit_button("Sign in as administrator")
            if go:
                sign_in(u, p, "admin")
        with tab_user:
            t_in, t_new = st.tabs(["Sign in", "Create account"])
            with t_in:
                with st.form("user_login"):
                    u = st.text_input("Username")
                    p = st.text_input("Password", type="password")
                    go = st.form_submit_button("Sign in")
                if go:
                    sign_in(u, p, "user")
            with t_new:
                with st.form("user_signup"):
                    name = st.text_input("Full name")
                    u = st.text_input("Choose a username")
                    p1 = st.text_input("Choose a password", type="password")
                    p2 = st.text_input("Confirm password", type="password")
                    go = st.form_submit_button("Create account")
                if go:
                    if p1 != p2:
                        note("The two passwords do not match.", warn=True)
                    else:
                        ok, msg = db.create_user(u, p1, "user", name)
                        if ok:
                            db.log_event(u, "user", "Registered")
                            note("Account created. Use the Sign in tab to continue.")
                        else:
                            note(msg, warn=True)


if "auth" not in st.session_state:
    login_screen()
    st.stop()

auth = st.session_state["auth"]
is_admin = auth["role"] == "admin"
df = live_dataset()
user_records = int((df.Source == "User report").sum())

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.markdown("<div class='brand'>Crime Rate Prediction<br>&amp; Data Mining System"
                    "<small>Analysis workbench</small></div>", unsafe_allow_html=True)
st.sidebar.markdown(f"<div class='chip'><span>{'Administrator' if is_admin else 'End user'}</span><br>"
                    f"<b>{auth['name']}</b></div>", unsafe_allow_html=True)
menu = st.sidebar.radio("Module", ADMIN_MENU if is_admin else USER_MENU, label_visibility="collapsed")
st.sidebar.markdown("---")
if st.sidebar.button("Sign out"):
    db.log_event(auth["username"], auth["role"], "Logout")
    del st.session_state["auth"]
    st.rerun()
st.sidebar.caption("Python · Streamlit · Pandas · Scikit-learn · mlxtend")

# ----------------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------------
if menu == "Dashboard":
    masthead("Overview", "Crime Rate Prediction & Data Mining System",
             "Ten Indian cities, 2019 to 2025, one record per area and month")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records", f"{len(df):,}")
    c2.metric("High-risk areas", f"{int((df.Risk_Level == 'High Risk').sum()):,}")
    c3.metric("Low-risk areas", f"{int((df.Risk_Level == 'Low Risk').sum()):,}")
    c4.metric("Avg crime rate", f"{df.Crime_Rate.mean():.1f} / 100k")
    st.write("")
    note(f"{user_records:,} of these records were registered by users through the Register Crime form; "
         "the rest come from the original generated dataset. Every chart and model on this site reads the combined data.")

    a, b = st.columns(2)
    with a:
        section("Records by crime type")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        counts = df.Crime_Type.value_counts().sort_values()
        ax.barh(counts.index, counts.values, color=RED)
        ax.set_xlabel("Records"); ax.grid(axis="y", visible=False)
        st.pyplot(fig)
    with b:
        section("Average crime rate by city")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        city_rate = df.groupby("City").Crime_Rate.mean().sort_values()
        ax.barh(city_rate.index, city_rate.values, color=TEAL)
        ax.set_xlabel("Crime rate per 100k"); ax.grid(axis="y", visible=False)
        st.pyplot(fig)

    section("Yearly trend in average crime rate")
    fig, ax = plt.subplots(figsize=(12, 2.8))
    yr = df.groupby("Year").Crime_Rate.mean()
    ax.plot(yr.index, yr.values, marker="o", color=INK, linewidth=1.6)
    ax.set_ylabel("Avg rate per 100k")
    st.pyplot(fig)

elif menu == "Dataset":
    masthead("Data", "Crime Records Dataset", f"{len(df):,} records: 10,000 generated plus {user_records:,} registered by users")
    f1, f2 = st.columns(2)
    city_pick = f1.multiselect("Filter by city", sorted(df.City.unique()))
    area_pick = f2.multiselect("Filter by area type", sorted(df.Area_Type.unique()))
    view = df.sort_values("Source", ascending=False, kind="stable") if user_records else df
    if city_pick: view = view[view.City.isin(city_pick)]
    if area_pick: view = view[view.Area_Type.isin(area_pick)]
    st.caption(f"Full dataset: {df.shape[0]:,} rows × {df.shape[1]} columns. Showing {min(len(view), 100)} of {len(view):,} matching rows.")
    st.dataframe(view.head(100), width="stretch", hide_index=True)
    section("Dataset statistics")
    st.dataframe(df.describe(include="all").T, width="stretch")
    if is_admin:
        st.download_button("Download CSV", df.to_csv(index=False).encode(),
                           "crime_data_live.csv", "text/csv")

elif menu == "Preprocessing":
    masthead("Data", "Data Preprocessing", "Cleaning steps applied before any model is trained")
    a, b = st.columns(2)
    with a:
        section("Missing values before")
        st.dataframe(df.isnull().sum().to_frame("Missing"), width="stretch")
    clean = preprocess(df)
    with b:
        section("Missing values after")
        st.dataframe(clean.isnull().sum().to_frame("Missing"), width="stretch")
    c1, c2 = st.columns(2)
    c1.metric("Original shape", f"{df.shape[0]:,} × {df.shape[1]}")
    c2.metric("Processed shape", f"{clean.shape[0]:,} × {clean.shape[1]}")
    note("Missing numeric values were filled with the column median, missing categorical values with the mode, "
         "and duplicate rows were removed.")

elif menu == "Classification":
    masthead("Supervised learning", "Decision Tree Classification",
             "Predicts whether an area-month is High Risk or Low Risk (J48 equivalent)")
    data = preprocess(df)
    X, y, (Xtr, Xte, ytr, yte) = split_classification(data)
    model = DecisionTreeClassifier(max_depth=6, random_state=42).fit(Xtr, ytr)
    pred = model.predict(Xte)
    c1, c2 = st.columns([1, 2])
    c1.metric("Accuracy", f"{accuracy_score(yte, pred) * 100:.2f}%")
    with c2:
        st.dataframe(cm_frame(confusion_matrix(yte, pred)), width="stretch")
    note("DecisionTreeClassifier (max depth 6) is used as the Python equivalent of the J48 decision-tree practical.")

    section("Which indicators matter most")
    fig, ax = plt.subplots(figsize=(8, 3))
    imp = pd.Series(model.feature_importances_, index=CLS_FEATURES).sort_values()
    ax.barh(imp.index, imp.values, color=TEAL); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Importance")
    st.pyplot(fig)

    section("Try a prediction")
    vals = input_form(CLS_FEATURES, "dt")
    row = pd.DataFrame([vals], columns=CLS_FEATURES)
    result = model.predict(row)[0]; prob = model.predict_proba(row)[0, 1]
    verdict("Predicted risk level", "High Risk" if result else "Low Risk",
            f"Model probability of High Risk: {prob * 100:.0f}%", "high" if result else "")

elif menu == "Naive Bayes":
    masthead("Supervised learning", "Naive Bayes Classification",
             "Gaussian Naive Bayes on the same indicators and split as the decision tree")
    data = preprocess(df)
    X, y, (Xtr, Xte, ytr, yte) = split_classification(data)
    model = GaussianNB().fit(Xtr, ytr); pred = model.predict(Xte)
    c1, c2 = st.columns([1, 2])
    c1.metric("Accuracy", f"{accuracy_score(yte, pred) * 100:.2f}%")
    with c2:
        st.dataframe(cm_frame(confusion_matrix(yte, pred)), width="stretch")

    section("Try a prediction")
    vals = input_form(CLS_FEATURES, "nb")
    row = pd.DataFrame([vals], columns=CLS_FEATURES)
    result = model.predict(row)[0]; prob = model.predict_proba(row)[0, 1]
    verdict("Predicted risk level", "High Risk" if result else "Low Risk",
            f"Model probability of High Risk: {prob * 100:.0f}%", "high" if result else "")

elif menu == "Regression":
    masthead("Supervised learning", "Crime Rate Prediction using Linear Regression",
             "Estimates the crime rate per 100,000 people from area indicators")
    data = preprocess(df)
    X, y = data[REG_FEATURES], data.Crime_Rate
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
    model = LinearRegression().fit(Xtr, ytr); pred = model.predict(Xte)
    c1, c2 = st.columns(2)
    c1.metric("R² score", f"{r2_score(yte, pred):.3f}")
    c2.metric("Mean absolute error", f"{mean_absolute_error(yte, pred):.2f} per 100k")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(yte, pred, alpha=.35, s=12, color=RED)
    lim = [0, max(yte.max(), pred.max())]
    ax.plot(lim, lim, color=INK, linewidth=1, linestyle="--")
    ax.set_xlabel("Actual crime rate per 100k"); ax.set_ylabel("Predicted crime rate per 100k")
    st.pyplot(fig)

    section("Predict the crime rate")
    vals = input_form(REG_FEATURES, "reg")
    p = model.predict(pd.DataFrame([vals], columns=REG_FEATURES))[0]
    p = max(0.0, p)
    band = "high" if p >= df.Crime_Rate.quantile(.66) else ("caution" if p >= df.Crime_Rate.quantile(.33) else "")
    verdict("Predicted crime rate", f"{p:.1f} per 100k",
            f"Dataset average is {df.Crime_Rate.mean():.1f} per 100k.", band)

elif menu == "Association Rules":
    masthead("Unsupervised learning", "Association Rule Mining",
             "Which place, time and crime attributes tend to occur together")
    freq, n_rows = run_apriori(preprocess(df))
    st.caption(f"Apriori with minimum support {ARM_SUPPORT} and minimum confidence {ARM_CONF}. "
               "Columns used: " + ", ".join(ARM_COLUMNS) + ".")
    if len(freq):
        rules = mine_rules(freq, n_rows)
        if len(rules):
            rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
            rules["antecedents"] = rules.antecedents.apply(lambda x: ", ".join(sorted(x)))
            rules["consequents"] = rules.consequents.apply(lambda x: ", ".join(sorted(x)))
            c1, c2, c3 = st.columns(3)
            c1.metric("Frequent itemsets", len(freq))
            c2.metric("Rules found", len(rules))
            c3.metric("Max lift", f"{rules.lift.max():.2f}")
            st.dataframe(rules.sort_values("lift", ascending=False).head(30).round(3),
                         width="stretch", hide_index=True)
        else:
            note("No rules found with these thresholds.", warn=True)
    else:
        note("No frequent itemsets found.", warn=True)

elif menu == "Clustering":
    masthead("Unsupervised learning", "Area Clustering using K-Means",
             "Groups area-months with similar social and policing profiles")
    data = preprocess(df)
    X = StandardScaler().fit_transform(data[CLUSTER_FEATURES])
    k = st.slider("Number of clusters", 2, 6, 3)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    data["Cluster"] = model.fit_predict(X)
    st.dataframe(data.groupby("Cluster")[CLUSTER_FEATURES].mean().round(2), width="stretch")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for c in sorted(data.Cluster.unique()):
        sub = data[data.Cluster == c]
        ax.scatter(sub.Population_Density, sub.Crime_Rate, s=8, alpha=.5, label=f"Cluster {c}")
    ax.set_xlabel("Population density (per km²)"); ax.set_ylabel("Crime rate per 100k")
    ax.legend(markerscale=2)
    st.pyplot(fig)

elif menu == "Comparison":
    masthead("Evaluation", "Comparison of Techniques",
             "Classification, regression and clustering are each compared across several algorithms")
    st.write("Every algorithm is run here with the **same settings, features and train/test split** as its own module, so "
             "the numbers match what you saw earlier. Classification, regression and clustering solve different problems, so "
             "each family is compared **inside its own group** and the results are then combined into one final recommendation. "
             "The comparison re-runs automatically when users register new crimes.")
    with st.spinner("Running all algorithms..."):
        R = compute_comparison(preprocess(df))
    clf, reg, ar, clu, base = R["clf"], R["reg"], R["ar"], R["clu"], R["baseline"]
    cnames, rnames, knames = list(clf), list(reg), list(clu)
    dt, nb = cnames

    # ---- scoring for each family
    c_spec = {"Accuracy": True, "F1-Score": True, "ROC-AUC": True, "CV Accuracy (5-fold)": True}
    r_spec = {"R2": True, "MAE": False, "RMSE": False, "CV R2 (5-fold)": True}
    k_spec = {"Silhouette": True, "Davies-Bouldin": False, "Calinski-Harabasz": True}
    for n in knames:
        rates = clu[n]["high_rate"]; clu[n]["Risk Separation (pts)"] = (max(rates) - min(rates)) * 100
    c_rank, r_rank, k_rank = mean_rank(clf, c_spec), mean_rank(reg, r_spec), mean_rank(clu, k_spec)
    c_best, r_best, k_best = min(c_rank, key=c_rank.get), min(r_rank, key=r_rank.get), min(k_rank, key=k_rank.get)

    section("1. Overview of all modules")
    overview = pd.DataFrame({
        "Module": ["Classification", "Regression", "Clustering", "Association Rules"],
        "Algorithms Compared": [" vs ".join(cnames), " vs ".join(rnames), " vs ".join(knames), "Apriori (single method)"],
        "Learning Type": ["Supervised", "Supervised", "Unsupervised", "Unsupervised"],
        "What It Does": ["Predicts High or Low risk", "Predicts crime rate per 100k",
                         "Groups similar areas", "Finds attributes that occur together"],
        "Best Algorithm": [c_best, r_best, k_best, "Apriori"],
        "Key Result": [f"Accuracy {clf[c_best]['Accuracy'] * 100:.2f}%",
                       f"R² {reg[r_best]['R2']:.3f}, MAE {reg[r_best]['MAE']:.2f}",
                       f"Silhouette {clu[k_best]['Silhouette']:.3f}",
                       f"{ar['rules']} rules, max lift {ar['max_lift']:.2f}" if ar["rules"] else "No rules found"]})
    st.dataframe(overview, width="stretch", hide_index=True)

    # ------------------------------------------------------------------ classification
    section("2. Classification: Decision Tree vs Naive Bayes")
    st.caption(f"Test set: {R['test_n']:,} records (20%). Always guessing the most common class would score "
               f"{base * 100:.2f}%, so that is the minimum a useful model must beat.")
    c_rows = [("Accuracy", True, "{:.4f}"), ("Precision", True, "{:.4f}"), ("Recall", True, "{:.4f}"),
              ("F1-Score", True, "{:.4f}"), ("ROC-AUC", True, "{:.4f}"), ("CV Accuracy (5-fold)", True, "{:.4f}"),
              ("CV Std Dev", False, "{:.4f}"), ("Training Time (ms)", False, "{:.1f}")]
    c_table, c_wins = comparison_table(clf, c_rows)
    st.dataframe(c_table, width="stretch", hide_index=True)

    g1, g2 = st.columns(2)
    with g1:
        st.write("**Metric comparison**")
        keys = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        x = np.arange(len(keys)); w = .38
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - w / 2, [clf[dt][k] for k in keys], w, label=dt, color=RED)
        ax.bar(x + w / 2, [clf[nb][k] for k in keys], w, label=nb, color=TEAL)
        ax.axhline(base, color=GREY, linestyle="--", linewidth=1, label=f"Majority baseline ({base:.2f})")
        ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20); ax.set_ylim(0, 1)
        ax.set_ylabel("Score"); ax.legend(fontsize=8)
        st.pyplot(fig)
    with g2:
        st.write("**ROC curves**")
        fig, ax = plt.subplots(figsize=(6, 4))
        for n, col in zip(cnames, [RED, TEAL]):
            fpr, tpr = R["roc"][n]
            ax.plot(fpr, tpr, color=col, label=f"{n} (AUC {clf[n]['ROC-AUC']:.3f})")
        ax.plot([0, 1], [0, 1], color=INK, linestyle="--", linewidth=1, label="Random guess")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(fontsize=8)
        st.pyplot(fig)
    st.write("**Confusion matrices (test set)**")
    cc1, cc2 = st.columns(2)
    for col, n in zip([cc1, cc2], cnames):
        col.write(n); col.dataframe(cm_frame(R["cms"][n]))

    c_other = [n for n in cnames if n != c_best][0]
    c_margin = abs(c_rank[dt] - c_rank[nb])
    verdict("Best classifier", c_best,
            f"Average rank {c_rank[c_best]:.2f} vs {c_rank[c_other]:.2f} for {c_other} (accuracy, F1, ROC-AUC, CV accuracy). "
            f"Metrics won overall: {c_wins[dt]} for {dt}, {c_wins[nb]} for {nb}.")
    reasons = []
    if clf[c_best]["ROC-AUC"] > clf[c_other]["ROC-AUC"]:
        reasons.append(f"**{c_best}** ranks areas by risk better (ROC-AUC {clf[c_best]['ROC-AUC']:.3f} vs {clf[c_other]['ROC-AUC']:.3f}).")
    if clf[c_best]["Accuracy"] > clf[c_other]["Accuracy"]:
        reasons.append(f"**{c_best}** has the higher test accuracy ({clf[c_best]['Accuracy'] * 100:.2f}% vs {clf[c_other]['Accuracy'] * 100:.2f}%).")
    if clf[c_other]["Recall"] > clf[c_best]["Recall"]:
        reasons.append(f"**{c_other}** catches more truly high-risk areas (recall {clf[c_other]['Recall']:.3f} vs {clf[c_best]['Recall']:.3f}), "
                       "which matters if missing a dangerous area is costly.")
    for r in reasons:
        st.markdown("- " + r)
    if clf[c_best]["Accuracy"] - base < .05:
        note(f"Honest note: the best accuracy ({clf[c_best]['Accuracy'] * 100:.2f}%) is only "
             f"{(clf[c_best]['Accuracy'] - base) * 100:.2f} points above the {base * 100:.2f}% majority baseline. The risk label in this "
             "generated dataset contains random noise, so no classifier can predict it very accurately.", warn=True)

    # ------------------------------------------------------------------ regression
    section("3. Regression: Linear Regression vs Decision Tree Regressor vs Random Forest")
    st.caption("Target: crime rate per 100,000 people. Same 80/20 split for all three, plus 5-fold cross-validation. "
               "Lower MAE and RMSE are better; higher R² is better.")
    r_rows = [("R2", True, "{:.4f}"), ("MAE", False, "{:.2f}"), ("RMSE", False, "{:.2f}"),
              ("CV R2 (5-fold)", True, "{:.4f}"), ("CV Std Dev", False, "{:.4f}"), ("Training Time (ms)", False, "{:.1f}")]
    r_table, r_wins = comparison_table(reg, r_rows)
    st.dataframe(r_table, width="stretch", hide_index=True)

    g1, g2 = st.columns(2)
    short = {"Linear Regression": "Linear", "Decision Tree Regressor": "Tree", "Random Forest Regressor": "Forest"}
    cols3 = [RED, TEAL, OCHRE]
    with g1:
        st.write("**R² score (higher is better)**")
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.bar([short[n] for n in rnames], [reg[n]["R2"] for n in rnames], color=cols3)
        ax.set_ylim(0, 1); ax.set_ylabel("R²"); ax.grid(axis="x", visible=False)
        st.pyplot(fig)
    with g2:
        st.write("**Average error in crime rate per 100k (lower is better)**")
        fig, ax = plt.subplots(figsize=(6, 3.4))
        xs = np.arange(len(rnames)); w = .38
        ax.bar(xs - w / 2, [reg[n]["MAE"] for n in rnames], w, label="MAE", color=GREY)
        ax.bar(xs + w / 2, [reg[n]["RMSE"] for n in rnames], w, label="RMSE", color=INK)
        ax.set_xticks(xs); ax.set_xticklabels([short[n] for n in rnames]); ax.legend(); ax.grid(axis="x", visible=False)
        st.pyplot(fig)

    st.write("**Actual vs predicted crime rate (test set)**")
    pcols = st.columns(3)
    lim = [0, float(max(R["reg_actual"].max(), max(p.max() for p in R["reg_pred"].values())))]
    for col, n, c in zip(pcols, rnames, cols3):
        with col:
            fig, ax = plt.subplots(figsize=(4, 3.6))
            ax.scatter(R["reg_actual"], R["reg_pred"][n], s=6, alpha=.35, color=c)
            ax.plot(lim, lim, color=INK, linestyle="--", linewidth=1)
            ax.set_title(n, fontsize=9); ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
            st.pyplot(fig)

    r_other_names = [n for n in rnames if n != r_best]
    verdict("Best regression algorithm", r_best,
            f"Average rank {r_rank[r_best]:.2f} (R², MAE, RMSE, CV R²). It explains about {reg[r_best]['R2'] * 100:.0f}% of the variation "
            f"and is off by about {reg[r_best]['MAE'] / R['reg_mean'] * 100:.1f}% of the average rate ({R['reg_mean']:.1f} per 100k).")
    lin = reg["Linear Regression"]
    if r_best != "Linear Regression":
        gap = reg[r_best]["R2"] - lin["R2"]
        if gap < .02:
            st.write(f"The gain over Linear Regression is only {gap:.3f} in R². Linear Regression is far simpler to explain and "
                     "audit, so it is a reasonable choice if explainability matters more than the last bit of accuracy.")
        else:
            st.write(f"{r_best} beats Linear Regression by {gap:.3f} in R², so the relationship between the indicators and the "
                     "crime rate is not purely linear. The cost is a model that is harder to explain.")
    else:
        st.write("The extra flexibility of the tree-based models does not pay off here: the relationship between the indicators "
                 "and the crime rate is close to linear, so the simplest model is also the best.")

    # ------------------------------------------------------------------ clustering
    section("4. Clustering: K-Means vs Gaussian Mixture vs Agglomerative")
    st.caption(f"All three use k = 3 on the same {R['clu_n']:,}-record sample and the same standardised features. Silhouette and "
               "Calinski-Harabasz: higher is better. Davies-Bouldin: lower is better. Risk Separation is how far apart the clusters "
               "are in their share of High Risk records (the labels were never used for clustering).")
    k_rows = [("Silhouette", True, "{:.4f}"), ("Davies-Bouldin", False, "{:.4f}"), ("Calinski-Harabasz", True, "{:.1f}"),
              ("Size Balance", True, "{:.3f}"), ("Risk Separation (pts)", True, "{:.1f}"), ("Training Time (ms)", False, "{:.1f}")]
    k_table, k_wins = comparison_table(clu, k_rows)
    st.dataframe(k_table, width="stretch", hide_index=True)

    g1, g2 = st.columns(2)
    kshort = {"K-Means": "K-Means", "Gaussian Mixture": "Gaussian", "Agglomerative (Ward)": "Agglom."}
    with g1:
        st.write("**Silhouette score (higher is better)**")
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.bar([kshort[n] for n in knames], [clu[n]["Silhouette"] for n in knames], color=cols3)
        ax.set_ylabel("Silhouette"); ax.grid(axis="x", visible=False)
        st.pyplot(fig)
    with g2:
        st.write("**Davies-Bouldin index (lower is better)**")
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.bar([kshort[n] for n in knames], [clu[n]["Davies-Bouldin"] for n in knames], color=cols3)
        ax.set_ylabel("Davies-Bouldin"); ax.grid(axis="x", visible=False)
        st.pyplot(fig)

    st.write("**Cluster layout (two principal components)**")
    pcols = st.columns(3)
    for col, n in zip(pcols, knames):
        with col:
            fig, ax = plt.subplots(figsize=(4, 3.6))
            for c, cc in zip(range(3), [RED, TEAL, OCHRE]):
                m = R["clu_labels"][n] == c
                ax.scatter(R["clu_xy"][m, 0], R["clu_xy"][m, 1], s=6, alpha=.5, color=cc, label=f"Cluster {c}")
            ax.set_title(n, fontsize=9); ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
            st.pyplot(fig)

    st.write("**Cluster sizes and High Risk share per algorithm**")
    prof = [{"Algorithm": n, "Cluster": c, "Records": clu[n]["sizes"][c],
             "High-Risk Share": f"{clu[n]['high_rate'][c] * 100:.1f}%"} for n in knames for c in range(3)]
    st.dataframe(pd.DataFrame(prof), width="stretch", hide_index=True)

    verdict("Best clustering algorithm", k_best,
            f"Average rank {k_rank[k_best]:.2f} on Silhouette, Davies-Bouldin and Calinski-Harabasz. "
            f"Silhouette {clu[k_best]['Silhouette']:.3f}, Davies-Bouldin {clu[k_best]['Davies-Bouldin']:.3f}.")
    st.write("Silhouette values below about 0.35 mean the groups overlap noticeably, which is normal for socio-economic data that "
             "changes gradually between areas. " + (f"Even so, the High Risk share differs by "
             f"{clu[k_best]['Risk Separation (pts)']:.1f} points across the {k_best} clusters, so the groups still carry useful "
             "information about risk." if clu[k_best]["Risk Separation (pts)"] >= 10 else
             "The High Risk share barely differs across clusters, so clustering is best used for segmentation, not prediction."))

    # ------------------------------------------------------------------ association rules
    section("5. Association Rules (Apriori)")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Frequent itemsets", ar["itemsets"])
    a2.metric("Rules found", ar["rules"])
    a3.metric("Max lift", f"{ar['max_lift']:.2f}" if ar["rules"] else "-")
    a4.metric("Avg confidence", f"{ar['avg_conf']:.2f}" if ar["rules"] else "-")
    st.write(f"Strongest rule: `{ar['top_rule']}`")
    st.write("Apriori is descriptive: it shows which places, times and crime types occur together. It does not use the risk label, "
             "so it cannot predict risk and is not ranked against the other families.")

    # ------------------------------------------------------------------ final verdict
    section("6. Final verdict: which algorithms suit our prediction system?")
    final = pd.DataFrame({
        "Task in the System": ["Predict High / Low risk", "Estimate crime rate per 100k", "Segment areas for planning", "Describe crime patterns"],
        "Recommended Algorithm": [c_best, r_best, k_best, "Apriori"],
        "Runner-up": [c_other, sorted(r_rank, key=r_rank.get)[1], sorted(k_rank, key=k_rank.get)[1], "-"]})
    st.dataframe(final, width="stretch", hide_index=True)
    st.markdown(f"- **{c_best}** is the main engine that tells users whether an area is High Risk or Low Risk.\n"
                f"- **{r_best}** backs it up with an expected crime rate, which says how serious the risk is.\n"
                f"- **{k_best}** groups areas into profiles so resources and patrols can be planned for each group.\n"
                "- **Apriori** explains which crime types, times and area types commonly go together.")

elif menu == "Risk Check":
    masthead("Assessment", "Area Risk Check",
             "Enter the indicators for an area to see its predicted risk level and expected crime rate")
    clf_model, reg_model = labeler(load_data())
    vals = input_form(ALL_NUM, "rc")
    row = pd.DataFrame([vals], columns=ALL_NUM)
    risk = clf_model.predict(row[CLS_FEATURES])[0]
    prob = clf_model.predict_proba(row[CLS_FEATURES])[0, 1]
    rate = max(0.0, float(reg_model.predict(row[REG_FEATURES])[0]))
    band = "high" if rate >= df.Crime_Rate.quantile(.66) else ("caution" if rate >= df.Crime_Rate.quantile(.33) else "")
    c1, c2 = st.columns(2)
    with c1:
        verdict("Predicted risk level", "High Risk" if risk else "Low Risk",
                f"Decision tree probability of High Risk: {prob * 100:.0f}%", "high" if risk else "")
    with c2:
        verdict("Expected crime rate", f"{rate:.1f} per 100k", f"Dataset average is {df.Crime_Rate.mean():.1f} per 100k.", band)

elif menu == "Register Crime":
    masthead("Reporting", "Register a Crime",
             "Add a new record. The dataset, dashboard and every model update immediately")
    clf_model, reg_model = labeler(load_data())
    with st.form("register_crime", clear_on_submit=False):
        section("Incident details")
        a, b, c = st.columns(3)
        city = a.selectbox("City", CITIES)
        area = b.selectbox("Area type", AREAS)
        crime = c.selectbox("Crime type", CRIMES)
        d, e, f = st.columns(3)
        when = d.date_input("Date of incident", value=date.today(), max_value=date.today())
        tod = e.selectbox("Time of day", TIMES)
        day_type = f.selectbox("Day type", DAY_TYPES)
        section("Area indicators")
        vals = input_form(ALL_NUM, "newcrime")
        notes = st.text_area("Notes (optional, stored with the report only)", max_chars=300)
        submitted = st.form_submit_button("Register crime")
    if submitted:
        row = pd.DataFrame([vals], columns=ALL_NUM)
        risk = "High Risk" if clf_model.predict(row[CLS_FEATURES])[0] else "Low Risk"
        rate = round(max(5.0, float(reg_model.predict(row[REG_FEATURES])[0])), 2)
        record = {"City": city, "Area_Type": area, "Year": when.year, "Month": when.month, "Day_Type": day_type,
                  "Crime_Type": crime, "Time_of_Day": tod, **dict(zip(ALL_NUM, vals)),
                  "Crime_Rate": rate, "Risk_Level": risk}
        rid = db.add_report(auth["username"], record, notes)
        verdict("Crime registered successfully", rid,
                f"{crime} in {city} ({area}). Estimated crime rate {rate:.1f} per 100k, classified as {risk}. "
                f"The dataset now holds {len(live_dataset()):,} records.", "high" if risk == "High Risk" else "")
        st.markdown("**Report status:** Registered")
        tracker(db.STATUSES[0])
        note(f"Keep your reference number <b>{rid}</b>. An administrator will review the report, and you can follow every "
             "status change under My Reports.")
    note("Crime rate and risk level are not typed in: they are estimated by the system's regression and decision-tree models "
         "from the indicators you enter, then stored as part of the record.")

elif menu == "My Reports":
    masthead("Reporting", "My Registered Crimes", "Records you have added, and where each one stands")
    mine = db.get_reports()
    mine = mine[mine.reported_by == auth["username"]]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total reports", len(mine))
    c2.metric("Registered", int((mine.status == "Registered").sum()))
    c3.metric("In progress", int(mine.status.isin(["Under Review", "Under Investigation"]).sum()))
    c4.metric("Resolved / closed", int(mine.status.isin(["Resolved", "Closed"]).sum()))
    if len(mine):
        show = mine.drop(columns=["id"]).rename(columns={"status": "Status", "reported_by": "Reported_By",
                                                         "reported_at": "Reported_At", "notes": "Notes"})
        lead = ["Record_ID", "Status", "Crime_Type", "City", "Reported_At"]
        show = show[lead + [c for c in show.columns if c not in lead]]
        st.dataframe(show, width="stretch", hide_index=True)
        section("Track a report")
        rid = st.selectbox("Reference number", show.Record_ID.tolist())
        current = show.loc[show.Record_ID == rid, "Status"].iloc[0]
        verdict("Current status", current, f"Reference {rid}",
                "caution" if current in ("Under Review", "Under Investigation") else "")
        tracker(current)
        st.write("**Status history**")
        st.dataframe(db.get_status_history(rid), width="stretch", hide_index=True)
        st.download_button("Download my reports (CSV)", show.to_csv(index=False).encode(),
                           f"my_crime_reports_{auth['username']}.csv", "text/csv")
    else:
        note("You have not registered any crimes yet. Use Register Crime in the sidebar.")

elif menu == "Registered Crimes":
    masthead("Administration", "User-Registered Crimes", "Everything added through the Register Crime form")
    rep = db.get_reports()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registered by users", len(rep))
    c2.metric("Awaiting review", int((rep.status == "Registered").sum()))
    c3.metric("In progress", int(rep.status.isin(["Under Review", "Under Investigation"]).sum()))
    c4.metric("Resolved / closed", int(rep.status.isin(["Resolved", "Closed"]).sum()))
    if len(rep):
        show = rep.drop(columns=["id"]).rename(columns={"status": "Status", "reported_by": "Reported_By",
                                                        "reported_at": "Reported_At", "notes": "Notes"})
        lead = ["Record_ID", "Status", "Reported_By", "Crime_Type", "City", "Reported_At"]
        show = show[lead + [c for c in show.columns if c not in lead]]
        pick = st.multiselect("Filter by status", db.STATUSES)
        st.dataframe(show[show.Status.isin(pick)] if pick else show, width="stretch", hide_index=True)
        st.download_button("Download registered crimes (CSV)", show.to_csv(index=False).encode(),
                           "registered_crimes.csv", "text/csv")

        section("Update report status")
        rid = st.selectbox("Record ID", show.Record_ID.tolist(), key="status_rid")
        current = show.loc[show.Record_ID == rid, "Status"].iloc[0]
        tracker(current)
        with st.form("update_status"):
            a, b = st.columns([1, 2])
            new_status = a.selectbox("New status", db.STATUSES, index=db.STATUSES.index(current))
            remarks = b.text_input("Remarks shown to the user (optional)")
            go = st.form_submit_button("Update status")
        if go:
            if new_status == current and not remarks.strip():
                note("Choose a different status or add a remark.", warn=True)
            else:
                db.update_status(rid, new_status, auth["username"], remarks)
                st.rerun()
        st.write("**Status history**")
        st.dataframe(db.get_status_history(rid), width="stretch", hide_index=True)

        section("Remove a record")
        st.caption("Deleting a record also removes it from the dataset and all models.")
        del_id = st.selectbox("Record ID to delete", show.Record_ID.tolist(), key="del_rid")
        if st.button("Delete this record"):
            db.delete_report(del_id)
            st.rerun()
    else:
        note("No crimes have been registered by users yet.")

elif menu == "Users":
    masthead("Administration", "User Accounts", "Administrators and registered end users")
    st.dataframe(db.list_users().rename(columns={"username": "Username", "full_name": "Full Name",
                                                 "role": "Role", "created_at": "Created"}),
                 width="stretch", hide_index=True)
    section("Add another administrator")
    with st.form("add_admin", clear_on_submit=True):
        a, b, c = st.columns(3)
        n = a.text_input("Full name"); u = b.text_input("Username"); p = c.text_input("Password", type="password")
        go = st.form_submit_button("Create administrator")
    if go:
        ok, msg = db.create_user(u, p, "admin", n)
        note(msg, warn=not ok)

elif menu == "Login History":
    masthead("Administration", "Login History", "Sign-ins, sign-outs, failed attempts and new registrations")
    stamp = date.today().strftime("%Y%m%d")
    for role, label, fname in [("admin", "Administrator", "admin_login_history"),
                               ("user", "End user", "user_login_history")]:
        hist = db.get_history(role)
        section(f"{label} login history")
        c1, c2, c3 = st.columns(3)
        c1.metric("Successful logins", int((hist.Event == "Login").sum()))
        c2.metric("Failed attempts", int((hist.Event == "Failed Login").sum()))
        c3.metric("Total events", len(hist))
        st.dataframe(hist, width="stretch", hide_index=True)
        st.download_button(f"Download {label.lower()} login history (CSV)", hist.to_csv(index=False).encode(),
                           f"{fname}_{stamp}.csv", "text/csv", key=f"dl_{role}")
        st.write("")
