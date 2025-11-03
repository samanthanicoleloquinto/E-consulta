# service.py — Barangay Forecasting API (continuous + default 5-year horizon + disease breakdown, case-insensitive)
# year_to_eff = last_hist_year + Added year here
# Run: uvicorn service:app --host 0.0.0.0 --port 8000 --reload
# pip install uvicorn fastapi
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process (command para sa activate.ps1 kung disabled script)

import io, os, base64, math, traceback, re
from pathlib import Path
from typing import List, Optional, Dict
import mysql.connector  

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent.resolve()

# Automatically read DB credentials from your existing PHP config file
def load_db_config(index_php_path=None):
    """
    Load MySQL credentials.
    - Local: read from index.php (XAMPP)
    - Render: use environment variables
    """
    try:
        # 🖥️ LOCAL (Windows / XAMPP)
        local_path = index_php_path or r"C:\\xampp\\htdocs\\E-consulta\\index.php"
        if Path(local_path).exists():
            content = Path(local_path).read_text(encoding="utf-8", errors="ignore")
            match = re.search(
                r'mysqli\(["\'](.*?)["\'],\s*["\'](.*?)["\'],\s*["\'](.*?)["\'],\s*["\'](.*?)["\']\)',
                content
            )
            if match:
                host, user, password, database = match.groups()
                return {"host": host, "user": user, "password": password, "database": database}
        
        # 🌐 RENDER (Environment Variables)
        host = os.getenv("DB_HOST", "localhost")
        user = os.getenv("DB_USER", "root")
        password = os.getenv("DB_PASS", "")
        database = os.getenv("DB_NAME", "econsulta")

        return {"host": host, "user": user, "password": password, "database": database}

    except Exception as e:
        print(f"[Warning] Could not read DB config: {e}")
        return {"host": "localhost", "user": "root", "password": "", "database": "econsulta"}



MONTHS = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]
M2N = {m:i+1 for i,m in enumerate(MONTHS)}

app = FastAPI(title="Barangay Forecasting API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ---------------- Utilities ----------------
def month_cyclical(m: int):
    ang = 2 * math.pi * (m - 1) / 12.0
    return np.sin(ang), np.cos(ang)

def infer_weather(m: int):
    if m in (3,4,5): return "Hot"
    if m in (6,7,8,9,10,11): return "Rainy"
    return "Cold"

def _fail500(msg: str, e: Exception):
    traceback.print_exc()
    raise HTTPException(status_code=500, detail=f"{msg}: {type(e).__name__}: {e}")

# ---- Case-insensitive disease column resolver ----
DISEASE_ALIASES = [
    "disease", "diagnosis", "dx", "condition", "icd", "icd_name",
    "icd10", "case_type", "case_name", "health_issue", "complaint"
]

def _resolve_disease_col(df: pd.DataFrame) -> Optional[str]:
    """
    Return the actual column name to use as the 'disease' field, case-insensitive.
    Falls back to light heuristics if no exact alias match.
    """
    cols_lower = {str(c).lower(): str(c) for c in df.columns}
    # 1) exact alias match (case-insensitive)
    for key in DISEASE_ALIASES:
        if key in cols_lower:
            return cols_lower[key]
    # 2) heuristic contains match
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ["disea", "diag", "icd", "condit", "complaint", "health"]):
            return str(c)
    return None

# ---------------- Data Loading / Aggregation ----------------
def load_monthly_total_from_mysql() -> pd.DataFrame:
    """Load data from the machine_learning MySQL table instead of CSV."""
    cfg = load_db_config()
    conn = mysql.connector.connect(**cfg)
    query = "SELECT year, month, month_num AS monthnum, weather, disease, cases FROM machine_learning"
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        raise ValueError("machine_learning table is empty.")

    df["year"] = df["year"].astype(int)
    df["monthnum"] = df["monthnum"].astype(int)
    df["cases"] = pd.to_numeric(df["cases"], errors="coerce").fillna(0).astype(int)

    monthly = (
        df.groupby(["year", "month", "monthnum", "weather"], as_index=False)["cases"]
        .sum()
        .rename(columns={"cases": "total_cases"})
        .sort_values(["year", "monthnum"])
        .reset_index(drop=True)
    )
    return monthly


def all_diseases_by_calendar_month_with_props_from_mysql() -> tuple[dict, dict]:
    """
    Compute per-month and global disease proportions directly from MySQL.
    Returns:
      month2props: {monthnum -> {disease -> proportion}}
      global_props: {disease -> overall proportion}
    """
    cfg = load_db_config()
    conn = mysql.connector.connect(**cfg)
    query = "SELECT month_num AS monthnum, disease, cases FROM machine_learning"
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        return {}, {}

    df["monthnum"] = pd.to_numeric(df["monthnum"], errors="coerce")
    df["cases"] = pd.to_numeric(df["cases"], errors="coerce").fillna(0)
    df["disease"] = df["disease"].astype(str).str.strip()

    # ---- Monthly proportions ----
    by_md = (
        df.groupby(["monthnum", "disease"], as_index=False)["cases"]
        .sum()
        .rename(columns={"cases": "total"})
    )

    month2props = {}
    for m, g in by_md.groupby("monthnum"):
        denom = float(g["total"].sum())
        month2props[int(m)] = (
            {row["disease"]: float(row["total"]) / denom for _, row in g.iterrows()}
            if denom > 0 else {}
        )

    # ---- Global proportions ----
    g_all = df.groupby("disease", as_index=False)["cases"].sum()
    total_all = float(g_all["cases"].sum())
    global_props = (
        {row["disease"]: float(row["cases"]) / total_all for _, row in g_all.iterrows()}
        if total_all > 0 else {}
    )

    return month2props, global_props


def allocate_per_disease(pred_total: int, monthnum: int, month2props: Dict[int, Dict[str,float]], global_props: Dict[str,float]):
    props = dict(month2props.get(int(monthnum), {}))
    if not props:
        props = dict(global_props)  # fallback to global
    if not props:
        return []  # no disease info available

    # normalize
    s = sum(props.values())
    if s <= 0:
        return []
    props = {d: v/s for d, v in props.items()}

    alloc = {d: pred_total * p for d, p in props.items()}
    rounded = {d: int(np.floor(v)) for d, v in alloc.items()}
    remainder = int(pred_total - sum(rounded.values()))
    if rounded:
        fracs = sorted(((d, alloc[d] - rounded[d]) for d in rounded), key=lambda x: x[1], reverse=True)
        for i in range(remainder):
            rounded[fracs[i % len(fracs)][0]] += 1

    # sorted by cases desc
    return sorted(({"name": d, "cases": c} for d, c in rounded.items()), key=lambda x: x["cases"], reverse=True)

# ---------------- Feature Engineering ----------------
def build_features(frame: pd.DataFrame):
    try:
        df = frame.copy()
        df["total_cases"] = pd.to_numeric(df["total_cases"], errors="coerce").fillna(0)

        sc = df["monthnum"].apply(lambda m: month_cyclical(int(m)))
        df["m_sin"] = [s for (s,c) in sc]
        df["m_cos"] = [c for (s,c) in sc]

        for L in [1,2,3,6]:
            df[f"lag_{L}"] = df["total_cases"].shift(L)
        df["roll3"]  = df["total_cases"].rolling(3,  min_periods=1).mean()
        df["roll6"]  = df["total_cases"].rolling(6,  min_periods=1).mean()
        df["roll12"] = df["total_cases"].rolling(12, min_periods=1).mean()

        df["weather"] = df["weather"].astype(str).str.title()
        df = pd.get_dummies(df, columns=["weather"], drop_first=False)

        req = [f"lag_{L}" for L in [1,2,3,6]]
        df = df.dropna(subset=req).reset_index(drop=True)
        if df.empty:
            raise ValueError("Not enough rows after generating lags/rolling windows.")

        feat_cols = [c for c in df.columns if c not in ["total_cases","year","month","monthnum"]]
        return df, feat_cols

    except Exception as e:
        _fail500("Failed to build features", e)

# ---------------- Modeling / Forecasting ----------------
def fit_pipeline(X_tr, y_tr):
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("ridge",  Ridge(alpha=1.0, random_state=42)),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe

def iterative_forecast(pipe, feat_cols, history_df, future_rows):
    preds = []
    for i in range(len(future_rows)):
        tmp = pd.concat([
            history_df,
            pd.DataFrame({
                "year": future_rows["year"].iloc[:i+1],
                "month": future_rows["month"].iloc[:i+1],
                "monthnum": future_rows["monthnum"].iloc[:i+1],
                "weather": future_rows["weather"].iloc[:i+1],
                "total_cases": pd.Series(list(preds) + [np.nan])
            })
        ], ignore_index=True)

        if i > 0:
            tmp.loc[len(history_df):len(history_df)+i-1, "total_cases"] = preds

        sc = tmp["monthnum"].apply(lambda m: month_cyclical(int(m)))
        tmp["m_sin"] = [s for (s,c) in sc]
        tmp["m_cos"] = [c for (s,c) in sc]
        for L in [1,2,3,6]:
            tmp[f"lag_{L}"] = tmp["total_cases"].shift(L)
        tmp["roll3"]  = tmp["total_cases"].rolling(3,  min_periods=1).mean()
        tmp["roll6"]  = tmp["total_cases"].rolling(6,  min_periods=1).mean()
        tmp["roll12"] = tmp["total_cases"].rolling(12, min_periods=1).mean()
        tmp["weather"] = tmp["weather"].astype(str).str.title()
        tmp = pd.get_dummies(tmp, columns=["weather"], drop_first=False)

        for col in feat_cols:
            if col not in tmp.columns:
                tmp[col] = 0.0

        row = tmp.iloc[[-1]][feat_cols]
        y_next = float(pipe.predict(row.values)[0])
        preds.append(max(0.0, y_next))

    out = future_rows.copy()
    out["predicted_cases"] = np.round(preds).astype(int)
    return out

def make_future_year(year: int) -> pd.DataFrame:
    return pd.DataFrame({
        "year":     [year]*12,
        "month":    MONTHS,
        "monthnum": list(range(1, 13)),
        "weather":  [infer_weather(m) for m in range(1, 13)],
    })

def make_continuous_future(monthly: pd.DataFrame, year_to: int) -> pd.DataFrame:
    last_y = int(monthly["year"].iloc[-1])
    last_m = int(monthly["monthnum"].iloc[-1])

    if last_m < 12:
        y, m = last_y, last_m
    else:
        y, m = last_y + 1, 1

    rows = []
    
    while (y < year_to) or (y == year_to and m <= 12):
        rows.append({
            "year": y,
            "month": MONTHS[m-1],
            "monthnum": m,
            "weather": infer_weather(m)
        })
        m += 1
        if m == 13:
            y, m = y + 1, 1

    return pd.DataFrame(rows)

# ---------------- API Schemas ----------------
class Metrics(BaseModel):
    r2_accuracy: float
    mae: float
    rmse: float

class DiseaseBreakdown(BaseModel):
    name: str
    cases: int

class ForecastRow(BaseModel):
    year: int
    month: str
    monthnum: int
    predicted_cases: int
    diseases: List[DiseaseBreakdown]

class ForecastResponse(BaseModel):
    metrics: Metrics
    forecast: List[ForecastRow]
    plot_png_base64: Optional[str] = None

# ---------------- Endpoints ----------------
@app.get("/health")
def health():
    info = {"cwd": str(os.getcwd()), "db_source": "machine_learning (MySQL)", "status": "connected"}
    try:
        cfg = load_db_config()
        conn = mysql.connector.connect(**cfg)
        query = "SELECT * FROM machine_learning LIMIT 5"
        df = pd.read_sql(query, conn)
        conn.close()
        
        if df.empty:
            info["message"] = "⚠️ The machine_learning table is empty."
        else:
            info["detected_columns"] = list(map(str, df.columns))
            info["sample_rows"] = df.head(3).to_dict(orient="records")
    except Exception as e:
            info["status"] = "error"
            info["error_detail"] = f"{type(e).__name__}: {e}"

    return info


@app.get("/metrics", response_model=Metrics)
def get_metrics():
    try:
        monthly = load_monthly_total_from_mysql()
        df_feat, feat_cols = build_features(monthly)

        X_all = df_feat[feat_cols].values
        y_all = df_feat["total_cases"].values
        n = len(y_all); split = max(1, int(n*0.8))
        X_tr, y_tr = X_all[:split], y_all[:split]
        X_te, y_te = X_all[split:], y_all[split:]

        pipe = fit_pipeline(X_tr, y_tr)
        y_hat = pipe.predict(X_te)

        r2   = r2_score(y_te, y_hat)
        mae  = mean_absolute_error(y_te, y_hat)
        rmse = float(np.sqrt(mean_squared_error(y_te, y_hat)))

        return Metrics(r2_accuracy=round(r2*100, 2), mae=round(mae, 2), rmse=round(rmse, 2))
    except Exception as e:
        _fail500("Failed in /metrics", e)

@app.get("/forecast", response_model=ForecastResponse)
def forecast(
    year_from: int = Query(2025, ge=2025),            # kept for backward compat; ignored for continuity
    year_to: Optional[int] = Query(None, ge=2025),    # optional; if None and years_ahead None -> default 5 years ahead
    years_ahead: Optional[int] = Query(None, ge=1),   # optional override (e.g., 2 = next 2 years)
    with_plot: bool = False
):
    
    try:
        # 1) Load & features
        monthly = load_monthly_total_from_mysql()
        df_feat, feat_cols = build_features(monthly)

        # 2) Determine effective horizon (default: years ahead)
        last_hist_year  = int(monthly["year"].iloc[-1])
        last_hist_month = int(monthly["monthnum"].iloc[-1])

        if year_to is not None:
            year_to_eff = int(year_to)
        elif years_ahead is not None:
            year_to_eff = last_hist_year + int(years_ahead)
        else:
            year_to_eff = last_hist_year  # year_to_eff = last_hist_year + Added year here

        if year_to_eff < last_hist_year:
            raise HTTPException(status_code=400, detail="'year_to' (effective) must be >= last historical year")

        # 3) Accuracy on historical 80/20 (unchanged)
        X_all = df_feat[feat_cols].values
        y_all = df_feat["total_cases"].values
        n = len(y_all); split = max(1, int(n*0.8))
        X_tr, y_tr = X_all[:split], y_all[:split]
        X_te, y_te = X_all[split:], y_all[split:]
        pipe = fit_pipeline(X_tr, y_tr)
        y_hat = pipe.predict(X_te)

        r2   = r2_score(y_te, y_hat)
        mae  = mean_absolute_error(y_te, y_hat)
        rmse = float(np.sqrt(mean_squared_error(y_te, y_hat)))

        # 4) Refit on ALL history for forecasting
        pipe.fit(X_all, y_all)
        hist = monthly[["year","month","monthnum","weather","total_cases"]].copy()

        # 5) Build a continuous future grid (from next month → Dec of year_to_eff)
        future_full = make_continuous_future(monthly, year_to_eff)
        if future_full.empty:
            return ForecastResponse(
                metrics=Metrics(r2_accuracy=round(r2*100, 2), mae=round(mae, 2), rmse=round(rmse, 2)),
                forecast=[],
                plot_png_base64=None
            )

        # 6) Iterative forecast across full future span
        fc_all = iterative_forecast(pipe, feat_cols, hist, future_full)

        # 7) Disease allocation per month (case-insensitive)
        month2props, global_props = all_diseases_by_calendar_month_with_props_from_mysql()

        # 8) Optional plot (kept)
        plot_png_base64 = None
        if with_plot:
            x = np.arange(len(fc_all))
            y = fc_all["predicted_cases"].astype(float).values
            labels = [f"{r.year}-{r.monthnum:02d}" for r in fc_all.itertuples()]
            fig, ax = plt.subplots(figsize=(12,5))
            ax.plot(x, y, marker="o", linewidth=2.8)
            ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_title(f"Continuous Forecast from {last_hist_year}-{last_hist_month:02d} → {year_to_eff}-12")
            ax.set_ylabel("Cases")
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=160, bbox_inches="tight"); plt.close(fig)
            plot_png_base64 = base64.b64encode(buf.getvalue()).decode("ascii")

        rows: List[ForecastRow] = []
        for r in fc_all.itertuples():
            diseases = allocate_per_disease(
                pred_total=int(r.predicted_cases),
                monthnum=int(r.monthnum),
                month2props=month2props,
                global_props=global_props
            )
            rows.append(ForecastRow(
                year=int(r.year),
                month=str(r.month),
                monthnum=int(r.monthnum),
                predicted_cases=int(r.predicted_cases),
                diseases=[DiseaseBreakdown(**d) for d in diseases]
            ))

        return ForecastResponse(
            metrics=Metrics(r2_accuracy=round(r2*100, 2), mae=round(mae, 2), rmse=round(rmse, 2)),
            forecast=rows,
            plot_png_base64=plot_png_base64
        )

    except Exception as e:
        _fail500("Failed in /forecast", e)

