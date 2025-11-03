#!/usr/bin/env python3
# forecast

import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ========= theme (polished dark blue) =========
THEME_BG   = "#0A4AA6"
GRID_CLR   = "#9EC0FF"
TXT_CLR    = "#E9F2FF"
COL_FORE   = "#58D0C9"  # teal
LINE_W     = 3.0
MARK_SZ    = 7.0

TITLE_FZ   = 15
LABEL_FZ   = 12
TICK_FZ    = 9
LEGEND_FZ  = 10

def apply_blue_theme(fig, ax):
    fig.patch.set_facecolor(THEME_BG)
    ax.set_facecolor(THEME_BG)
    ax.grid(True, color=GRID_CLR, alpha=0.25, linewidth=1.0)
    for s in ax.spines.values():
        s.set_color(GRID_CLR); s.set_alpha(0.35)
    ax.tick_params(colors=TXT_CLR, labelsize=TICK_FZ)
    ax.xaxis.label.set_color(TXT_CLR); ax.xaxis.label.set_fontsize(LABEL_FZ)
    ax.yaxis.label.set_color(TXT_CLR); ax.yaxis.label.set_fontsize(LABEL_FZ)
    ax.title.set_color(TXT_CLR);       ax.title.set_fontsize(TITLE_FZ)

def annotate_points(ax, x_vals, y_vals, color=TXT_CLR):
    """Put numeric labels on each dot (slightly above)."""
    for xi, yi in zip(x_vals, y_vals):
        ax.annotate(f"{int(round(yi))}",
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 8),   # upward offset
                    ha="center",
                    color=color,
                    fontsize=TICK_FZ)

# ========= data helpers =========
import mysql.connector
import re

def load_db_config(index_php_path=r"C:\\xampp\\htdocs\\E-consulta\\index.php"):
    """Auto-read MySQL credentials from your PHP config file."""
    try:
        content = Path(index_php_path).read_text(encoding="utf-8", errors="ignore")
        match = re.search(
            r'mysqli\(["\'](.*?)["\'],\s*["\'](.*?)["\'],\s*["\'](.*?)["\'],\s*["\'](.*?)["\']\)',
            content
        )
        if match:
            host, user, password, database = match.groups()
        else:
            host, user, password, database = "localhost", "root", "", "econsulta"
    except Exception as e:
        print(f"[Warning] Could not read index.php: {e}")
        host, user, password, database = "localhost", "root", "", "econsulta"
    return {"host": host, "user": user, "password": password, "database": database}

def load_monthly_total_from_mysql() -> pd.DataFrame:
    """Load existing monthly totals from machine_learning table instead of CSV."""
    cfg = load_db_config()
    conn = mysql.connector.connect(**cfg)
    query = "SELECT year, month, month_num, weather, disease, cases FROM machine_learning"
    df = pd.read_sql(query, conn)
    conn.close()

    # Normalize column names
    df.rename(columns={"month_num": "monthnum"}, inplace=True)

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


MONTHS = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]
M2N = {m:i+1 for i,m in enumerate(MONTHS)}

def month_cyclical(m: int):
    ang = 2 * math.pi * (m - 1) / 12.0
    return np.sin(ang), np.cos(ang)

def infer_weather(m: int):
    if m in (3,4,5): return "Hot"
    if m in (6,7,8,9,10,11): return "Rainy"
    return "Cold"

def load_monthly_total(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}

    col_year     = low.get("year")
    col_month    = low.get("month")
    col_monthnum = low.get("monthnum")
    col_weather  = low.get("weather")
    col_cases    = low.get("total_cases") or low.get("cases") or low.get("count") or low.get("case_count")
    if col_cases is None:
        raise ValueError("CSV needs a cases column (e.g., total_cases, cases).")

    # month handling
    if col_monthnum is None and col_month is None:
        raise ValueError("CSV needs 'monthnum' or 'month'.")
    if col_monthnum is None:
        df[col_month] = df[col_month].astype(str).str.title()
        df["monthnum"] = df[col_month].map(M2N); col_monthnum = "monthnum"
    else:
        df[col_monthnum] = pd.to_numeric(df[col_monthnum], errors="coerce")
    if col_month is None:
        df["month"] = df[col_monthnum].map({i+1:n for i,n in enumerate(MONTHS)}); col_month = "month"
    else:
        df[col_month] = df[col_month].astype(str).str.title()

    # year handling
    if col_year is None:
        for cand in ["date","consult_date","created_at","timestamp"]:
            if cand in low:
                dt = pd.to_datetime(df[low[cand]], errors="coerce")
                df["year"] = dt.dt.year; col_year = "year"; break
    if col_year is None:
        raise ValueError("Need a 'year' column.")

    # weather
    if col_weather is None:
        df["weather"] = df[col_monthnum].apply(lambda m: infer_weather(int(m))); col_weather = "weather"
    else:
        df[col_weather] = df[col_weather].astype(str).str.title()

    # types
    df[col_year] = pd.to_numeric(df[col_year], errors="coerce").astype(int)
    df[col_monthnum] = pd.to_numeric(df[col_monthnum], errors="coerce").astype(int)
    df[col_cases] = pd.to_numeric(df[col_cases], errors="coerce").fillna(0).astype(int)

    # aggregate to monthly totals
    monthly = (df.groupby([col_year, col_month, col_monthnum, col_weather], as_index=False)[col_cases]
                 .sum()
                 .rename(columns={col_cases: "total_cases"})
                 .sort_values([col_year, col_monthnum])
                 .reset_index(drop=True)
                 .rename(columns={col_year:"year", col_month:"month",
                                  col_monthnum:"monthnum", col_weather:"weather"}))
    return monthly

def all_diseases_by_calendar_month_with_props(csv_path: Path):
    try:
        raw = pd.read_csv(csv_path)
    except Exception:
        return {}, {}

    cols = {c.lower(): c for c in raw.columns}
    if "disease" not in cols:
        return {}, {}

    # derive monthnum
    if cols.get("monthnum") in raw:
        raw["monthnum"] = pd.to_numeric(raw[cols["monthnum"]], errors="coerce")
    else:
        raw["monthnum"] = np.nan
    if raw["monthnum"].isna().all() and "month" in cols:
        raw["monthnum"] = raw[cols["month"]].astype(str).str.title().map(M2N)

    raw["disease"] = raw[cols["disease"]].astype(str).str.strip()
    if "cases" in cols:
        raw["cases"] = pd.to_numeric(raw[cols["cases"]], errors="coerce").fillna(0)
    else:
        raw["cases"] = 1

    raw = raw.dropna(subset=["monthnum"])
    raw["monthnum"] = raw["monthnum"].astype(int)

    by_md = (raw.groupby(["monthnum", "disease"], as_index=False)["cases"]
               .sum()
               .rename(columns={"cases": "total"}))

    month2names, month2props = {}, {}
    for m, g in by_md.groupby("monthnum"):
        g_sorted = g.sort_values("total", ascending=False).reset_index(drop=True)
        names = g_sorted["disease"].tolist()
        totals = g_sorted["total"].astype(float).values
        denom = totals.sum()
        props = (totals / denom).tolist() if denom > 0 else [0.0]*len(totals)
        month2names[int(m)] = names
        month2props[int(m)] = dict(zip(names, props))
    return month2names, month2props

def global_disease_props(csv_path: Path):
    raw = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in raw.columns}
    if "disease" not in cols:
        return {}
    raw["disease"] = raw[cols["disease"]].astype(str).str.strip()
    if "cases" in cols:
        raw["cases"] = pd.to_numeric(raw[cols["cases"]], errors="coerce").fillna(0)
    else:
        raw["cases"] = 1
    g = raw.groupby("disease", as_index=False)["cases"].sum()
    total = g["cases"].sum()
    if total <= 0:
        return {}
    return {row["disease"]: row["cases"] / total for _, row in g.iterrows()}

def build_features(frame: pd.DataFrame):
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

    feature_cols = [c for c in df.columns if c not in ["total_cases","year","month","monthnum"]]
    return df, feature_cols

def iterative_forecast(pipe, feat_cols, history_df, future_rows):
    preds = []
    for i in range(len(future_rows)):
        tmp = pd.concat([
            history_df,
            pd.DataFrame({
                "year":     future_rows["year"].iloc[:i+1],
                "month":    future_rows["month"].iloc[:i+1],
                "monthnum": future_rows["monthnum"].iloc[:i+1],
                "weather":  future_rows["weather"].iloc[:i+1],
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

def allocate_per_disease(pred_total: int, monthnum: int, month2props: dict, global_props: dict = None):
    props = month2props.get(int(monthnum), {})
    if not props or sum(props.values()) == 0:
        props = (global_props or {}).copy()

    if not props:
        all_names = sorted({d for mp in month2props.values() for d in mp.keys()})
        if not all_names:
            return []
        props = {d: 1.0 / len(all_names) for d in all_names}

    s = sum(props.values())
    if s > 0:
        props = {d: v / s for d, v in props.items()}

    alloc = {d: pred_total * p for d, p in props.items()}
    rounded = {d: int(np.floor(v)) for d, v in alloc.items()}
    remainder = int(pred_total - sum(rounded.values()))
    fracs = sorted(((d, alloc[d] - rounded[d]) for d in rounded),
                   key=lambda x: x[1], reverse=True)
    for i in range(max(0, remainder)):
        rounded[fracs[i % len(fracs)][0]] += 1

    return sorted(rounded.items(), key=lambda kv: kv[1], reverse=True)

def make_bullet_labels(rows_df: pd.DataFrame, month2props: dict, global_props: dict,
                       header_fmt=lambda y,m: f"{int(y)}-{int(m):02d}"):
    bullet = "\u2022"  # •
    dash   = "\u2013"  # –
    labels = []
    for _, r in rows_df.iterrows():
        y, m, total = int(r["year"]), int(r["monthnum"]), int(r["predicted_cases"])
        alloc = allocate_per_disease(total, m, month2props, global_props)
        lines = [header_fmt(y, m)]
        for d, v in alloc:
            lines.append(f"{bullet} {d} {dash} {v}")
        labels.append("\n".join(lines))
    return labels

def print_per_disease_tables(window_name: str, fc_df: pd.DataFrame, month2props: dict, global_props: dict):
    print(f"\n=== Per-Disease Estimated Cases — {window_name} ===")
    window_totals = {}
    for _, row in fc_df.iterrows():
        y, m = int(row["year"]), int(row["monthnum"])
        pred = int(row["predicted_cases"])
        alloc = allocate_per_disease(pred, m, month2props, global_props)
        label = f"{y}-{m:02d}"
        print(f"\n{label}  (Total: {pred})")
        for d, v in alloc:
            print(f"  - {d}: {v}")
            window_totals[d] = window_totals.get(d, 0) + v
    print(f"\n--- {window_name} TOTAL per disease ---")
    for d, v in sorted(window_totals.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  • {d}: {v}")

def main():
    print("🔹 Loading dataset from MySQL...")

    # ---- 1️⃣ Load dataset ----
    try:
        monthly = load_monthly_total_from_mysql()
        if monthly.empty:
            print("⚠️ machine_learning table is empty. Nothing to forecast.")
            return
    except Exception as e:
        print(f"❌ Failed to load data from MySQL: {e}")
        return

    # Normalize column names
    monthly.columns = [c.lower().strip().replace(" ", "_") for c in monthly.columns]
    if "month_num" in monthly.columns and "monthnum" not in monthly.columns:
        monthly.rename(columns={"month_num": "monthnum"}, inplace=True)

    # Ensure numeric types
    monthly["year"] = pd.to_numeric(monthly["year"], errors="coerce")
    monthly["monthnum"] = pd.to_numeric(monthly["monthnum"], errors="coerce")
    monthly["total_cases"] = pd.to_numeric(monthly["total_cases"], errors="coerce").fillna(0)
    monthly = monthly.dropna(subset=["year", "monthnum"])
    monthly = monthly.sort_values(["year", "monthnum"]).reset_index(drop=True)

    print(f"✅ Loaded {len(monthly)} monthly records from MySQL.")

    # ---- 2️⃣ Compute disease proportions ----
    try:
        _, month2props = all_diseases_by_calendar_month_with_props()
        global_props = global_disease_props()

    except Exception as e:
        print(f"⚠️ Failed to compute disease proportions: {e}")
        month2props, global_props = {}, {}

    # ---- 3️⃣ Build features ----
    try:
        df_feat, feat_cols = build_features(monthly)
    except Exception as e:
        print(f"❌ Feature-building error: {e}")
        return

    # ---- 4️⃣ Train & Evaluate model ----
    X_all = df_feat[feat_cols].values
    y_all = df_feat["total_cases"].values
    n = len(y_all)
    if n < 6:
        print("⚠️ Not enough data points to train a model.")
        return
    split = max(1, int(n * 0.8))
    X_tr, y_tr = X_all[:split], y_all[:split]
    X_te, y_te = X_all[split:], y_all[split:]

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42)),
    ])
    pipe.fit(X_tr, y_tr)

    yhat_te = pipe.predict(X_te)
    r2 = r2_score(y_te, yhat_te)
    mae = mean_absolute_error(y_te, yhat_te)
    rmse = mean_squared_error(y_te, yhat_te, squared=False)

    print("\n=== TEST (80/20) ===")
    print(f"R² Accuracy: {r2*100:.2f}%")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # ---- 5️⃣ Retrain on all data for final forecasting ----
    pipe.fit(X_all, y_all)
    hist = monthly[["year", "month", "monthnum", "weather", "total_cases"]].copy()

    # ---- 6️⃣ Find latest record ----
    last_record = monthly.sort_values(["year", "monthnum"]).iloc[-1]
    last_year = int(last_record["year"])
    last_monthnum = int(last_record["monthnum"])
    print(f"\n📅 Latest record: {MONTHS[last_monthnum-1]} {last_year}")

    # ---- 7️⃣ Forecast remaining months of 2025 ----
        # ---- 7️⃣ Continuous Forecast — One Year Ahead (Preserves All Logic) ----
    target_year = last_year + 1  # only forecast one year ahead
    y, m = last_year, last_monthnum + 1
    if m == 13:
        y, m = y + 1, 1

    future_rows = []
    while (y < target_year) or (y == target_year and m <= last_monthnum):
        future_rows.append({
            "year": y,
            "month": MONTHS[m - 1],
            "monthnum": m,
            "weather": infer_weather(m),
        })
        m += 1
        if m == 13:
            y, m = y + 1, 1

    future_df = pd.DataFrame(future_rows)
    if future_df.empty:
        print("✅ No future months to forecast — data already includes one full year.")
        return
    
    print(f"🔁 Continuing forecast one year beyond {MONTHS[last_monthnum-1]} {last_year}...")


    print(f"📈 Forecasting next {len(future_df)} months "
          f"({MONTHS[last_monthnum-1]} {last_year} → "
          f"{MONTHS[future_df.iloc[-1]['monthnum']-1]} {future_df.iloc[-1]['year']})")

    # Perform forecasting
    fc_all = iterative_forecast(pipe, feat_cols, hist, future_df)

    # Show per-disease breakdown (reusing your same logic)
    print_per_disease_tables("Continuous Forecast (Next 12 Months)", fc_all, month2props, global_props)

    # ---- Visualization (unchanged style) ----
    labels = [f"{int(r.year)}-{int(r.monthnum):02d}" for _, r in fc_all.iterrows()]
    y_pred = fc_all["predicted_cases"].astype(float).values
    x_idx = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 7))
    apply_blue_theme(fig, ax)
    ax.plot(
        x_idx, y_pred,
        marker="o", linewidth=3, markersize=7,
        color=COL_FORE, label="Forecast (Next 12 Months)"
    )
    annotate_points(ax, x_idx, y_pred)
    ax.set_xlabel("Month")
    ax.set_ylabel("Predicted Cases")
    ax.set_title(f"Forecast — {MONTHS[last_monthnum-1]} {last_year} → "
                 f"{MONTHS[future_df.iloc[-1]['monthnum']-1]} {future_df.iloc[-1]['year']}")
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels, rotation=45, ha="right", color=TXT_CLR, fontsize=9)
    plt.tight_layout()
    plt.show()

# ---- standard entry point ----
if __name__ == "__main__":
    main()
