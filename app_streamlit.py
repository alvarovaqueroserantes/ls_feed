# app_streamlit_moderno.py
# -*- coding: utf-8 -*-
"""
Streamlit app mejorada: diseño moderno, minimalista y detallado.
Sustituye a tu app original. Incluye CSS embebido para evitar ficheros externos.
Requisitos:
    pip install streamlit streamlit-echarts pandas numpy pyarrow statsmodels scipy
Ejecución:
    streamlit run app_streamlit_moderno.py
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, ttest_ind

import streamlit as st
from streamlit_echarts import st_echarts

# ----------------------------- PARÁMETROS GLOBALES -----------------------------

BASE_DIR = Path(__file__).resolve().parent
IN_DIR = BASE_DIR / "data_drive" / "_parquet_export" / "split"
TS_COMBINED = {
    "LS1": BASE_DIR / "data_drive" / "TS1_combinado.csv",
    "LS5": BASE_DIR / "data_drive" / "TS2_combinado.csv",
}
TS_FOLDERS = {
    "LS1": BASE_DIR / "data_drive" / "TS1",
    "LS5": BASE_DIR / "data_drive" / "TS2",
}

TIMEZONE = "Europe/Madrid"
ASSUME_LOCALTIME = True

DIAM_CM = 30.0
R_CM = DIAM_CM / 2.0
AREA_CM2 = math.pi * (R_CM**2)
DENS_G_PER_ML = 1.0

ROLL_MED_MINUTES_DEFAULT = 2.0
HAMPEL_WINDOW_MIN_DEFAULT = 3.0
HAMPEL_K_DEFAULT = 3.0
REFILL_MIN_SEP_MIN_DEFAULT = 10.0
REFILL_JUMP_MM_DEFAULT = -50.0

# ----------------------------- ESTILO (CSS EMBEBIDO) ---------------------------

CSS_DEFAULT = """
:root{
  --bg:#0f1720;
  --card:#0b1220;
  --muted:#94a3b8;
  --accent:#60a5fa;
  --accent-2:#34d399;
  --glass: rgba(255,255,255,0.03);
  --kpi-bg: linear-gradient(90deg, rgba(96,165,250,0.08), rgba(52,211,153,0.06));
  --radius:12px;
  --font-sans: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}
html, body, [data-testid="stAppViewContainer"]{
  background: linear-gradient(180deg, #071026 0%, #071026 40%, #071a2b 100%) !important;
  color: #e6eef8;
  font-family: var(--font-sans);
}
.stApp, .stApp .main{
  padding: 14px 18px 36px 18px;
}
header .decoration, footer {display:none;}
.block-container {padding-top: 8px;}
.kpi-row {display:flex; gap:12px; align-items:stretch; margin-bottom:14px; flex-wrap:wrap;}
.kpi-card{
  background: var(--kpi-bg);
  border-radius: var(--radius);
  padding: 14px;
  min-width: 180px;
  box-shadow: 0 6px 18px rgba(2,6,23,0.7);
  color: #e8f1ff;
}
.kpi-title{font-size:13px; color:var(--muted); margin-bottom:6px;}
.kpi-value{font-size:22px; font-weight:600;}
.kpi-sub{font-size:12px; color:var(--muted); margin-top:6px;}
.panel{
  background: var(--glass);
  border-radius: var(--radius);
  padding: 14px;
  box-shadow: 0 6px 20px rgba(2,6,23,0.7);
  margin-bottom:14px;
}
.small-muted{color:var(--muted); font-size:13px;}
.title-row{display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:6px;}
.app-title{font-size:20px; font-weight:700; letter-spacing:0.2px;}
.logo-pill{background:linear-gradient(90deg,#60a5fa,#34d399); padding:8px 12px; border-radius:999px; color:#04263a; font-weight:700;}
.inputs-grid{display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:10px; margin-bottom:12px;}
.footer-note{font-size:12px; color:var(--muted); margin-top:10px;}
.stDownloadButton>button {background: linear-gradient(90deg,#60a5fa,#34d399); color:#04263a; border:none;}
"""

def inject_css():
    css_path = BASE_DIR / "styles.css"
    if css_path.exists():
        try:
            css_content = css_path.read_text(encoding="utf-8")
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
            return
        except Exception:
            pass
    st.markdown(f"<style>{CSS_DEFAULT}</style>", unsafe_allow_html=True)

# ----------------------------- UTILIDADES -------------------------------------

def parse_time_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        m = s.median()
        if m > 1e12:
            dt = pd.to_datetime(s, unit="ms", errors="coerce")
        elif m > 1e9:
            dt = pd.to_datetime(s, unit="s", errors="coerce")
        else:
            dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    else:
        dt = pd.to_datetime(s.astype(str), errors="coerce", infer_datetime_format=True)
    if ASSUME_LOCALTIME:
        if getattr(dt.dt, "tz", None) is None:
            try:
                dt = dt.dt.tz_localize(TIMEZONE, nonexistent="NaT", ambiguous="NaT")
            except Exception:
                dt = dt.dt.tz_localize(None)
        else:
            try:
                dt = dt.dt.tz_convert(TIMEZONE)
            except Exception:
                dt = dt.dt.tz_localize(None)
    return dt

def estimate_sampling_seconds(t: pd.Series) -> float:
    dt = t.sort_values().diff().dropna()
    if hasattr(dt, "dt"):
        dt = dt.dt.total_seconds()
    return float(np.median(dt)) if len(dt) else 1.0

def minutes_to_points(minutes: float, sampling_s: float) -> int:
    return max(1, int(round((minutes * 60.0) / max(0.5, sampling_s))))

def hampel_filter(x: pd.Series, window_points: int, k: float = 3.0) -> pd.Series:
    w = max(3, int(window_points))
    med = x.rolling(w, center=True, min_periods=w).median()
    mad = (x - med).abs().rolling(w, center=True, min_periods=w).median()
    sigma = 1.4826 * mad
    outliers = (x - med).abs() > (k * sigma)
    return x.where(~outliers, med).astype(float)

def mm_to_ml(delta_mm: np.ndarray | pd.Series) -> np.ndarray:
    return AREA_CM2 * (np.asarray(delta_mm, float) / 10.0)

def ml_to_g(ml: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(ml, float) * DENS_G_PER_ML

# ----------------------------- CARGA DE DATOS ---------------------------------

@st.cache_data(show_spinner=False)
def load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df

@st.cache_data(show_spinner=False)
def prepare_df(parquet_path: Path,
               roll_med_min=ROLL_MED_MINUTES_DEFAULT,
               hampel_win_min=HAMPEL_WINDOW_MIN_DEFAULT,
               hampel_k=HAMPEL_K_DEFAULT) -> Tuple[pd.DataFrame, str, float]:
    df = load_parquet(parquet_path).copy()
    cols = [c.lower() for c in df.columns]
    ren = {}
    for i, c in enumerate(df.columns):
        lc = cols[i]
        if lc == "mac" or "mac" in lc: ren[c] = "mac"
        elif lc in ("time", "timestamp", "datetime"): ren[c] = "time"
        elif lc == "vlx" or "distance" in lc: ren[c] = "vlx"
    df = df.rename(columns=ren)
    for need in ("mac", "time", "vlx"):
        if need not in df.columns:
            raise ValueError(f"Falta columna '{need}' en {parquet_path.name}")
    df["mac"] = df["mac"].astype("string").str.strip()
    df["time"] = parse_time_series(df["time"])
    df["vlx_mm"] = pd.to_numeric(df["vlx"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    df = df.drop_duplicates(subset=["time"], keep="last")
    mac_mode = df["mac"].mode().iloc[0] if not df["mac"].isna().all() else ""
    if len(df) < 3:
        raise ValueError(f"Datos insuficientes en {parquet_path.name}")
    sampling_s = estimate_sampling_seconds(df["time"])
    w_h = minutes_to_points(hampel_win_min, sampling_s)
    df["vlx_filt"] = hampel_filter(df["vlx_mm"], w_h, hampel_k)
    w_m = minutes_to_points(roll_med_min, sampling_s)
    df["vlx_smooth"] = (
        df["vlx_filt"].rolling(w_m, center=True, min_periods=w_m).median().ffill().bfill()
    )
    return df[["time", "mac", "vlx_mm", "vlx_filt", "vlx_smooth"]], mac_mode, sampling_s

def compute_flows(df: pd.DataFrame) -> pd.DataFrame:
    s = df["vlx_smooth"].ffill().bfill()
    diff_mm = s.diff()
    consumo_mm = diff_mm.clip(lower=0.0)
    refill_mm = (-diff_mm.clip(upper=0.0))
    out = df.copy()
    out["diff_mm"] = diff_mm
    out["consumo_ml"] = mm_to_ml(consumo_mm)
    out["refill_ml"] = mm_to_ml(refill_mm)
    out["consumo_g"] = ml_to_g(out["consumo_ml"])
    out["refill_g"] = ml_to_g(out["refill_ml"])
    return out

def detect_refills(df: pd.DataFrame, sampling_s: float,
                   threshold_mm: float = REFILL_JUMP_MM_DEFAULT,
                   min_sep_min: float = REFILL_MIN_SEP_MIN_DEFAULT) -> pd.DataFrame:
    d = df["vlx_smooth"].diff()
    candidates = d[d < threshold_mm]
    if candidates.empty:
        return pd.DataFrame(columns=["time", "jump_mm", "jump_ml", "jump_g"])
    min_sep_pts = minutes_to_points(min_sep_min, sampling_s)
    keep, last = [], None
    for i in candidates.index:
        if last is None or (i - last) >= min_sep_pts:
            keep.append(i)
            last = i
    jumps_mm = -d.loc[keep].values
    jumps_ml = mm_to_ml(jumps_mm)
    jumps_g = ml_to_g(jumps_ml)
    return pd.DataFrame({
        "time": df.loc[keep, "time"].values,
        "jump_mm": jumps_mm,
        "jump_ml": jumps_ml,
        "jump_g": jumps_g,
    })

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.set_index("time")
    daily = pd.DataFrame()
    daily["consumo_g"] = tmp["consumo_g"].resample("1D").sum(min_count=1)
    daily["refill_g"] = tmp["refill_g"].resample("1D").sum(min_count=1)
    daily["n_points"] = tmp["vlx_mm"].resample("1D").size()
    return daily

# ----------------------------- AMBIENTE (TS1/TS2) -----------------------------

@st.cache_data(show_spinner=False)
def _combine_ts_folder(folder: Path) -> Optional[pd.DataFrame]:
    if not folder.exists():
        return None
    files = [p for p in folder.iterdir() if p.suffix.lower() == ".csv"]
    if not files:
        return None
    rows = []
    for fp in files:
        try:
            df = pd.read_csv(fp, sep=";")
        except Exception:
            df = pd.read_csv(fp)
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    return df

@st.cache_data(show_spinner=False)
def load_env_combined(ls_label: str) -> Optional[pd.DataFrame]:
    fp = TS_COMBINED[ls_label]
    if fp.exists():
        try:
            df = pd.read_csv(fp, sep=";")
        except Exception:
            df = pd.read_csv(fp)
    else:
        folder = TS_FOLDERS[ls_label]
        df = _combine_ts_folder(folder)
        if df is None:
            return None
    df.columns = [c.strip().lower() for c in df.columns]
    ren = {}
    for c in df.columns:
        if "mac" in c: ren[c] = "mac"
        elif c in ("time", "timestamp", "datetime"): ren[c] = "time"
        elif "moist" in c: ren[c] = "moisture"
        elif "temp" in c: ren[c] = "temperature"
    df = df.rename(columns=ren)
    if "time" not in df.columns:
        return None
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in ("moisture", "temperature"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ls"] = ls_label
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df[["time", "ls"] + [c for c in ("temperature", "moisture") if c in df.columns]]

def env_hourly(ts: pd.DataFrame) -> pd.DataFrame:
    env = (
        ts.set_index("time")
          .groupby("ls")
          .resample("1H")
          .agg(temperature=("temperature", "mean"),
               moisture=("moisture", "mean"))
          .reset_index()
    )
    env["datehour"] = env["time"].dt.floor("H")
    return env.drop(columns=["time"])

# ----------------------------- SESIONES Y EMPAREJADOS -------------------------

def sessions_from_flow(df: pd.DataFrame) -> pd.DataFrame:
    x = df.set_index("time")["consumo_g"].resample("1min").sum(min_count=1).fillna(0)
    active = x.rolling(10, min_periods=1).sum() > 0
    blocks = (active != active.shift()).cumsum()
    rows = []
    for k, g in x.groupby(blocks):
        if not active[g.index[0]]:
            continue
        dur = len(g)
        grams = float(g.sum())
        rows.append({
            "start": g.index[0],
            "end": g.index[-1],
            "duration_min": int(dur),
            "grams": grams,
            "rate_g_per_min": grams / max(dur, 1),
        })
    return pd.DataFrame(rows)

def pair_refills(ref1: pd.DataFrame, ref5: pd.DataFrame, window="30min") -> pd.DataFrame:
    if ref1.empty or ref5.empty:
        return pd.DataFrame(columns=["t_ls1","jump_g_ls1","t_ls5","jump_g_ls5","delay_s","ratio_g"])
    r1 = ref1.sort_values("time").reset_index(drop=True)
    r5 = ref5.sort_values("time").reset_index(drop=True)
    out = []
    for _, a in r1.iterrows():
        t1 = pd.to_datetime(a["time"])
        lo, hi = t1 - pd.Timedelta(window), t1 + pd.Timedelta(window)
        cand = r5[(r5["time"] >= lo) & (r5["time"] <= hi)]
        if cand.empty:
            continue
        k = (cand["time"] - t1).abs().idxmin()
        b = cand.loc[k]
        delay = (b["time"] - t1).total_seconds()
        ratio = float(b["jump_g"]) / float(a["jump_g"]) if float(a["jump_g"]) != 0 else np.nan
        out.append({
            "t_ls1": t1, "jump_g_ls1": float(a["jump_g"]),
            "t_ls5": pd.to_datetime(b["time"]), "jump_g_ls5": float(b["jump_g"]),
            "delay_s": float(delay), "ratio_g": float(ratio),
        })
    return pd.DataFrame(out)

# ----------------------------- CHART HELPERS (ECharts) -------------------------

def toolbox_default():
    return {
        "feature": {
            "saveAsImage": {},
            "dataZoom": {"yAxisIndex": "none"},
            "restore": {},
        }
    }

def echarts_line_time(series_list, title="", ylabel="", height=360, stacked=False):
    opt = {
        "title": {"text": title, "textStyle": {"color":"#e6eef8", "fontSize":14}},
        "tooltip": {"trigger": "axis"},
        "toolbox": toolbox_default(),
        "xAxis": {"type": "time", "axisLine": {"lineStyle": {"color": "#6b7280"}}},
        "yAxis": {"type": "value", "name": ylabel, "axisLine": {"lineStyle": {"color": "#6b7280"}}},
        "series": [],
        "grid": {"left": 70, "right": 30, "top": 60, "bottom": 60},
        "dataZoom": [{"type": "inside"}, {"type": "slider"}],
        "legend": {"top": 10, "textStyle": {"color":"#cbd5e1"}},
    }
    for s in series_list:
        z = {"type": "line", "symbol": "none", "smooth": True, "areaStyle": {"opacity": 0.08}}
        if stacked:
            z["stack"] = "total"
        z.update(s)
        opt["series"].append(z)
    st_echarts(opt, height=height, key=title)

def echarts_bar(categories, series_dict, title="", ylabel="", height=360, horizontal=False):
    x = {"type": "category", "data": categories, "axisLabel": {"rotate": 45}}
    y = {"type": "value", "name": ylabel}
    opt = {
        "title": {"text": title, "textStyle": {"color":"#e6eef8", "fontSize":14}},
        "tooltip": {"trigger": "axis"},
        "toolbox": toolbox_default(),
        "xAxis": y if horizontal else x,
        "yAxis": x if horizontal else y,
        "series": [],
        "grid": {"left": 70, "right": 30, "top": 60, "bottom": 140},
        "legend": {"top": 10, "textStyle": {"color":"#cbd5e1"}},
        "dataZoom": [{"type": "inside"}, {"type": "slider"}],
    }
    for name, values in series_dict.items():
        opt["series"].append({
            "name": name,
            "type": "bar",
            "data": values,
            "barMaxWidth": 28,
        })
    st_echarts(opt, height=height, key=title)

def echarts_scatter(x, y, name="", title="", xlabel="", ylabel="", height=360, line_fit: Optional[Tuple[float,float]]=None):
    data = [[float(x[i]), float(y[i])] for i in range(len(x))]
    opt = {
        "title": {"text": title, "textStyle": {"color":"#e6eef8", "fontSize":14}},
        "tooltip": {"trigger": "item"},
        "toolbox": toolbox_default(),
        "xAxis": {"type": "value", "name": xlabel},
        "yAxis": {"type": "value", "name": ylabel},
        "series": [{"name": name, "type": "scatter", "data": data, "symbolSize": 8}],
        "grid": {"left": 70, "right": 30, "top": 60, "bottom": 60},
        "legend": {"top": 10, "textStyle": {"color":"#cbd5e1"}},
    }
    if line_fit is not None:
        m, b = line_fit
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
        ys = m * xs + b
        opt["series"].append({
            "name": "OLS",
            "type": "line",
            "data": [[float(xs[i]), float(ys[i])] for i in range(len(xs))],
            "symbol": "none",
            "lineStyle": {"color": "#ffd166"}
        })
    st_echarts(opt, height=height, key=title)

def echarts_hist(values, bins=30, title="", xlabel="", ylabel="Frecuencia", height=320):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        st.info("Sin datos para histograma.")
        return
    hist, edges = np.histogram(v, bins=bins)
    cats = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(edges)-1)]
    echarts_bar(cats, {"Conteo": hist.tolist()}, title=title, ylabel=ylabel, height=height)

# ----------------------------- LAYOUT -----------------------------------------

st.set_page_config(page_title="LS1 vs LS5 — Dashboard", layout="wide", initial_sidebar_state="collapsed")
inject_css()

# Top header
with st.container():
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap" rel="stylesheet">
        <div style="
            margin: 12px 0 22px 0;
            text-align: center;">
            <div style="
                font-family: 'Montserrat', sans-serif;
                font-size: 28px;
                font-weight: 800;
                color: #94a3b8;
                letter-spacing: 0.6px;">
                Comparativa LS1 — LS5
            </div>
            <div style="
                font-family: 'Montserrat', sans-serif;
                font-size: 15px;
                font-weight: 600;
                color: #94a3b8;
                margin-top: 6px;">
                Consumo · Refills · Análisis ambiental
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )



# Sidebar minimal con controles agrupados
with st.sidebar:
    st.header("Controles")
    roll_med = st.number_input("Mediana rodante (min)", min_value=1.0, max_value=60.0,
                              value=ROLL_MED_MINUTES_DEFAULT, step=0.5)
    hampel_win = st.number_input("Ventana Hampel (min)", min_value=1.0, max_value=180.0,
                                value=HAMPEL_WINDOW_MIN_DEFAULT, step=0.5)
    hampel_k = st.number_input("Hampel k", min_value=1.0, max_value=10.0,
                               value=HAMPEL_K_DEFAULT, step=0.5)
    refill_thr = st.number_input("Umbral refill (mm, negativo)", min_value=-1000.0, max_value=-1.0,
                                 value=REFILL_JUMP_MM_DEFAULT, step=1.0)
    refill_sep = st.number_input("Separación mínima entre refills (min)", min_value=1.0, max_value=1440.0,
                                 value=REFILL_MIN_SEP_MIN_DEFAULT, step=1.0)
    excl_refill = st.checkbox("Excluir ±1 h alrededor de refills", value=True)
    show_ls1 = st.checkbox("Mostrar LS1", value=True)
    show_ls5 = st.checkbox("Mostrar LS5", value=True)
    bins_hist = st.slider("Bins histograma", 10, 80, 30)
    st.markdown("---")
    st.markdown("Rutas de datos relativas al fichero. Si falta algún archivo, la app mostrará un error claro.")

# Carga y verificación de ficheros
p_ls1 = IN_DIR / "ls1.parquet"
p_ls5 = IN_DIR / "ls5.parquet"

err_paths = []
if not p_ls1.exists(): err_paths.append(str(p_ls1))
if not p_ls5.exists(): err_paths.append(str(p_ls5))
if err_paths:
    st.error("Faltan ficheros parquet:\n" + "\n".join(err_paths))
    st.stop()

df1_raw, mac1, samp1 = prepare_df(p_ls1, roll_med, hampel_win, hampel_k)
df5_raw, mac5, samp5 = prepare_df(p_ls5, roll_med, hampel_win, hampel_k)
df1 = compute_flows(df1_raw)
df5 = compute_flows(df5_raw)
ref1 = detect_refills(df1, samp1, refill_thr, refill_sep)
ref5 = detect_refills(df5, samp5, refill_thr, refill_sep)

t_min = min(df1["time"].min(), df5["time"].min())
t_max = max(df1["time"].max(), df5["time"].max())

# Date range selector prominent
dr = st.date_input("Rango de fechas", value=(t_min.date(), t_max.date()),
                   min_value=t_min.date(), max_value=t_max.date())
start_dt = pd.Timestamp(dr[0])
end_dt = pd.Timestamp(dr[1]) + pd.Timedelta(days=1)

# Forzar naive en tiempo para operaciones locales
for df in (df1, df5, ref1, ref5):
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)

mask1 = (df1["time"] >= start_dt) & (df1["time"] < end_dt)
mask5 = (df5["time"] >= start_dt) & (df5["time"] < end_dt)
df1 = df1[mask1].copy()
df5 = df5[mask5].copy()
ref1v = ref1[(ref1["time"] >= start_dt) & (ref1["time"] < end_dt)].copy()
ref5v = ref5[(ref5["time"] >= start_dt) & (ref5["time"] < end_dt)].copy()

# KPI summary helper
def resumen(nombre: str, df: pd.DataFrame, ref: pd.DataFrame) -> Dict[str, float]:
    total_g = float(df["consumo_g"].sum())
    total_ref_g = float(df["refill_g"].sum())
    days = df.set_index("time")["consumo_g"].resample("1D").sum(min_count=1)
    media_dia = float(days.mean()) if len(days) else np.nan
    n_ref = int(len(ref))
    avg_ref = float(ref["jump_g"].mean()) if n_ref else 0.0
    return dict(total_g=total_g, total_ref_g=total_ref_g, media_dia=media_dia,
                n_ref=n_ref, avg_ref=avg_ref)

k1 = resumen("LS1", df1, ref1v)
k5 = resumen("LS5", df5, ref5v)

# KPIs display minimalista y claro
st.markdown('<div class="kpi-row">', unsafe_allow_html=True)
def kpi_card(title, v, sub):
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-title">{title}</div>
      <div class="kpi-value">{v}</div>
      <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

col_k1, col_k2, col_k3, col_k4 = st.columns(4)
with col_k1:
    kpi_card("LS1 — Consumo total (g)", f"{k1['total_g']:,.0f}", f"Media/día {k1['media_dia']:,.0f}")
with col_k2:
    kpi_card("LS5 — Consumo total (g)", f"{k5['total_g']:,.0f}", f"Media/día {k5['media_dia']:,.0f}")
with col_k3:
    kpi_card("LS1 — Refills", f"{k1['n_ref']}", f"Media {k1['avg_ref']:,.0f} g")
with col_k4:
    kpi_card("LS5 — Refills", f"{k5['n_ref']}", f"Media {k5['avg_ref']:,.0f} g")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="small-muted">Previsualización datos: primeras filas (LS1 / LS5)</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.write("LS1 — muestra")
    st.dataframe(df1.head(6).reset_index(drop=True), use_container_width=True)
with c2:
    st.write("LS5 — muestra")
    st.dataframe(df5.head(6).reset_index(drop=True), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# --------------------- 1) Nivel (vlx_smooth) con refills ----------------------
series = []
if show_ls1 and not df1.empty:
    s1 = [[pd.to_datetime(t).isoformat(), float(v)] for t, v in zip(df1["time"], df1["vlx_smooth"])]
    series.append({"name": "LS1 nivel (mm)", "data": s1, "itemStyle": {"color": "#60a5fa"}})
if show_ls5 and not df5.empty:
    s5 = [[pd.to_datetime(t).isoformat(), float(v)] for t, v in zip(df5["time"], df5["vlx_smooth"])]
    series.append({"name": "LS5 nivel (mm)", "data": s5, "itemStyle": {"color": "#34d399"}})
if series:
    echarts_line_time(series, title="Nivel suavizado (mm) — LS1 vs LS5", ylabel="mm", height=360)

def mark_refills_to_series(ref_df, name, color="#ffd166"):
    return {
        "name": name,
        "type": "line",
        "data": [[pd.to_datetime(x).isoformat(), None] for x in ref_df["time"]],
        "markLine": {
            "symbol": "none",
            "data": [{"xAxis": pd.to_datetime(x).isoformat()} for x in ref_df["time"]],
            "lineStyle": {"type": "dashed", "opacity": 0.6, "color": color},
        },
    }

mrk = []
if show_ls1 and not ref1v.empty:
    mrk.append(mark_refills_to_series(ref1v, "Refills LS1", color="#60a5fa"))
if show_ls5 and not ref5v.empty:
    mrk.append(mark_refills_to_series(ref5v, "Refills LS5", color="#34d399"))
if mrk:
    echarts_line_time(mrk, title="Refills — marcadores temporales", ylabel="", height=140)

# --------------------- 2) Consumo acumulado ----------------------------------
series = []
if show_ls1 and not df1.empty:
    csum = df1["consumo_g"].cumsum()
    series.append({"name": "LS1 Consumo acumulado (g)",
                   "data": [[pd.to_datetime(t).isoformat(), float(v)] for t, v in zip(df1["time"], csum)],
                   "itemStyle": {"color": "#60a5fa"}})
if show_ls5 and not df5.empty:
    csum = df5["consumo_g"].cumsum()
    series.append({"name": "LS5 Consumo acumulado (g)",
                   "data": [[pd.to_datetime(t).isoformat(), float(v)] for t, v in zip(df5["time"], csum)],
                   "itemStyle": {"color": "#34d399"}})
if series:
    echarts_line_time(series, title="Consumo acumulado — LS1 vs LS5", ylabel="g", height=300)

# --------------------- 3) Consumo diario (barras) ----------------------------
d1 = aggregate_daily(df1) if not df1.empty else pd.DataFrame()
d5 = aggregate_daily(df5) if not df5.empty else pd.DataFrame()
idx = sorted(set((d1.index.date.tolist() if not d1.empty else []) + (d5.index.date.tolist() if not d5.empty else [])))
cats = [str(d) for d in idx]
s_ls1 = [float(d1["consumo_g"].get(pd.Timestamp(i), np.nan)) if (not d1.empty and pd.Timestamp(i) in d1.index) else np.nan for i in idx]
s_ls5 = [float(d5["consumo_g"].get(pd.Timestamp(i), np.nan)) if (not d5.empty and pd.Timestamp(i) in d5.index) else np.nan for i in idx]
series_dict = {}
if show_ls1: series_dict["LS1"] = s_ls1
if show_ls5: series_dict["LS5"] = s_ls5
if series_dict:
    echarts_bar(cats, series_dict, title="Consumo diario (g) — LS1 vs LS5", ylabel="g", height=340)

# --------------------- 4) Recarga diaria (barras) ----------------------------
s_ls1_ref = [float(d1["refill_g"].get(pd.Timestamp(i), np.nan)) if (not d1.empty and pd.Timestamp(i) in d1.index) else np.nan for i in idx]
s_ls5_ref = [float(d5["refill_g"].get(pd.Timestamp(i), np.nan)) if (not d5.empty and pd.Timestamp(i) in d5.index) else np.nan for i in idx]
series_dict2 = {}
if show_ls1: series_dict2["LS1 Recarga"] = s_ls1_ref
if show_ls5: series_dict2["LS5 Recarga"] = s_ls5_ref
if series_dict2:
    echarts_bar(cats, series_dict2, title="Recarga diaria (g)", ylabel="g", height=300)

# --------------------- 5) Sesiones de consumo --------------------------------
sess1 = sessions_from_flow(df1) if not df1.empty else pd.DataFrame()
sess5 = sessions_from_flow(df5) if not df5.empty else pd.DataFrame()

cA, cB = st.columns(2)
with cA:
    echarts_hist(sess1["rate_g_per_min"] if not sess1.empty else [], bins=bins_hist,
                 title="LS1 — Tasa de consumo (g/min)", xlabel="g/min")
with cB:
    echarts_hist(sess5["rate_g_per_min"] if not sess5.empty else [], bins=bins_hist,
                 title="LS5 — Tasa de consumo (g/min)", xlabel="g/min")

# --------------------- 6) Ambiente y consumo horario -------------------------
ts1 = load_env_combined("LS1")
ts5 = load_env_combined("LS5")
if ts1 is not None and ts5 is not None:
    ts = pd.concat([ts1, ts5], ignore_index=True)
    env_h = env_hourly(ts)

    st.markdown("#### Análisis ambiental y consumo horario")

    ts_daily_1 = ts1.set_index('time')[['temperature', 'moisture']].resample('D').mean() if not ts1.empty else pd.DataFrame()
    ts_daily_5 = ts5.set_index('time')[['temperature', 'moisture']].resample('D').mean() if not ts5.empty else pd.DataFrame()

    series_temp = []
    if show_ls1 and 'temperature' in ts_daily_1:
        data_temp_1 = [[d.isoformat(), v] for d, v in ts_daily_1['temperature'].dropna().items()]
        series_temp.append({"name": "LS1 Temperatura (°C)", "data": data_temp_1, "itemStyle": {"color": "#60a5fa"}})
    if show_ls5 and 'temperature' in ts_daily_5:
        data_temp_5 = [[d.isoformat(), v] for d, v in ts_daily_5['temperature'].dropna().items()]
        series_temp.append({"name": "LS5 Temperatura (°C)", "data": data_temp_5, "itemStyle": {"color": "#34d399"}})
    if series_temp:
        echarts_line_time(series_temp, title="Temperatura Media Diaria", ylabel="°C", height=300)

    series_moist = []
    if show_ls1 and 'moisture' in ts_daily_1:
        data_moist_1 = [[d.isoformat(), v] for d, v in ts_daily_1['moisture'].dropna().items()]
        series_moist.append({"name": "LS1 Humedad (%)", "data": data_moist_1, "itemStyle": {"color": "#7dd3fc"}})
    if show_ls5 and 'moisture' in ts_daily_5:
        data_moist_5 = [[d.isoformat(), v] for d, v in ts_daily_5['moisture'].dropna().items()]
        series_moist.append({"name": "LS5 Humedad (%)", "data": data_moist_5, "itemStyle": {"color": "#facc15"}})
    if series_moist:
        echarts_line_time(series_moist, title="Humedad Media Diaria", ylabel="%", height=300)

    def hourly_from_sessions(sess: pd.DataFrame, label: str) -> pd.DataFrame:
        if sess.empty:
            return pd.DataFrame(columns=["ls", "datehour", "grams_hour"])
        s = sess.copy()
        s["midpoint"] = s["start"] + (s["end"] - s["start"]) / 2
        s["datehour"] = s["midpoint"].dt.floor("H")
        g = s.groupby("datehour", as_index=False)["grams"].sum()
        g["ls"] = label
        g = g.rename(columns={"grams": "grams_hour"})
        return g[["ls", "datehour", "grams_hour"]]

    gh1 = hourly_from_sessions(sess1, "LS1")
    gh5 = hourly_from_sessions(sess5, "LS5")
    hourly = pd.concat([gh1, gh5], ignore_index=True)

    if excl_refill:
        def flag_near_refill(hh: pd.DataFrame, ref: pd.DataFrame, label: str) -> pd.DataFrame:
            if ref.empty or hh.empty:
                hh["near_refill"] = False
                return hh
            m = pd.merge_asof(hh.sort_values("datehour"),
                              ref[["time"]].sort_values("time"),
                              left_on="datehour", right_on="time",
                              direction="nearest", tolerance=pd.Timedelta("1H"))
            hh["near_refill"] = m["time"].notna()
            return hh
        gh1 = flag_near_refill(gh1, ref1v, "LS1")
        gh5 = flag_near_refill(gh5, ref5v, "LS5")
        hourly = pd.concat([gh1[~gh1["near_refill"]], gh5[~gh5["near_refill"]]], ignore_index=True)

    hourly_env = hourly.merge(env_h, on=["ls", "datehour"], how="left")

    series = []
    if show_ls1:
        sub = hourly_env[hourly_env["ls"] == "LS1"].sort_values("datehour")
        series.append({"name": "LS1 g/h",
                      "data": [[pd.to_datetime(t).isoformat(), float(v)] for t, v in zip(sub["datehour"], sub["grams_hour"].fillna(0.0))],
                      "itemStyle": {"color": "#60a5fa"}})
    if show_ls5:
        sub = hourly_env[hourly_env["ls"] == "LS5"].sort_values("datehour")
        series.append({"name": "LS5 g/h",
                      "data": [[pd.to_datetime(t).isoformat(), float(v)] for t, v in zip(sub["datehour"], sub["grams_hour"].fillna(0.0))],
                      "itemStyle": {"color": "#34d399"}})
    if series:
        echarts_line_time(series, title="Consumo horario (g/h)", ylabel="g/h", height=300)

    for var, label in (("temperature", "°C"), ("moisture", "%")):
        for ls in ("LS1", "LS5"):
            sub = hourly_env[(hourly_env["ls"] == ls) & hourly_env[var].notna() & hourly_env["grams_hour"].notna()]
            if sub.empty:
                continue
            x = sub[var].values.astype(float)
            y = sub["grams_hour"].values.astype(float)
            if len(x) >= 2:
                m, b = np.polyfit(x, y, 1)
                line_fit = (float(m), float(b))
            else:
                line_fit = None
            echarts_scatter(
                x.tolist(), y.tolist(), name=f"{ls}",
                title=f"{ls} — g/h vs {var}",
                xlabel=label, ylabel="g/h", line_fit=line_fit, height=320
            )

    def fe_ols(df: pd.DataFrame, label: str) -> Optional[pd.DataFrame]:
        sub = df[df["ls"] == label].dropna(subset=["grams_hour", "temperature", "moisture", "datehour"]).copy()
        if sub.empty or len(sub) < 30:
            return None
        sub["hod"] = pd.to_datetime(sub["datehour"]).dt.hour.astype("int16")
        sub["date"] = pd.to_datetime(sub["datehour"]).dt.date.astype("object")
        fit = smf.ols("grams_hour ~ temperature + moisture + C(hod) + C(date)", data=sub)\
                 .fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        return pd.DataFrame({
            "n": [int(fit.nobs)],
            "r2_adj": [float(fit.rsquared_adj)],
            "beta_temp": [float(fit.params.get("temperature", np.nan))],
            "p_temp": [float(fit.pvalues.get("temperature", np.nan))],
            "beta_moist": [float(fit.params.get("moisture", np.nan))],
            "p_moist": [float(fit.pvalues.get("moisture", np.nan))],
        })

    c1, c2 = st.columns(2)
    with c1:
        if (tbl := fe_ols(hourly_env, "LS1")) is not None:
            st.markdown("**LS1 — OLS FE(hora,día) con HAC**")
            st.dataframe(tbl.round(4), use_container_width=True)
    with c2:
        if (tbl := fe_ols(hourly_env, "LS5")) is not None:
            st.markdown("**LS5 — OLS FE(hora,día) con HAC**")
            st.dataframe(tbl.round(4), use_container_width=True)

    # Diferencias Horarias (LS1 - LS5)
    st.markdown("#### Diferencias Horarias (LS1 - LS5)")
    hourly_pivot = hourly_env.pivot_table(
        index='datehour',
        columns='ls',
        values=['grams_hour', 'temperature', 'moisture']
    ).dropna()
    hourly_pivot.columns = ['_'.join(col).strip() for col in hourly_pivot.columns.values]
    if not hourly_pivot.empty:
        hourly_pivot['grams_diff'] = hourly_pivot['grams_hour_LS1'] - hourly_pivot['grams_hour_LS5']
        hourly_pivot['temp_diff'] = hourly_pivot.get('temperature_LS1', 0) - hourly_pivot.get('temperature_LS5', 0)
        diff_data = [[r.name.isoformat(), r['grams_diff']] for _, r in hourly_pivot.iterrows()]
        series_diff = [{"name": "Diferencia Consumo (g/h)", "data": diff_data, "itemStyle": {"color": "#5cb85c"}}]
        echarts_line_time(series_diff, title="Diferencia de Consumo Horario (LS1 - LS5)", ylabel="g/h", height=300)
    else:
        st.info("No hay suficientes datos horarios solapados para calcular las diferencias.")

# --------------------- 7) Contraste de medias diarias -------------------------
a = d1["consumo_g"].dropna().values if not d1.empty else np.array([])
b = d5["consumo_g"].dropna().values if not d5.empty else np.array([])
if len(a) > 1 and len(b) > 1:
    u_stat, u_p = mannwhitneyu(a, b, alternative="two-sided")
    t_stat, t_p = ttest_ind(a, b, equal_var=False, nan_policy="omit")
    st.markdown("**Contraste de consumo diario LS1 vs LS5**")
    st.write({"Mann–Whitney_U": float(u_stat), "p_U": float(u_p),
              "Welch_t": float(t_stat), "p_t": float(t_p)})

# --------------------- 8) Emparejado de refills y latencias -------------------
pairs = pair_refills(ref1v, ref5v, window="30min")
if not pairs.empty:
    echarts_hist(pairs["delay_s"], bins=bins_hist, title="Delay LS1→LS5 en refills (s)", xlabel="s")
    echarts_hist(pairs["ratio_g"], bins=bins_hist, title="Ratio magnitud (LS5/LS1) en refills", xlabel="ratio")

# --------------------- 9) Correlación Cruzada (CCF) --------------------------
st.markdown("### Correlación Cruzada (CCF) del Nivel")
s1 = df1.set_index("time")["vlx_smooth"].resample("1min").median().interpolate() if not df1.empty else pd.Series(dtype=float)
s5 = df5.set_index("time")["vlx_smooth"].resample("1min").median().interpolate() if not df5.empty else pd.Series(dtype=float)
s_join = pd.concat([s1, s5], axis=1, keys=['LS1', 'LS5']).dropna()

if len(s_join) > 30:
    x = s_join['LS1'] - s_join['LS1'].mean()
    y = s_join['LS5'] - s_join['LS5'].mean()
    max_lags_minutes = 120
    x_arr = x.to_numpy()
    y_arr = y.to_numpy()
    correlation = np.correlate(x_arr - np.mean(x_arr), y_arr - np.mean(y_arr), mode='full')
    n = len(x_arr)
    lags = np.arange(-(n - 1), n)
    mask = (lags >= -max_lags_minutes) & (lags <= max_lags_minutes)
    lags_filtered = lags[mask]
    correlation_filtered = correlation[mask]
    std_x = np.std(x_arr)
    std_y = np.std(y_arr)
    if std_x > 0 and std_y > 0:
        ccf_values = correlation_filtered / (n * std_x * std_y)
    else:
        ccf_values = np.zeros_like(correlation_filtered)
    lag_star_idx = np.argmax(ccf_values)
    lag_star = int(lags_filtered[lag_star_idx])
    ccf_data = [[int(lag), float(corr)] for lag, corr in zip(lags_filtered, ccf_values)]
    lag_interpretation = "LS5 retrasado respecto a LS1" if lag_star > 0 else "LS1 retrasado respecto a LS5" if lag_star < 0 else "Sin desfase dominante"
    title_text = f"CCF LS1 vs LS5 — Lag* = {lag_star} min ({lag_interpretation})"
    opt_ccf = {
        "title": {"text": title_text},
        "tooltip": {"trigger": "axis"},
        "toolbox": toolbox_default(),
        "xAxis": {
            "type": "value",
            "name": "Lag (minutos)",
            "min": -max_lags_minutes,
            "max": max_lags_minutes
        },
        "yAxis": {"type": "value", "name": "Correlación"},
        "series": [{
            "name": "CCF",
            "type": "line",
            "data": ccf_data,
            "symbol": "none",
            "markLine": {
                "data": [{"xAxis": lag_star, "label": {"formatter": f"Pico en {lag_star} min"}}],
                "lineStyle": {"color": "#d9534f"}
            }

        }],
        "grid": {"left": 70, "right": 40, "top": 60, "bottom": 50},
    }
    st_echarts(opt_ccf, height=340, key="ccf_chart")
else:
    st.info("No hay suficientes datos solapados para calcular la Correlación Cruzada.")

# --------------------- 10) Descargas -----------------------------------------
@st.cache_data(show_spinner=False)
def make_kpis_table() -> pd.DataFrame:
    return pd.DataFrame({
        "feeder": ["LS1", "LS5"],
        "mac": [mac1, mac5],
        "sampling_s_median": [samp1, samp5],
        "total_consumo_g": [float(df1["consumo_g"].sum()), float(df5["consumo_g"].sum())],
        "total_refill_g": [float(df1["refill_g"].sum()), float(df5["refill_g"].sum())],
        "n_refill_events": [int(len(ref1v)), int(len(ref5v))],
        "avg_refill_g": [
            float(ref1v["jump_g"].mean()) if len(ref1v) else 0.0,
            float(ref5v["jump_g"].mean()) if len(ref5v) else 0.0,
        ],
    })

st.markdown("### Descargas")
kpis_df = make_kpis_table()
c1, c2, c3 = st.columns(3)
with c1:
    st.download_button("Descargar KPIs (CSV)", data=kpis_df.to_csv(index=False).encode("utf-8"),
                      file_name="kpis_ls1_ls5.csv", mime="text/csv")
with c2:
    joined = pd.DataFrame({"date": [str(d) for d in idx],
                           "LS1_consumo_g": s_ls1,
                           "LS5_consumo_g": s_ls5})
    st.download_button("Consumo diario (CSV)", data=joined.to_csv(index=False).encode("utf-8"),
                      file_name="consumo_diario_g_ls1_vs_ls5.csv", mime="text/csv")
with c3:
    ref_join = pd.concat([ref1v.assign(ls="LS1"), ref5v.assign(ls="LS5")], ignore_index=True)
    st.download_button("Refills detectados (CSV)", data=ref_join.to_csv(index=False).encode("utf-8"),
                      file_name="refill_events.csv", mime="text/csv")

st.markdown("---")
st.caption("Periodo filtrado por el selector de fechas. TZ Europe/Madrid. Gráficos generados con ECharts.")
