# streamlit_app.py  (v3.1: no replicate overlay)
# - Derives Plate and Well from Sample_ID (e.g., "py3B1" -> plate="py3", well="B1")
# - Sorting: Mean ↑ / Mean ↓
# - Single plot: mean bars with SD error bars
# - Metric toggle: Indel% vs %OOF

import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="CRISPR Indel/%OOF Explorer", layout="wide")
st.title("CRISPR Indel / %OOF Explorer")
st.caption(
    "Replicates are grouped by identical **Description**. "
    "Plate and Well are parsed from **Sample_ID** (e.g., `py3B1` → plate `py3`, well `B1`)."
)

# ---------- Sample dataset ----------
def load_sample() -> pd.DataFrame:
    return pd.DataFrame({
        "Sample_ID": ["py3A1","py3A2","py3A3","py3B1","py3B2","py3B3","py4A1","py4A2","py4A3","py4B1","py4B2","py4B3"],
        "Description": ["Cas9_g1"]*3 + ["Cas9_g2"]*3 + ["Cas12_g7"]*3 + ["Cas12_g8"]*3,
        "Indel%": [45.2, 47.1, 46.5, 10.1, 12.3, 9.8, 33.2, 32.5, 35.0, 55.0, 54.2, 53.5],
        "%OOF":   [ 5.1,  4.8,  5.3,  0.9,  1.2, 1.0,  3.2,  3.5,  3.1,  8.0,  7.6,  7.9],
    })

with st.sidebar:
    st.header("1) Load data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    use_sample = st.checkbox("Use sample dataset", value=not uploaded)
    st.markdown("---")
    st.header("2) Options")

@st.cache_data
def read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def infer_columns(df: pd.DataFrame):
    cols = list(df.columns)

    desc_col = next((c for c in cols if re.search(r"^desc(ription)?$|description", c, flags=re.I)), None)
    sid_col  = next((c for c in cols if re.search(r"^sample[_ ]?id$|sampleid|sample[-_ ]?name", c, flags=re.I)), None)

    indel_col = next((c for c in cols if re.search(r"\bindel\b", c, flags=re.I)), None)
    oof_col   = next((c for c in cols if re.search(r"\boof\b|out[-_ ]?of[-_ ]?frame", c, flags=re.I)), None)
    if indel_col is None:
        indel_col = next((c for c in cols if re.search(r"indel.*%|%.*indel|indel[_ ]?percent", c, flags=re.I)), None)
    if oof_col is None:
        oof_col = next((c for c in cols if re.search(r"oof.*%|%.*oof|oof[_ ]?percent", c, flags=re.I)), None)

    return {"desc": desc_col, "sid": sid_col, "indel": indel_col, "oof": oof_col}

def clean_metric_column(df, col):
    def to_float(x):
        if pd.isna(x): return np.nan
        if isinstance(x, str):
            m = re.search(r"(-?\d+(\.\d+)?)", x)
            return float(m.group(1)) if m else np.nan
        return float(x)
    if col and col in df.columns:
        df[col] = df[col].apply(to_float)
    return df

def parse_plate_well(sample_id: str):
    """
    Accepts patterns like 'py3B1' or 'py3_B01'.
    Returns (plate, well). If cannot parse, plate='Unknown', well='NA'.
    """
    if sample_id is None or (isinstance(sample_id, float) and np.isnan(sample_id)):
        return "Unknown", "NA"
    s = str(sample_id).strip()
    m = re.match(r"^(?P<plate>.*?)[ _-]?(?P<well>[A-Ha-h](?:0?[1-9]|1[0-2]))$", s)
    if not m:
        return "Unknown", s
    plate = m.group("plate")
    well  = m.group("well").upper()
    well = well[0] + str(int(well[1:]))  # normalize B01->B1
    return plate, well

# ---------- Load ----------
if uploaded:
    df = read_csv(uploaded)
elif use_sample:
    df = load_sample()
    st.info("Using a built-in sample dataset. Uncheck to upload your CSV.")
else:
    st.stop()

info = infer_columns(df)

with st.expander("Detected columns / change if needed"):
    c1, c2 = st.columns(2)
    with c1:
        desc_col  = st.selectbox("Description column", options=df.columns, index=(df.columns.get_loc(info["desc"]) if info["desc"] in df.columns else 0))
        sid_col   = st.selectbox("Sample_ID column",  options=df.columns, index=(df.columns.get_loc(info["sid"])  if info["sid"]  in df.columns else 0))
    with c2:
        indel_col = st.selectbox("Indel% column",     options=df.columns, index=(df.columns.get_loc(info["indel"]) if info["indel"] in df.columns else 0))
        oof_col   = st.selectbox("%OOF column",       options=[None] + list(df.columns), index=(1 + df.columns.get_loc(info["oof"]) if info["oof"] in df.columns else 0))

df = clean_metric_column(df, indel_col)
if oof_col:
    df = clean_metric_column(df, oof_col)

# Parse plate + well from Sample_ID
plates, wells = zip(*df[sid_col].map(parse_plate_well))
df["_Plate"] = list(plates)
df["_Well"]  = list(wells)

# ---------- Sidebar ----------
st.sidebar.subheader("Filters")
available_plates = sorted(pd.Series(df["_Plate"].astype(str)).unique().tolist())
selected_plates = st.sidebar.multiselect("Plate(s)", options=available_plates, default=available_plates)

metric = st.sidebar.radio("Metric", options=["Indel%", "%OOF"], index=0)
metric_col = indel_col if metric == "Indel%" else oof_col
if metric_col is None or metric_col not in df.columns:
    st.error(f"Metric column for '{metric}' not found. Please fix in 'Detected columns'.")
    st.stop()

st.sidebar.subheader("Sort")
sort_by = st.sidebar.radio("Sort by mean", options=["Mean ↑", "Mean ↓"], index=1)
ascending = (sort_by == "Mean ↑")

# ---------- Filter / aggregate ----------
fdf = df[df["_Plate"].astype(str).isin(selected_plates)].copy()

agg = (
    fdf.groupby(desc_col, dropna=False)[metric_col]
       .agg(mean="mean", std="std", count="count")
       .reset_index()
    .sort_values(by=["mean", desc_col], ascending=[ascending, True])
)

# Keep chart x-order consistent
fdf[desc_col] = pd.Categorical(fdf[desc_col], categories=agg[desc_col].tolist(), ordered=True)

# ---------- Plot: mean bars with SD ----------
fig = px.bar(
    agg,
    x=desc_col,
    y="mean",
    error_y="std",
    hover_data={"mean":":.2f", "std":":.2f", "count":True},
)
fig.update_layout(
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis_title="Description",
    yaxis_title=f"Mean {metric} (± SD)",
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption(
    "Notes: Plate and well parsed from Sample_ID. Sorting is by group mean only. "
    "Bars show group mean with SD error bars."
)
