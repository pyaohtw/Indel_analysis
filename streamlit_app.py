# streamlit_app.py  (v7.3)
# - Filters: Select all/Clear with hierarchy clears (Plate > Amplicon > Dose)
# - In vivo mode: parse Description like '...-A1-0.2mpk' -> group='...-A', rep=1, dose='0.2mpk'
# - Indel% main chart; optional %OOF chart (same order/colors)
# - QC trimming; GraphPad tables incl. %OOF; Excel export; hi-res camera export

import io
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="CRISPR Indel/%OOF Explorer", layout="wide")
st.title("CRISPR Indel / %OOF Explorer")

st.caption(
    "Replicates are grouped by identical **Description** (or animal pattern in *in vivo* mode). "
    "**Sample_ID** → Plate/Well (e.g., `py3B1` → plate `py3`, well `B1`). "
    "**Description** → Group and Dose (suffix). In *in vivo* mode, animal replicates are decoded from the token just before the last `-` (e.g., `...-A1-0.2mpk`)."
)

# ---------------- Sample dataset (for demo only) ----------------
def load_sample() -> pd.DataFrame:
    return pd.DataFrame({
        "Sample_ID": ["py3A1","py3A2","py3A3","py3B1","py3B2","py3B3","py4A1","py4A2","py4A3","py4B1","py4B2","py4B3"],
        "Description": ["A-120","A-120","A-120","A-60","A-60","A-60","B-120","B-120","B-120","B-30","B-30","B-30"],
        "Amplicon_No": ["amp1","amp1","amp1","amp1","amp1","amp1","amp2","amp2","amp2","amp2","amp2","amp2"],
        "Indel%": [45.2, 47.1, 46.5, 10.1, 12.3, 9.8, 33.2, 32.5, 35.0, 55.0, 54.2, 53.5],
        "%OOF":   [ 5.1,  4.8,  5.3,  0.9,  1.2, 1.0,  3.2,  3.5,  3.1,  8.0,  7.6,  7.9],
        "Reads_in_input": [1.2e5, 1.1e5, 1.3e5, 9.0e4, 8.5e4, 8.8e4, 2.0e5, 2.1e5, 1.8e5, 6.0e4, 6.5e4, 6.2e4],
        "Reads_aligned_all_amplicons": [1.0e5, 1.0e5, 1.1e5, 7.0e4, 6.8e4, 6.9e4, 1.6e5, 1.7e5, 1.5e5, 5.5e4, 5.6e4, 5.4e4],
    })

# ---------------- Sidebar: Load data + QC ----------------
with st.sidebar:
    st.header("1) Load data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    use_sample = st.checkbox("Use example dataset", value=not uploaded)

    st.markdown("**Trim low-quality rows (optional)**")
    apply_qc = st.checkbox("Enable QC trimming", value=True)
    qc_reads_min = st.number_input("Min Reads_in_input", min_value=0, value=500, step=50)
    qc_align_min = st.number_input("Min alignment%", min_value=0, max_value=100, value=80, step=1)

@st.cache_data
def read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def infer_columns(df: pd.DataFrame):
    cols = list(df.columns)
    amp_exact = "Amplicon_No" if "Amplicon_No" in cols else None
    desc_col = next((c for c in cols if re.search(r"^desc(ription)?$|description", c, flags=re.I)), None)
    sid_col  = next((c for c in cols if re.search(r"^sample[_ ]?id$|sampleid|sample[-_ ]?name", c, flags=re.I)), None)
    indel_col = next((c for c in cols if re.search(r"\bindel\b", c, flags=re.I)), None)
    oof_col   = next((c for c in cols if re.search(r"\boof\b|out[-_ ]?of[-_ ]?frame", c, flags=re.I)), None)
    rin_col   = next((c for c in cols if re.search(r"reads.*input|input.*reads", c, flags=re.I)), None)
    raa_col   = next((c for c in cols if re.search(r"reads.*aligned.*amplicon|aligned.*amplicon", c, flags=re.I)), None)
    amp_col   = amp_exact or next((c for c in cols if re.search(r"\bamplicon(_?no)?\b|\bamplicon\b", c, flags=re.I)), None)
    if indel_col is None:
        indel_col = next((c for c in cols if re.search(r"indel.*%|%.*indel|indel[_ ]?percent", c, flags=re.I)), None)
    if oof_col is None:
        oof_col = next((c for c in cols if re.search(r"oof.*%|%.*oof|oof[_ ]?percent", c, flags=re.I)), None)
    return {"desc": desc_col, "sid": sid_col, "indel": indel_col, "oof": oof_col,
            "rin": rin_col, "raa": raa_col, "amp": amp_col}

def numeric_from_string(s):
    if pd.isna(s): return np.nan
    m = re.search(r"(-?\d+(\.\d+)?)", str(s))
    return float(m.group(1)) if m else np.nan

def clean_metric_column(df, col):
    if col and col in df.columns:
        df[col] = df[col].apply(numeric_from_string)
    return df

def parse_plate_well(sample_id: str):
    if sample_id is None or (isinstance(sample_id, float) and np.isnan(sample_id)):
        return "Unknown", "NA"
    s = str(sample_id).strip()
    m = re.match(r"^(?P<plate>.*?)[ _-]?(?P<well>[A-Ha-h](?:0?[1-9]|1[0-2]))$", s)
    if not m:
        return "Unknown", s
    plate = m.group("plate"); well = m.group("well").upper()
    well = well[0] + str(int(well[1:]))  # normalize B01->B1
    return plate, well

# -------- In vivo toggle (before parsing) --------
with st.sidebar:
    st.header("Study type")
    in_vivo_mode = st.checkbox(
        "This is *in vivo* data (animal replicates encoded in Description like `...-A1-0.2mpk`)",
        value=False,
        help="When enabled, the token right before the last '-' is parsed as Group+Replicate (e.g., A1,B2). Group becomes '...-A', replicate becomes 1."
    )

def parse_group_and_dose(description: str, in_vivo: bool):
    # returns: group_label, dose_str, dose_val, rep_num
    if description is None or (isinstance(description, float) and np.isnan(description)):
        return "Unknown", "NA", np.nan, np.nan
    s = str(description).strip()
    if "-" not in s:
        return s, "NA", np.nan, np.nan
    prefix, dose = s.rsplit("-", 1)
    dose_val = numeric_from_string(dose)
    rep_num = np.nan
    if in_vivo:
        # match trailing Letter+Digits at end of prefix
        m = re.search(r"([A-Za-z])(\d+)$", prefix)
        if m:
            # group label = prefix with the trailing digits removed (keep letter)
            group_label = re.sub(r"([A-Za-z])(\d+)$", r"\1", prefix)
            rep_num = float(m.group(2))
            return group_label, dose, dose_val, rep_num
    # default: group is everything before last '-'
    return prefix, dose, dose_val, np.nan

# ---------- Load raw ----------
if uploaded:
    raw = read_csv(uploaded)
elif use_sample:
    raw = load_sample()
    st.info("Using a built-in example dataset. Uncheck to upload your CSV.")
else:
    st.stop()

info = infer_columns(raw)

# ---------- Column confirmation ----------
with st.expander("Detected columns / change if needed", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        desc_col  = st.selectbox("Description", options=raw.columns, index=(raw.columns.get_loc(info["desc"]) if info["desc"] in raw.columns else 0))
        sid_col   = st.selectbox("Sample_ID",  options=raw.columns, index=(raw.columns.get_loc(info["sid"])  if info["sid"]  in raw.columns else 0))
    with c2:
        indel_col = st.selectbox("Indel% column", options=raw.columns, index=(raw.columns.get_loc(info["indel"]) if info["indel"] in raw.columns else 0))
        oof_col   = st.selectbox("%OOF column",   options=[None] + list(raw.columns), index=(1 + raw.columns.get_loc(info["oof"]) if info["oof"] in raw.columns else 0))
    with c3:
        rin_col   = st.selectbox("Reads_in_input", options=[None] + list(raw.columns), index=(1 + raw.columns.get_loc(info["rin"]) if info["rin"] in raw.columns else 0))
        raa_col   = st.selectbox("Reads_aligned_all_amplicons", options=[None] + list(raw.columns), index=(1 + raw.columns.get_loc(info["raa"]) if info["raa"] in raw.columns else 0))
    amp_col = st.selectbox("Amplicon column", options=[None] + list(raw.columns),
                           index=(1 + raw.columns.get_loc(info["amp"]) if info["amp"] in raw.columns else 0))

# ---------- Clean numerics & parse fields ----------
df = raw.copy()
df = clean_metric_column(df, indel_col)
if oof_col: df = clean_metric_column(df, oof_col)
if rin_col: df[rin_col] = df[rin_col].apply(numeric_from_string)
if raa_col: df[raa_col] = df[raa_col].apply(numeric_from_string)

plates, wells = zip(*df[sid_col].map(parse_plate_well))
df["_Plate"] = list(plates); df["_Well"] = list(wells)
group_label, dose, dose_val, repnum = zip(*df[desc_col].map(lambda s: parse_group_and_dose(s, in_vivo_mode)))
df["_Group"]   = list(group_label)
df["_Dose"]    = list(dose)
df["_DoseVal"] = list(dose_val)
df["_RepNum"]  = list(repnum)  # NaN for non-in-vivo or unmatched rows
# Normalize missing/blank doses → "0", and recompute numeric value
df["_Dose"] = df["_Dose"].astype(str).replace({None: "0", "nan": "0", "NA": "0", "None": "0", "": "0"})
df["_DoseVal"] = df["_Dose"].apply(numeric_from_string)

# ---------- Compute alignment% and apply QC trimming ----------
if rin_col and raa_col:
    df["alignment%"] = np.where(df[rin_col] > 0, (df[raa_col] / df[rin_col]) * 100.0, np.nan)
else:
    df["alignment%"] = np.nan

with st.sidebar:
    st.markdown("---")

if apply_qc:
    mask_qc = pd.Series(True, index=df.index)
    if rin_col:
        mask_qc &= df[rin_col].ge(qc_reads_min)
    if rin_col and raa_col:
        mask_qc &= df["alignment%"].ge(qc_align_min)
    df = df[mask_qc].copy()
    kept = len(df)
    total = len(raw)
    removed = total - kept
    st.success(f"QC trimming kept {kept}/{total} rows ({removed} removed).")

# ---------- QC plots ----------
st.subheader("QC: Sequencing depth & alignment")
qc_cols_missing = []
if rin_col is None: qc_cols_missing.append("Reads_in_input")
if raa_col is None: qc_cols_missing.append("Reads_aligned_all_amplicons")
if qc_cols_missing:
    st.warning(f"QC plots require {', '.join(qc_cols_missing)}. Select them above to enable QC.")
else:
    df["_ReadsIn_pos"] = np.where(df[rin_col] > 0, df[rin_col], np.nan)
    safe_max = np.nanmax(df["_ReadsIn_pos"]) if np.isfinite(np.nanmax(df["_ReadsIn_pos"])) else 1.0
    ticks = []; t = 1.0
    while t <= max(1.0, safe_max*1.2):
        ticks.append(t); t *= 10.0
    qc1, qc2 = st.columns(2)
    with qc1:
        st.caption("Reads_in_input (log scale)")
        fig1 = px.strip(df.assign(_X="All"), x="_X", y="_ReadsIn_pos",
                        hover_data={sid_col: True, desc_col: True, rin_col: True})
        fig1.update_yaxes(type="log", tickvals=ticks, ticktext=[str(int(v)) if v >= 1 else str(v) for v in ticks])
        fig1.update_layout(xaxis_title="", yaxis_title="Reads_in_input (log scale)",
                           showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig1, use_container_width=True)
    with qc2:
        st.caption("alignment% = Reads_aligned_all_amplicons / Reads_in_input × 100")
        fig2 = px.strip(df.assign(_X="All"), x="_X", y="alignment%",
                        hover_data={sid_col: True, desc_col: True, rin_col: True, raa_col: True})
        fig2.update_layout(xaxis_title="", yaxis_title="alignment% (all samples)",
                           showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ---------------- Sidebar: Filters & options ----------------
def _on_plates_change():
    # user changed Plate(s) manually → reset downstream
    st.session_state.selected_amplicons = []
    st.session_state.selected_doses = []

def _on_amplicons_change():
    # user changed Amplicon(s) manually → reset downstream
    st.session_state.selected_doses = []

with st.sidebar:
    st.header("2) Filters (to show)")

    # keep state keys stable
    for k in ("selected_plates", "selected_amplicons", "selected_doses"):
        st.session_state.setdefault(k, [])

    # ---------- PLATE(S) ----------
    available_plates = sorted(pd.Series(df["_Plate"].astype(str)).unique().tolist())

    st.write("**Plate(s)**")
    c1, c2 = st.columns(2)
    with c1:
        st.button(
            "Select all",
            key="plates_all",
            on_click=lambda: st.session_state.update(
                selected_plates=available_plates,
                selected_amplicons=[],
                selected_doses=[],
            ),
        )
    with c2:
        st.button(
            "Clear",
            key="plates_clear",
            on_click=lambda: st.session_state.update(
                selected_plates=[],
                selected_amplicons=[],
                selected_doses=[],
            ),
        )

    selected_plates = st.multiselect(
        "",
        options=available_plates,
        default=st.session_state["selected_plates"],
        key="selected_plates",
        label_visibility="collapsed",
        on_change=_on_plates_change,   # clears downstream when user edits manually
    )

    # ---------- AMPLICON ----------
    if amp_col:
        amp_pool = (
            df[df["_Plate"].astype(str).isin(selected_plates)][amp_col].astype(str)
            if selected_plates else df[amp_col].astype(str)
        )
        available_amplicons = sorted(
            [a for a in amp_pool.dropna().unique().tolist() if a != ""]
        )
    else:
        available_amplicons = []

    st.write("**Amplicon**")
    c1, c2 = st.columns(2)
    with c1:
        st.button(
            "Select all",
            key="amp_all",
            on_click=lambda: st.session_state.update(
                selected_amplicons=available_amplicons,
                selected_doses=[],
            ),
        )
    with c2:
        st.button(
            "Clear",
            key="amp_clear",
            on_click=lambda: st.session_state.update(
                selected_amplicons=[],
                selected_doses=[],
            ),
        )

    selected_amplicons = st.multiselect(
        "",
        options=available_amplicons,
        default=st.session_state["selected_amplicons"],
        key="selected_amplicons",
        label_visibility="collapsed",
        on_change=_on_amplicons_change,  # clears doses when user edits manually
    )

    # ---------- DOSE(S) ----------
    if selected_plates:
        dose_pool = df[df["_Plate"].astype(str).isin(selected_plates)].copy()
        if amp_col and selected_amplicons:
            dose_pool = dose_pool[dose_pool[amp_col].astype(str).isin(selected_amplicons)]
        available_doses = [
            d for d in sorted(dose_pool["_Dose"].astype(str).unique()) if d != "NA"
        ]
    else:
        available_doses = []

    st.write("**Dose(s)**")
    c1, c2 = st.columns(2)
    with c1:
        st.button(
            "Select all",
            key="dose_all",
            on_click=lambda: st.session_state.update(selected_doses=available_doses),
        )
    with c2:
        st.button(
            "Clear",
            key="dose_clear",
            on_click=lambda: st.session_state.update(selected_doses=[]),
        )

    selected_doses = st.multiselect(
        "",
        options=available_doses,
        default=st.session_state["selected_doses"],
        key="selected_doses",
        label_visibility="collapsed",
    )


    st.header("3) Plot options")
    group_by_series = st.checkbox("Group by dose series (keep doses of same group together)", value=True)
    show_oof = st.checkbox("Show %OOF chart", value=False, key="show_oof")
    dose_order_choice = st.radio("Dose order within group", options=["High → Low", "Low → High"], index=0)
    if group_by_series:
        group_order_mode = st.radio(
            "Group order",
            options=["As input order", "By selected dose mean ↓", "By selected dose mean ↑"],
            index=0,
        )
        dose_for_group_sort = None
        if group_order_mode != "As input order" and selected_doses:
            dose_for_group_sort = st.selectbox("Select dose for mean-based ordering", options=selected_doses, index=0)
    else:
        group_order_mode = None
        dose_for_group_sort = None
        ungrouped_order = st.radio("Ungrouped bar order", options=["By mean ↓", "By mean ↑"], index=0)

    st.header("4) Image export (camera icon)")
    lock_ratio = st.checkbox("Lock aspect ratio 16:9", value=True)
    png_w = st.number_input("PNG width (px)", min_value=800, max_value=2400, value=1200, step=50)
    if lock_ratio:
        png_h = int(round(png_w * 9 / 16))
        st.caption(f"Height auto-set to {png_h}px (16:9)")
    else:
        png_h = st.number_input("PNG height (px)", min_value=500, max_value=1500, value=675, step=25)
    png_scale = st.slider("Scale (sharpness)", 1, 3, 2)

# guards
if not selected_plates:
    st.info("Select at least one plate to continue."); st.stop()
if not selected_doses:
    st.info("Select one or more doses to continue."); st.stop()

# ---------------- Prepare filtered df ----------------
mask = df["_Plate"].astype(str).isin(selected_plates) & df["_Dose"].astype(str).isin(selected_doses)
if amp_col and selected_amplicons:
    mask &= df[amp_col].astype(str).isin(selected_amplicons)
fdf = df[mask].copy()

# warn if in vivo mode but no rows match animal pattern
if in_vivo_mode and not fdf["_RepNum"].notna().any():
    st.warning("**In vivo mode is enabled, but none of the selected rows match the animal pattern** "
               "(letter+number right before the last '-'). Showing standard grouping instead.")

# ---------------- Aggregate for Indel% ----------------
agg_indel = (
    fdf.groupby([ "_Group", "_Dose", "_DoseVal"], dropna=False)[indel_col]
       .agg(mean="mean", std="std", count="count")
       .reset_index()
)
# reconstruct label for x-axis
agg_indel["_DescLabel"] = agg_indel["_Group"].astype(str) + "-" + agg_indel["_Dose"].astype(str)

# Dose ordering
dose_low_to_high = (
    agg_indel[["_Dose", "_DoseVal"]].drop_duplicates().sort_values(
        by=["_DoseVal", "_Dose"], ascending=[True, True]
    )["_Dose"].astype(str).tolist()
)
dose_high_to_low = list(reversed(dose_low_to_high))
dose_order = dose_low_to_high if dose_order_choice == "Low → High" else dose_high_to_low

# X categories
if group_by_series:
    groups_input_order = pd.unique(fdf["_Group"].astype(str)).tolist()
    if group_order_mode == "As input order":
        groups_sorted = groups_input_order
    elif group_order_mode in ("By selected dose mean ↓", "By selected dose mean ↑") and dose_for_group_sort:
        by_group_dose = agg_indel[agg_indel["_Dose"].astype(str) == dose_for_group_sort].groupby("_Group")["mean"].mean()
        all_groups = agg_indel["_Group"].astype(str).unique().tolist()
        scores = {g: by_group_dose.get(g, np.nan) for g in all_groups}
        desc = group_order_mode.endswith("↓")
        def sort_key(g):
            v = scores[g]
            return (-np.inf if pd.isna(v) and not desc else (np.inf if pd.isna(v) and desc else v))
        groups_sorted = sorted(all_groups, key=sort_key, reverse=desc)
    else:
        groups_sorted = groups_input_order

    x_categories = []
    for g in groups_sorted:
        present = [d for d in dose_order if d in set(agg_indel.loc[agg_indel["_Group"] == g, "_Dose"].astype(str))]
        x_categories.extend([f"{g}-{d}" for d in present])
else:
    asc = (ungrouped_order == "By mean ↑")
    order_by_mean = agg_indel.sort_values(by=["mean", "_Group", "_DoseVal"], ascending=[asc, True, False])["_DescLabel"].tolist()
    x_categories = list(dict.fromkeys(order_by_mean))

# Apply categorical order
agg_indel["_DescLabel"] = pd.Categorical(agg_indel["_DescLabel"], categories=x_categories, ordered=True)

# ---------------- Colors: darkest = highest dose ----------------
palette_cycle = [
    px.colors.sequential.Reds,
    px.colors.sequential.Blues,
    px.colors.sequential.Greens,
    px.colors.sequential.Oranges,
    px.colors.sequential.Purples,
    px.colors.sequential.Greys,
]
def darkest_to_lightest(pal, n):
    half = pal[max(0, len(pal)//2):]
    seq = list(reversed(half))
    if n <= 1: return [seq[0]]
    idxs = [int(i*(len(seq)-1)/(n-1)) for i in range(n)]
    return [seq[i] for i in idxs]

color_discrete_map = {}
for gi, (g, sub) in enumerate(agg_indel.groupby("_Group")):
    pal = palette_cycle[gi % len(palette_cycle)]
    subu = sub.drop_duplicates(subset="_Dose").copy()
    if subu["_DoseVal"].notna().any():
        rank_order = subu.sort_values("_DoseVal", ascending=False)["_Dose"].astype(str).tolist()
    else:
        rank_order = [d for d in dose_high_to_low if d in set(subu["_Dose"].astype(str))]
    shades = darkest_to_lightest(pal, max(len(rank_order), 1))
    for d, c in zip(rank_order, shades):
        color_discrete_map[f"{g}-{d}"] = c

# ---------------- Indel% plot ----------------
fig = px.bar(
    agg_indel.sort_values("_DescLabel"),
    x="_DescLabel", y="mean", error_y="std",
    color="_DescLabel", color_discrete_map=color_discrete_map,
    hover_data={"_Group": True, "_Dose": True, "_DoseVal": True, "mean":":.2f", "std":":.2f", "count":True},
)
fig.update_layout(margin=dict(l=10, r=10, t=30, b=10),
                  xaxis_title=None, yaxis_title="Mean Indel% (± SD)", showlegend=False)
config = {
    "toImageButtonOptions": {"format": "png", "filename": "indel_bars",
                             "width": int(png_w), "height": int(png_h), "scale": int(png_scale)},
    "displaylogo": False,
}
st.plotly_chart(fig, use_container_width=True, config=config)

# ---------------- Optional %OOF plot (same order/colors) ----------------
if show_oof and oof_col and oof_col in fdf.columns:
    agg_oof = (
        fdf.groupby(["_Group", "_Dose", "_DoseVal"], dropna=False)[oof_col]
           .agg(mean="mean", std="std", count="count")
           .reset_index()
    )
    agg_oof["_DescLabel"] = agg_oof["_Group"].astype(str) + "-" + agg_oof["_Dose"].astype(str)
    agg_oof["_DescLabel"] = pd.Categorical(agg_oof["_DescLabel"], categories=x_categories, ordered=True)
    fig2 = px.bar(
        agg_oof.sort_values("_DescLabel"),
        x="_DescLabel", y="mean", error_y="std",
        color="_DescLabel", color_discrete_map=color_discrete_map,
        hover_data={"_Group": True, "_Dose": True, "_DoseVal": True, "mean":":.2f", "std":":.2f", "count":True},
    )
    fig2.update_layout(margin=dict(l=10, r=10, t=30, b=10),
                       xaxis_title=None, yaxis_title="Mean %OOF (± SD)", showlegend=False)
    config2 = {
        "toImageButtonOptions": {"format": "png", "filename": "oof_bars",
                                 "width": int(png_w), "height": int(png_h), "scale": int(png_scale)},
        "displaylogo": False,
    }
    st.plotly_chart(fig2, use_container_width=True, config=config2)

# ===================== Tables & Downloads =====================

# Rows used (trimmed by QC + filters)
ordered_cols = ["Sample_ID", "Description", "Indel%", "%OOF",
                "Reads_in_input", "Reads_aligned_all_amplicons", "alignment%"]
col_map = {"Sample_ID": sid_col, "Description": desc_col, "Indel%": indel_col, "%OOF": oof_col,
           "Reads_in_input": rin_col, "Reads_aligned_all_amplicons": raa_col, "alignment%": "alignment%"}
rows_df = pd.DataFrame({k: (np.nan if col_map[k] is None else fdf[col_map[k]]) for k in ordered_cols})

# Build order for GraphPad tables from plotted categories
groups_in_order = []
for lbl in x_categories:
    g = str(lbl).rsplit("-", 1)[0]
    if g not in groups_in_order:
        groups_in_order.append(g)
doses_high_to_low = list(reversed(dose_low_to_high))  # columns in high->low order

def fmt1(x):
    try: return f"{float(x):.1f}"
    except Exception: return str(x)

# Sort within (group, dose) by explicit animal replicate if available, else by well/sample order
def well_sort_key(w):
    s = str(w); m = re.match(r"^([A-Ha-h])\s*[-:]?\s*(\d{1,2})$", s)
    if m: return (ord(m.group(1).upper()) - ord('A'), int(m.group(2)))
    return (99, 99, s)

if fdf["_RepNum"].notna().any():
    fdf_sorted = fdf.sort_values(by=["_Group", "_Dose", "_RepNum", sid_col])
else:
    if "_Well" in fdf.columns and fdf["_Well"].notna().any():
        fdf_sorted = fdf.sort_values(by=["_Group", "_Dose", "_Well"], key=lambda s: s.map(well_sort_key) if s.name == "_Well" else s)
    else:
        fdf_sorted = fdf.sort_values(by=["_Group", "_Dose", sid_col])

fdf_sorted["_rep_idx"] = fdf_sorted.groupby(["_Group", "_Dose"]).cumcount() + 1

# ---------- GraphPad Type 1 (rows = Group, cols = Doses; uniform replicates) ----------
def make_type1(metric_label, metric_src_col):
    max_reps_global = 1
    for d in doses_high_to_low:
        cnt = (fdf_sorted[fdf_sorted["_Dose"].astype(str) == d]
               .groupby("_Group")[metric_src_col].count())
        if not cnt.empty:
            max_reps_global = max(max_reps_global, int(cnt.max()))
    cols = [f"{d}_{metric_label}_rep{i}" for d in doses_high_to_low for i in range(1, max_reps_global+1)]
    rows = []
    for g in groups_in_order:
        row = {"Description": g}
        subg = fdf_sorted[fdf_sorted["_Group"] == g]
        for d in doses_high_to_low:
            subgd = subg[subg["_Dose"].astype(str) == d]
            vals = [fmt1(v) for v in subgd.sort_values(["_RepNum","_rep_idx"]) [metric_src_col].tolist()]
            for i in range(1, max_reps_global+1):
                row[f"{d}_{metric_label}_rep{i}"] = (vals[i-1] if i-1 < len(vals) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows, columns=["Description"] + cols)

type1_indel = make_type1("Indel%", indel_col) if indel_col in fdf_sorted.columns else pd.DataFrame({"Description": groups_in_order})
type1_df = type1_indel
if oof_col and oof_col in fdf_sorted.columns:
    type1_oof = make_type1("%OOF", oof_col)
    type1_df = type1_indel.merge(type1_oof.drop(columns=["Description"]), left_index=True, right_index=True)

# ---------- GraphPad Type 2 (rows = Dose, cols = Groups; uniform replicates) ----------
def make_type2(metric_label, metric_src_col):
    max_reps_global = 1
    for g in groups_in_order:
        cnt = (fdf_sorted[fdf_sorted["_Group"] == g]
               .groupby("_Dose")[metric_src_col].count())
        if not cnt.empty:
            max_reps_global = max(max_reps_global, int(cnt.max()))
    cols = [f"{g}_{metric_label}_rep{i}" for g in groups_in_order for i in range(1, max_reps_global+1)]
    rows = []
    for d in doses_high_to_low:
        row = {"Dose": d}
        subd = fdf_sorted[fdf_sorted["_Dose"].astype(str) == d]
        for g in groups_in_order:
            subdg = subd[subd["_Group"] == g]
            vals = [fmt1(v) for v in subdg.sort_values(["_RepNum","_rep_idx"]) [metric_src_col].tolist()]
            for i in range(1, max_reps_global+1):
                row[f"{g}_{metric_label}_rep{i}"] = (vals[i-1] if i-1 < len(vals) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows, columns=["Dose"] + cols)

type2_indel = make_type2("Indel%", indel_col) if indel_col in fdf_sorted.columns else pd.DataFrame({"Dose": doses_high_to_low})
type2_df = type2_indel
if oof_col and oof_col in fdf_sorted.columns:
    type2_oof = make_type2("%OOF", oof_col)
    type2_df = type2_indel.merge(type2_oof.drop(columns=["Dose"]), left_index=True, right_index=True)

# ---------- Download ----------
st.subheader("Download")
excel_engine = None
try:
    import openpyxl  # noqa
    excel_engine = "openpyxl"
except Exception:
    try:
        import xlsxwriter  # noqa
        excel_engine = "xlsxwriter"
    except Exception:
        excel_engine = None

if excel_engine:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine=excel_engine) as writer:
        rows_df.to_excel(writer, index=False, sheet_name="rows_used")
        type1_df.to_excel(writer, index=False, sheet_name="graphpad_type1_groups")
        type2_df.to_excel(writer, index=False, sheet_name="graphpad_type2_doses")
    st.download_button("Download Excel (.xlsx) — rows + both GraphPad tables",
                       data=buf.getvalue(),
                       file_name="plot_data_and_graphpad.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Install `openpyxl` or `xlsxwriter` to enable .xlsx downloads.")

# ---------- Preview tables ----------
st.subheader("GraphPad-friendly wide table (rounded to 1 decimal)")
which_gp = st.radio("Layout to preview", options=["Type 1: rows=Group, cols=Dose", "Type 2: rows=Dose, cols=Group"],
                    index=0, horizontal=True)
if which_gp.startswith("Type 1"):
    st.dataframe(type1_df, use_container_width=True, hide_index=True)
else:
    st.dataframe(type2_df, use_container_width=True, hide_index=True)

# ---------- Rows used ----------
st.subheader("Rows used for this plot (filtered)")
st.dataframe(rows_df, use_container_width=True, hide_index=True)
