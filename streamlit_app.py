# streamlit_app.py  (v7.3)
# - Filters: Select all/Clear with hierarchy clears (Plate > Amplicon > Dose)
# - In vivo mode: parse Description like 'A1-...-0.2mpk' -> group='A-...', rep=1, dose='0.2mpk'
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
    "Replicates are grouped by identical **Description** (or by animal pattern in **in vivo** mode). "
    "**Sample_ID** → Plate/Well (e.g., `py3B1` → plate `py3`, well `B1`). "
    "**In vitro**: **Description** ⇒ group = text before the last `-`, dose = token after the last `-` (blank → `0`). "
    "**In vivo**: parse the **first** letter+number at the beginning (e.g., `A1-…-0.2mpk`) as Group=`A`, Replicate=`1`; "
    "dose still comes from the last `-`. Rows not matching this pattern are removed from filters, plots, and exports."
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
        "Analyze as **in vivo** data (names like `A1-…-0.2mpk`)",
        value=False,
        help=(
            "When on: parse the FIRST letter+number at the beginning (A1, B2, …) "
            "as Group=letter and Replicate=number. Dose comes from the last '-' token "
            "(blank doses become 0). Rows that don’t match this pattern are removed "
            "from filters, plots, and exports."
        ),
    )

def parse_group_and_dose(description: str, in_vivo: bool):
    """
    Returns: (group_label, dose_str, dose_val, rep_num, in_vivo_match)

    In vitro: group = text before last '-'; dose = text after last '-' (or '0' if absent); rep_num = NaN
    In vivo:  group = FIRST letter at start, replicate = digits immediately after it (e.g., A1, B2),
              dose = text after last '-' (or '0' if absent). Rows not matching A<digits> at start are flagged False.
    """
    if description is None or (isinstance(description, float) and np.isnan(description)):
        return "Unknown", "0", 0.0, np.nan, False

    s = str(description).strip()

    # dose: last token after '-' (fallback to "0")
    if "-" in s:
        prefix, dose = s.rsplit("-", 1)
    else:
        prefix, dose = s, "0"
    dose_val = numeric_from_string(dose)

    if in_vivo:
        # FIRST token: letter+number at the BEGINNING, e.g., A1, B2, Z10
        m = re.match(r"^\s*([A-Za-z])(\d+)\b", s)
        if m:
            group_label = m.group(1)           # just the letter (A, B, C…)
            rep_num = float(m.group(2))        # 1,2,3,...
            return group_label, dose, dose_val, rep_num, True
        else:
            # mark as non-match; caller will drop these rows in in-vivo mode
            return "NON-MATCH", dose, dose_val, np.nan, False

    # default: in vitro rule (group = prefix before last '-')
    return prefix, dose, dose_val, np.nan, True

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
group_label, dose, dose_val, repnum, match = zip(*df[desc_col].map(lambda s: parse_group_and_dose(s, in_vivo_mode)))
df["_Group"]   = list(group_label)
df["_Dose"]    = list(dose)
df["_DoseVal"] = list(dose_val)
df["_RepNum"]  = list(repnum)       # NaN for non-in-vivo or unmatched
df["_InVivoMatch"] = list(match)

# Normalize missing/blank doses → "0", and recompute numeric value
df["_Dose"] = df["_Dose"].astype(str).replace({None: "0", "nan": "0", "NA": "0", "None": "0", "": "0"})
df["_DoseVal"] = df["_Dose"].apply(numeric_from_string)

# STRICT: when in-vivo is ON, DROP non-matching rows up front (shrinks Plates/Doses to only relevant)
if in_vivo_mode:
    df = df[df["_InVivoMatch"]].copy()
    if df.empty:
        st.warning("In vivo mode is enabled, but none of the rows match the pattern like `A1-…-0.2mpk`. Turn off in vivo mode or check the data.")
        st.stop()

# ---------- Compute alignment% and apply QC trimming ----------
if rin_col and raa_col:
    df["alignment%"] = np.where(df[rin_col] > 0, (df[raa_col] / df[rin_col]) * 100.0, np.nan)
else:
    df["alignment%"] = np.nan

with st.sidebar:
    st.markdown("---")

if apply_qc:
    # Keep a copy BEFORE QC so we can report what's removed
    df_before_qc = df.copy()

    mask_qc = pd.Series(True, index=df.index)
    if rin_col:
        mask_qc &= df[rin_col].ge(qc_reads_min)
    if rin_col and raa_col:
        mask_qc &= df["alignment%"].ge(qc_align_min)

    # Rows removed by QC (from the pre-QC df)
    removed_mask = ~mask_qc
    removed_df = df_before_qc.loc[removed_mask].copy()

    # Pick columns to show (robust to missing)
    cols_show = []
    if sid_col: cols_show.append(sid_col)
    if desc_col: cols_show.append(desc_col)
    if rin_col: cols_show.append(rin_col)
    if raa_col: cols_show.append(raa_col)
    if "alignment%" in removed_df.columns: cols_show.append("alignment%")
    removed_df = removed_df[cols_show] if cols_show else removed_df

    # Canonical pretty names
    rename_map = {}
    if sid_col: rename_map[sid_col] = "Sample_ID"
    if desc_col: rename_map[desc_col] = "Description"
    if rin_col: rename_map[rin_col] = "Reads_in_input"
    if raa_col: rename_map[raa_col] = "Reads_aligned_all_amplicons"
    removed_df = removed_df.rename(columns=rename_map)

    # Apply QC filter
    df = df.loc[mask_qc].copy()

    kept = len(df)
    total_pre_qc = len(df_before_qc)
    removed = total_pre_qc - kept
    st.success(f"QC trimming kept {kept}/{total_pre_qc} rows ({removed} removed).")

    # Popout with removed Sample_IDs/details
    with st.expander(f"Show filtered-out samples ({removed})", expanded=False):
        if removed == 0:
            st.caption("No samples were removed by QC filters.")
        else:
            # If Sample_ID exists, show it first
            ordered = ["Sample_ID", "Description", "Reads_in_input", "Reads_aligned_all_amplicons", "alignment%"]
            cols_present = [c for c in ordered if c in removed_df.columns]
            st.dataframe(removed_df[cols_present], use_container_width=True, hide_index=True)

else:
    st.info("QC trimming is disabled. Enable it in the sidebar to filter by reads/alignment thresholds.")

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

    with st.expander("Choose plate(s)", expanded=False):
        q_pl = st.text_input(
            "Search plates", key="plate_search",
            label_visibility="collapsed", placeholder="Search plates"
        )
        plate_opts = [o for o in available_plates if not q_pl or q_pl.lower() in o.lower()]
        for o in plate_opts:
            st.checkbox(o, key=f"plate_opt_{o}", value=(o in st.session_state.selected_plates))
        if st.button("Apply plates", key="apply_plates"):
            st.session_state.selected_plates = [
                o for o in available_plates if st.session_state.get(f"plate_opt_{o}", False)
            ]
            _on_plates_change()  # clears downstream

    # current plate selection (read-only summary)
    selected_plates = st.session_state["selected_plates"]
    st.caption("Selected plates: " + (", ".join(selected_plates) if selected_plates else "none"))

    # ---------- AMPLICON ----------
    if amp_col:
        amp_pool = (
            df[df["_Plate"].astype(str).isin(selected_plates)][amp_col].astype(str)
            if selected_plates else df[amp_col].astype(str)
        )
        available_amplicons = sorted([a for a in amp_pool.dropna().unique().tolist() if a != ""])
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

    with st.expander("Choose amplicon(s)", expanded=False):
        q_amp = st.text_input(
            "Search amplicons", key="amp_search",
            label_visibility="collapsed", placeholder="Search amplicons"
        )
        amp_opts = [o for o in available_amplicons if not q_amp or q_amp.lower() in o.lower()]
        for o in amp_opts:
            st.checkbox(o, key=f"amp_opt_{o}", value=(o in st.session_state.selected_amplicons))
        if st.button("Apply amplicons", key="apply_amplicons"):
            st.session_state.selected_amplicons = [
                o for o in available_amplicons if st.session_state.get(f"amp_opt_{o}", False)
            ]
            _on_amplicons_change()  # clears doses

    selected_amplicons = st.session_state["selected_amplicons"]
    st.caption("Selected amplicons: " + (", ".join(selected_amplicons) if selected_amplicons else "none"))

    # ---------- DOSE(S) ----------
    if selected_plates:
        dose_pool = df[df["_Plate"].astype(str).isin(selected_plates)].copy()
        if amp_col and selected_amplicons:
            dose_pool = dose_pool[dose_pool[amp_col].astype(str).isin(selected_amplicons)]
        available_doses = [d for d in sorted(dose_pool["_Dose"].astype(str).unique()) if d != "NA"]
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

    with st.expander("Choose dose(s)", expanded=False):
        q_dose = st.text_input(
            "Search doses", key="dose_search",
            label_visibility="collapsed", placeholder="Search doses"
        )
        dose_opts = [o for o in available_doses if not q_dose or q_dose.lower() in o.lower()]
        for o in dose_opts:
            st.checkbox(o, key=f"dose_opt_{o}", value=(o in st.session_state.selected_doses))
        if st.button("Apply doses", key="apply_doses"):
            st.session_state.selected_doses = [
                o for o in available_doses if st.session_state.get(f"dose_opt_{o}", False)
            ]

    selected_doses = st.session_state["selected_doses"]
    st.caption("Selected doses: " + (", ".join(selected_doses) if selected_doses else "none"))

    st.header("3) Plot options")

    # Always show OOF toggle
    show_oof = st.checkbox("Show %OOF chart", value=False, key="show_oof")

    if in_vivo_mode:
        st.caption("In vivo mode: one bar per cohort = Group–Amplicon–Dose (replicates = animals).")

        group_by_amplicon = st.checkbox(
            "Group by amplicon (cluster by amplicon; input order within each)",
            value=False,
            disabled=(amp_col is None),
        )

        vivo_group_order = st.radio(
            "Cohort order",
            ["As input order", "By dose: High → Low", "By dose: Low → High"],
            index=0,
            help="Sort cohorts by their dose. Cohorts with the same dose keep input order.",
        )

    else:
        # original in vitro controls
        group_by_series = st.checkbox("Group by dose series (keep doses of same group together)", value=True)
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

# Frame used for plotting/GraphPad
fdf_plot = fdf.copy()

def _amp_label(x):
    s = str(x)
    # make "1" -> "Amp1", leave "Amp2" as-is
    return s if s.lower().startswith("amp") else (f"Amp{s}" if s.replace('.','',1).isdigit() else s)

if in_vivo_mode:
    if not amp_col:
        st.warning("In vivo mode needs an amplicon column to form cohorts (Group–Amp–Dose).")
        st.stop()
    fdf_plot["_AmpLbl"] = fdf_plot[amp_col].astype(str).map(_amp_label)
    fdf_plot["_Cohort"] = fdf_plot["_Group"].astype(str) + "-" + fdf_plot["_AmpLbl"].astype(str)
else:
    # keep a neutral label so downstream code works
    fdf_plot["_AmpLbl"] = ""
    fdf_plot["_Cohort"] = fdf_plot["_Group"].astype(str)

# warn if in vivo mode but no rows match animal pattern
if in_vivo_mode and not fdf["_RepNum"].notna().any():
    st.warning("**In vivo mode is enabled, but none of the selected rows match the animal pattern** "
               "(letter+number right before the last '-'). Showing standard grouping instead.")

# ---------------- Aggregate for Indel% ----------------
# ----- Aggregation for plotting -----
if in_vivo_mode:
    # one bar per (Cohort, Dose) = (Group–Amp, Dose); animals are replicates
    agg_indel = (
        fdf_plot.groupby(["_Cohort", "_Group", "_Dose", "_DoseVal"], dropna=False)[indel_col]
                .agg(mean="mean", std="std", count="count")
                .reset_index()
    )
    agg_indel["_DescLabel"] = agg_indel["_Cohort"].astype(str) + "-" + agg_indel["_Dose"].astype(str)
else:
    agg_indel = (
        fdf_plot.groupby(["_Group", "_Dose", "_DoseVal"], dropna=False)[indel_col]
                .agg(mean="mean", std="std", count="count")
                .reset_index()
    )
    agg_indel["_DescLabel"] = agg_indel["_Group"].astype(str) + "-" + agg_indel["_Dose"].astype(str)

# ----- dose orders (used variously below) -----
dose_low_to_high = (
    agg_indel[["_Dose", "_DoseVal"]]
    .drop_duplicates()
    .sort_values(by=["_DoseVal", "_Dose"], ascending=[True, True])["_Dose"]
    .astype(str).tolist()
)
dose_high_to_low = list(reversed(dose_low_to_high))

# ---------------- Build x-axis categories (ordering) ----------------
if in_vivo_mode:
    # Cohorts are Group–Amp; preserve input order by default
    cohorts_input_order = list(pd.unique(fdf_plot["_Cohort"].astype(str)))

    # dose per cohort (each should have one dose)
    c2dose = agg_indel.groupby("_Cohort")["_DoseVal"].mean()

    # read UI choices safely
    group_by_amplicon = locals().get("group_by_amplicon", False)
    vivo_group_order  = locals().get("vivo_group_order", "As input order")

    if group_by_amplicon and amp_col:
        # cluster cohorts by amplicon (first appearance order)
        cohort2amp = (fdf_plot.drop_duplicates(["_Cohort"])
                               .set_index("_Cohort")["_AmpLbl"].to_dict())
        amp_order = []
        for c in cohorts_input_order:
            a = cohort2amp.get(c, "")
            if a not in amp_order:
                amp_order.append(a)
        cohorts_sorted = []
        for a in amp_order:
            cohorts_sorted.extend([c for c in cohorts_input_order if cohort2amp.get(c, "") == a])
    else:
        # order by dose across cohorts; ties keep input order; cohorts without dose go last
        hasdose = [c for c in cohorts_input_order if pd.notna(c2dose.get(c))]
        nodose  = [c for c in cohorts_input_order if c not in hasdose]
        if vivo_group_order == "By dose: High → Low":
            cohorts_sorted = sorted(hasdose, key=lambda c: c2dose[c], reverse=True) + nodose
        elif vivo_group_order == "By dose: Low → High":
            cohorts_sorted = sorted(hasdose, key=lambda c: c2dose[c]) + nodose
        else:
            cohorts_sorted = cohorts_input_order

    # one (cohort, dose) per bar
    x_categories = []
    for c in cohorts_sorted:
        d_list = agg_indel.loc[agg_indel["_Cohort"] == c, "_Dose"].astype(str).tolist()
        for d in d_list:
            x_categories.append(f"{c}-{d}")

else:
    # in vitro path (original behavior, with missing-dose groups sent to the end)
    group_by_series    = locals().get("group_by_series", True)
    dose_order_choice  = locals().get("dose_order_choice", "High → Low")
    group_order_mode   = locals().get("group_order_mode", "As input order")
    dose_for_group_sort= locals().get("dose_for_group_sort", None)
    ungrouped_order    = locals().get("ungrouped_order", "By mean ↓")

    groups_input_order = list(pd.unique(fdf_plot["_Group"].astype(str)))
    if group_by_series:
        if group_order_mode == "As input order" or not dose_for_group_sort:
            groups_sorted = groups_input_order
        else:
            by_group_dose = (agg_indel[agg_indel["_Dose"].astype(str) == dose_for_group_sort]
                             .groupby("_Group")["mean"].mean())
            groups_with = [g for g in groups_input_order if g in by_group_dose.index]
            groups_without = [g for g in groups_input_order if g not in by_group_dose.index]
            desc = group_order_mode.endswith("↓")
            groups_with_sorted = sorted(groups_with, key=lambda g: by_group_dose[g], reverse=desc)
            groups_sorted = groups_with_sorted + groups_without

        within_order = dose_low_to_high if dose_order_choice == "Low → High" else dose_high_to_low
        x_categories = []
        for g in groups_sorted:
            present = [d for d in within_order
                       if d in set(agg_indel.loc[agg_indel["_Group"] == g, "_Dose"].astype(str))]
            x_categories.extend([f"{g}-{d}" for d in present])
    else:
        asc = (ungrouped_order == "By mean ↑")
        order_by_mean = (
            agg_indel.sort_values(by=["mean", "_Group", "_DoseVal"], ascending=[asc, True, False])
            ["_DescLabel"].astype(str).tolist()
        )
        x_categories = list(dict.fromkeys(order_by_mean))

# ----- Robust category assignment & fallback -----
desc_vals = agg_indel["_DescLabel"].astype(str).tolist()
present = set(desc_vals)
if 'x_categories' not in locals() or not x_categories:
    x_categories = list(dict.fromkeys(desc_vals))
else:
    x_categories = [x for x in x_categories if x in present]
    if not x_categories:
        x_categories = list(dict.fromkeys(desc_vals))

agg_indel["_DescLabel"] = pd.Categorical(agg_indel["_DescLabel"], categories=x_categories, ordered=True)

# ---------------- Colors: ensure darkest = highest dose; cover all labels ----------------
palette_cycle = [
    px.colors.sequential.Blues,
    px.colors.sequential.Mint,
    px.colors.sequential.Oranges,
    px.colors.sequential.Purples,
    px.colors.sequential.Greens,
    px.colors.sequential.Reds,
    px.colors.sequential.Greys,
]

def darkest_to_lightest(pal, n):
    half = pal[max(0, len(pal)//2):]
    seq = list(reversed(half))
    if n <= 1: return [seq[0]]
    idxs = [int(i*(len(seq)-1)/(n-1)) for i in range(n)]
    return [seq[i] for i in idxs]

color_discrete_map = {}
if in_vivo_mode:
    # shade by dose within each GROUP; keys are cohort-dose labels
    for gi, g in enumerate(pd.unique(agg_indel["_Group"].astype(str))):
        pal = palette_cycle[gi % len(palette_cycle)]
        sub = agg_indel[agg_indel["_Group"] == g]
        subu = sub.drop_duplicates(subset="_Dose").copy()
        if subu["_DoseVal"].notna().any():
            rank_order = subu.sort_values("_DoseVal", ascending=False)["_Dose"].astype(str).tolist()
        else:
            rank_order = list(dict.fromkeys(sub["_Dose"].astype(str)))
        shades = darkest_to_lightest(pal, max(len(rank_order), 1))
        dose2shade = dict(zip(rank_order, shades))
        for _, row in sub.iterrows():
            lbl = f"{row['_Cohort']}-{row['_Dose']}"
            color_discrete_map[lbl] = dose2shade.get(str(row["_Dose"]), shades[0])
else:
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

# Ensure every label has a color
fallback = px.colors.qualitative.Plotly
for lbl in pd.unique(agg_indel["_DescLabel"].astype(str)):
    if lbl not in color_discrete_map:
        color_discrete_map[lbl] = fallback[len(color_discrete_map) % len(fallback)]

# ---------------- Plots ----------------
# Safe defaults for image export if UI didn't run
if "png_w" not in locals(): png_w = 1200
if "lock_ratio" not in locals(): lock_ratio = True
if "png_h" not in locals(): png_h = int(round(png_w * 9 / 16)) if lock_ratio else 675
if "png_scale" not in locals(): png_scale = 2

if agg_indel.empty:
    st.warning("No data to plot after QC/filters. Try selecting more plates, amplicons, or doses.")
else:
    fig = px.bar(
        agg_indel.sort_values("_DescLabel"),
        x="_DescLabel", y="mean", error_y="std",
        color="_DescLabel", color_discrete_map=color_discrete_map,
        hover_data={"_Group": True, "_Dose": True, "_DoseVal": True, "mean":":.2f", "std":":.2f", "count":True},
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title=None, yaxis_title="Mean Indel% (± SD)", showlegend=False
    )
    config = {
        "toImageButtonOptions": {"format": "png", "filename": "indel_bars",
                                 "width": int(png_w), "height": int(png_h), "scale": int(png_scale)},
        "displaylogo": False,
    }
    st.plotly_chart(fig, use_container_width=True, config=config)

# ---------------- Optional %OOF plot (same order/colors) ----------------
if show_oof and oof_col and (oof_col in fdf_plot.columns):
    agg_oof = (
        (fdf_plot.groupby(["_Cohort", "_Group", "_Dose", "_DoseVal"], dropna=False)[oof_col].agg(mean="mean", std="std", count="count").reset_index())
        if in_vivo_mode else
        (fdf_plot.groupby(["_Group", "_Dose", "_DoseVal"], dropna=False)[oof_col].agg(mean="mean", std="std", count="count").reset_index())
    )
    agg_oof["_DescLabel"] = (
        agg_oof["_Cohort"].astype(str) + "-" + agg_oof["_Dose"].astype(str)
        if in_vivo_mode else
        agg_oof["_Group"].astype(str) + "-" + agg_oof["_Dose"].astype(str)
    )
    # lock to same x order
    agg_oof["_DescLabel"] = pd.Categorical(agg_oof["_DescLabel"], categories=x_categories, ordered=True)
    if not agg_oof.empty:
        fig2 = px.bar(
            agg_oof.sort_values("_DescLabel"),
            x="_DescLabel", y="mean", error_y="std",
            color="_DescLabel", color_discrete_map=color_discrete_map,
            hover_data={"_Group": True, "_Dose": True, "_DoseVal": True, "mean":":.2f", "std":":.2f", "count":True},
        )
        fig2.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title=None, yaxis_title="Mean %OOF (± SD)", showlegend=False
        )
        config2 = {
            "toImageButtonOptions": {"format": "png", "filename": "oof_bars",
                                     "width": int(png_w), "height": int(png_h), "scale": int(png_scale)},
            "displaylogo": False,
        }
        st.plotly_chart(fig2, use_container_width=True, config=config2)

# ===================== Tables & Downloads =====================

# Rows used (trimmed by QC + filters) — keep the raw filtered rows (fdf)
ordered_cols = ["Sample_ID", "Description", "Indel%", "%OOF",
                "Reads_in_input", "Reads_aligned_all_amplicons", "alignment%"]
col_map = {"Sample_ID": sid_col, "Description": desc_col, "Indel%": indel_col, "%OOF": oof_col,
           "Reads_in_input": rin_col, "Reads_aligned_all_amplicons": raa_col, "alignment%": "alignment%"}
rows_df = pd.DataFrame({k: (np.nan if col_map[k] is None else fdf[col_map[k]]) for k in ordered_cols})

# Build order for GraphPad tables from plotted categories
# In vivo: x_categories contain 'Cohort-Dose' -> cohort is 'Group-Amp'
groups_in_order = []
for lbl in x_categories:
    g = str(lbl).rsplit("-", 1)[0]  # strip trailing '-Dose'
    if g not in groups_in_order:
        groups_in_order.append(g)

# Dose columns in high->low order (to the right)
doses_high_to_low = list(reversed(dose_low_to_high))

def fmt1(x):
    try:
        return f"{float(x):.1f}"
    except Exception:
        return str(x)

# Replicate sort helper (by well if available)
def well_sort_key(w):
    s = str(w)
    m = re.match(r"^([A-Ha-h])\s*[-:]?\s*(\d{1,2})$", s)
    if m:
        return (ord(m.group(1).upper()) - ord('A'), int(m.group(2)))
    return (99, 99, s)

# --- Choose the grouping key for GraphPad tables ---
# In vivo: group == cohort ('Group-Amp'); In vitro: group == '_Group'
gp_key = "_Cohort" if in_vivo_mode else "_Group"

# Source for GraphPad tables / plots is the plotting frame
src_df = fdf_plot.copy()

# Sort replicates within (group, dose)
if "_RepNum" in src_df.columns and src_df["_RepNum"].notna().any():
    fdf_sorted = src_df.sort_values(by=[gp_key, "_Dose", "_RepNum", sid_col])
else:
    if "_Well" in src_df.columns and src_df["_Well"].notna().any():
        fdf_sorted = src_df.sort_values(
            by=[gp_key, "_Dose", "_Well"],
            key=lambda s: s.map(well_sort_key) if s.name == "_Well" else s
        )
    else:
        fdf_sorted = src_df.sort_values(by=[gp_key, "_Dose", sid_col])

fdf_sorted["_rep_idx"] = fdf_sorted.groupby([gp_key, "_Dose"]).cumcount() + 1

# ---------- GraphPad Type 1 (rows = Group/Cohort, cols = Doses; uniform replicates) ----------
def make_type1(metric_label: str, metric_src_col: str) -> pd.DataFrame:
    max_reps_global = 1
    for d in doses_high_to_low:
        cnt = (fdf_sorted[fdf_sorted["_Dose"].astype(str) == d]
               .groupby(gp_key)[metric_src_col].count())
        if not cnt.empty:
            max_reps_global = max(max_reps_global, int(cnt.max()))

    cols = [f"{d}_{metric_label}_rep{i}"
            for d in doses_high_to_low
            for i in range(1, max_reps_global + 1)]

    rows = []
    for g in groups_in_order:
        row = {"Description": g}
        subg = fdf_sorted[fdf_sorted[gp_key] == g]
        for d in doses_high_to_low:
            subgd = subg[subg["_Dose"].astype(str) == d]
            vals = [fmt1(v) for v in subgd.sort_values(["_RepNum", "_rep_idx"])[metric_src_col].tolist()]
            for i in range(1, max_reps_global + 1):
                row[f"{d}_{metric_label}_rep{i}"] = (vals[i-1] if i-1 < len(vals) else np.nan)
        rows.append(row)

    return pd.DataFrame(rows, columns=["Description"] + cols)

type1_indel = (make_type1("Indel%", indel_col)
               if indel_col in fdf_sorted.columns else pd.DataFrame({"Description": groups_in_order}))

type1_df = type1_indel
if oof_col and oof_col in fdf_sorted.columns:
    type1_oof = make_type1("%OOF", oof_col)
    type1_df = type1_indel.merge(type1_oof.drop(columns=["Description"]),
                                 left_index=True, right_index=True)

# ---------- GraphPad Type 2 (rows = Dose, cols = Group/Cohort; uniform replicates) ----------
def make_type2(metric_label: str, metric_src_col: str) -> pd.DataFrame:
    max_reps_global = 1
    for g in groups_in_order:
        cnt = (fdf_sorted[fdf_sorted[gp_key] == g]
               .groupby("_Dose")[metric_src_col].count())
        if not cnt.empty:
            max_reps_global = max(max_reps_global, int(cnt.max()))

    cols = [f"{g}_{metric_label}_rep{i}"
            for g in groups_in_order
            for i in range(1, max_reps_global + 1)]

    rows = []
    for d in doses_high_to_low:
        row = {"Dose": d}
        subd = fdf_sorted[fdf_sorted["_Dose"].astype(str) == d]
        for g in groups_in_order:
            subdg = subd[subd[gp_key] == g]
            vals = [fmt1(v) for v in subdg.sort_values(["_RepNum", "_rep_idx"])[metric_src_col].tolist()]
            for i in range(1, max_reps_global + 1):
                row[f"{g}_{metric_label}_rep{i}"] = (vals[i-1] if i-1 < len(vals) else np.nan)
        rows.append(row)

    return pd.DataFrame(rows, columns=["Dose"] + cols)

type2_indel = (make_type2("Indel%", indel_col)
               if indel_col in fdf_sorted.columns else pd.DataFrame({"Dose": doses_high_to_low}))

type2_df = type2_indel
if oof_col and oof_col in fdf_sorted.columns:
    type2_oof = make_type2("%OOF", oof_col)
    type2_df = type2_indel.merge(type2_oof.drop(columns=["Dose"]),
                                 left_index=True, right_index=True)

# ---------- Download ----------
st.subheader("Download")
excel_engine = None
try:
    import openpyxl  # noqa: F401
    excel_engine = "openpyxl"
except Exception:
    try:
        import xlsxwriter  # noqa: F401
        excel_engine = "xlsxwriter"
    except Exception:
        excel_engine = None

if excel_engine:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine=excel_engine) as writer:
        rows_df.to_excel(writer, index=False, sheet_name="rows_used")
        type1_df.to_excel(writer, index=False, sheet_name="graphpad_type1_groups")
        type2_df.to_excel(writer, index=False, sheet_name="graphpad_type2_doses")
    st.download_button(
        "Download Excel (.xlsx) — rows + both GraphPad tables",
        data=buf.getvalue(),
        file_name="plot_data_and_graphpad.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.info("Install `openpyxl` or `xlsxwriter` to enable .xlsx downloads.")

# ---------- Preview tables ----------
st.subheader("GraphPad-friendly wide table (rounded to 1 decimal)")
which_gp = st.radio(
    "Layout to preview",
    options=["Type 1: rows=Group/Cohort, cols=Dose", "Type 2: rows=Dose, cols=Group/Cohort"],
    index=0, horizontal=True,
)
if which_gp.startswith("Type 1"):
    st.dataframe(type1_df, use_container_width=True, hide_index=True)
else:
    st.dataframe(type2_df, use_container_width=True, hide_index=True)

# ---------- Rows used ----------
st.subheader("Rows used for this plot (filtered)")
st.dataframe(rows_df, use_container_width=True, hide_index=True)
