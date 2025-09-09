# streamlit_app.py  (v8.0)
# - NEW in vivo semantics:
#   * Group letter + replicate parsed from leading token (A1, B2, …)
#   * gRNA = text between the 1st and last '-' in Description
#   * Dose = text after the last '-' (blank -> '0'); numeric extracted for ordering
#   * Bars aggregated by (GroupLetter, gRNA, Dose); x label: "A-<gRNA>-<dose>"
# - In vitro behavior, plotting, QC, and GraphPad tables remain as before
# - In vivo adds a compact gRNA table: rows=gRNA, cols=<dose>-Indel / <dose>-OOF

import io
import re
import os
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
    "**In vivo**: Group = **first letter** at the beginning (e.g., `A1-…-0.2mpk` → `A`), "
    "Replicate = digits after that letter (`1`), **gRNA** = the text **between the 1st and last '-'**, "
    "dose from the last '-' (blank → `0`). In vivo Excel table rows are by **(Group, gRNA)** (PBS rows kept per group)."
)

# ---------------- Sample dataset (for demo only) ----------------
def load_sample() -> pd.DataFrame:
    return pd.DataFrame({
        "Sample_ID": ["py3A1","py3A2","py3A3","py3B1","py3B2","py3B3","py4A1","py4A2","py4A3","py4B1","py4B2","py4B3"],
        "Description": [
            "A1-XXX-0.2", "A2-XXX-0.2", "A3-XXX-",   # dose blank -> 0
            "B1-YYY-0.2", "B2-YYY-0.2", "B3-YYY-0.2",
            "A1-XXX-0.4", "A2-XXX-0.4", "A3-XXX-0.4",
            "B1-YYY-0.4", "B2-YYY-0.4", "B3-YYY-0.4"
        ],
        "Amplicon_No": ["amp1"]*12,
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

def numeric_from_string(x):
    """Parse numbers robustly from mixed strings.

    Handles:
      - scientific notation:  -4.298e-09, 1E6
      - plain numbers:        12, 3.45, -0.7
      - with units/text:      '0.2mpk', '45%', 'reads: 1,234'
      - unicode minus:        '−3.5'
      - thousand separators:  '1,234.56'
    Returns float or NaN.
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip()
    if not s:
        return np.nan

    # normalize minus and remove thousands separators
    s = s.replace("−", "-").replace("–", "-").replace(",", "")

    # fast path: try full-string numeric parse first
    val = pd.to_numeric(s, errors="coerce")
    if not pd.isna(val):
        return float(val)

    # fallback: extract first numeric token (with optional exponent)
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else np.nan


def clean_metric_column(df, col):
    """Safely coerce a metric column to float using numeric_from_string."""
    if col and col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            df[col] = pd.to_numeric(s, errors="coerce")
        else:
            df[col] = s.apply(numeric_from_string)
    return df

# ---- helpers used by tables/plots ----
def fmt1(x):
    try:
        v = float(x)
        if v < 0 and abs(v) < 1e-12:  # kill tiny negatives so "-0.0" never appears
            v = 0.0
        s = f"{v:.1f}"
        return "0.0" if s == "-0.0" else s
    except Exception:
        return "" if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)

def clip_nonnegative_inplace(df, cols):
    """Coerce to numeric and clip negatives to 0.0 for the given columns."""
    for c in cols:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].mask(df[c] < 0, 0.0)

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

# ---- Helper to sort doses in dropdown menu ----
def sort_doses_for_ui(dose_list):
    # Try numeric sort (desc) if *all* entries parse to numbers; else alphabetical.
    vals = [numeric_from_string(d) for d in dose_list]
    if all(pd.notna(v) for v in vals):
        return [d for d, _ in sorted(zip(dose_list, vals), key=lambda t: t[1], reverse=True)]
    # alphabetical, case-insensitive, stable
    return sorted(dose_list, key=lambda x: (str(x).lower(), str(x)))


# ---- Excel helpers: autofit + header styling for both engines ----
def _is_oof_col(name: str) -> bool:
    return "oof" in str(name).lower()  # catches OOF, %OOF, -OOF-, etc.

def _guess_col_widths(df, min_w=6, max_w=28):
    # width in "Excel characters"; cap to keep it tidy
    widths = []
    for c in df.columns:
        header = str(c)
        body_max = max((len(str(v)) for v in df[c].fillna("").tolist()), default=0)
        w = max(len(header), body_max) + 2
        widths.append(max(min_w, min(w, max_w)))
    return widths

def style_sheet_xlsxwriter(writer, sheet_name, df):
    wb = writer.book
    ws = writer.sheets[sheet_name]

    # Header formats
    hfmt_indel = wb.add_format({
        "bold": True, "text_wrap": True, "align": "center", "valign": "vcenter",
        "bg_color": "#F8D7DA", "border": 1
    })
    hfmt_oof = wb.add_format({
        "bold": True, "text_wrap": True, "align": "center", "valign": "vcenter",
        "bg_color": "#D9E8FF", "border": 1  # light blue for OOF headers
    })

    # Write wrapped headers with per-column color
    for col, name in enumerate(df.columns):
        txt = format_header_for_excel(name)  # uses your hyphen/underscore-aware formatter
        ws.write(0, col, txt, hfmt_oof if _is_oof_col(name) else hfmt_indel)

    # Narrower width cap for these sheets (Type1/Type1-mean/Type2/in vivo)
    narrow_sheets = {"graphpad_type1_groups", "graphpad_type1_groups_mean",
                     "graphpad_type2_doses", "group_gRNA_by_dose_reps"}
    max_cap = 10 if sheet_name in narrow_sheets else 16
    widths = _guess_col_widths(df, min_w=6, max_w=max_cap)
    for col, w in enumerate(widths):
        ws.set_column(col, col, w)

    ws.freeze_panes(1, 1)
    ws.autofilter(0, 0, len(df), len(df.columns) - 1)
    ws.set_row(0, 48)  # header height

def style_sheet_openpyxl(writer, sheet_name, df):
    from openpyxl.utils import get_column_letter
    from openpyxl.styles import Alignment, Font, PatternFill, Border, Side

    ws = writer.sheets[sheet_name]
    ws.freeze_panes = "A2"
    last_col = get_column_letter(df.shape[1])
    ws.auto_filter.ref = f"A1:{last_col}{df.shape[0] + 1}"

    align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    bold = Font(bold=True)
    border = Border(left=Side(style="thin"), right=Side(style="thin"),
                    top=Side(style="thin"), bottom=Side(style="thin"))
    fill_indel = PatternFill("solid", fgColor="F8D7DA")
    fill_oof   = PatternFill("solid", fgColor="D9E8FF")  # light blue for OOF headers

    # Header cells (wrapped + colored per column)
    for j, name in enumerate(df.columns, start=1):
        cell = ws.cell(row=1, column=j)
        cell.value = format_header_for_excel(name)
        cell.alignment = align
        cell.font = bold
        cell.border = border
        cell.fill = fill_oof if _is_oof_col(name) else fill_indel

    ws.row_dimensions[1].height = 48

    # Narrower width cap for these sheets
    narrow_sheets = {"graphpad_type1_groups", "graphpad_type1_groups_mean",
                     "graphpad_type2_doses", "group_gRNA_by_dose_reps"}
    max_cap = 10 if sheet_name in narrow_sheets else 16
    widths = _guess_col_widths(df, min_w=6, max_w=max_cap)
    for j, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(j)].width = w

# -------- Excel helper to wrap header -------- #
import re, textwrap

def _soft_wrap_desc(s: str, width: int = 24) -> str:
    """
    Wrap long 'description' parts at '-' or '_' boundaries if possible,
    otherwise fall back to a soft width wrap.
    """
    s = str(s)
    # Prefer splitting on '-' first, then '_'
    for delim in ["-", "_"]:
        if delim in s and len(s) > width:
            parts = s.split(delim)
            lines, cur = [], ""
            for p in parts:
                add = (delim if cur else "") + p
                if len(cur) + len(add) > width and cur:
                    lines.append(cur)
                    cur = p
                else:
                    cur += add
            if cur:
                lines.append(cur)
            return "\n".join(lines)
    # Fallback plain soft wrap
    return "\n".join(textwrap.wrap(s, width=width)) if len(s) > width else s

def format_header_for_excel(name: str, desc_width: int = 24) -> str:
    """
    Supports:
      - in vivo:  '<dose>-Indel-repN' or '<dose>-OOF-repN'  →  '<dose>\\nindel|oof\\nrepN'
      - in vitro: '..._Indel%_repN' / '..._%OOF_repN' / '..._Indel%_mean' → '<left>\\nindel%|%oof\\nrepN|mean'
    """
    s = str(name)

    # A) in vivo: dose-metric-repN (e.g., '0.25-Indel-rep1')
    m = re.match(r"^(?P<dose>.+?)-(Indel|OOF)-(?P<rep>rep\d+)$", s, flags=re.I)
    if m:
        left_wrapped = _soft_wrap_desc(m.group("dose"), width=desc_width)
        metric = m.group(2)
        return f"{left_wrapped}\n{metric}\n{m.group('rep')}"

    # B) in vitro / underscore pattern
    parts = s.split("_")
    if len(parts) >= 3 and re.fullmatch(r"(rep\d+|mean)", parts[-1] or "", flags=re.I):
        metric = parts[-2]
        rep_or_mean = parts[-1]
        left = "_".join(parts[:-2])
        left_wrapped = _soft_wrap_desc(left, width=desc_width)
        return f"{left_wrapped}\n{metric}\n{rep_or_mean}"

    # Fallback: soft wrap and break on separators
    return _soft_wrap_desc(s, width=desc_width).replace("_", "\n").replace("-", "\n")

def _guess_col_widths(df, min_w=6, max_w=28):
    widths = []
    for c in df.columns:
        header = str(c)
        body_max = max((len(str(v)) for v in df[c].fillna("").tolist()), default=0)
        w = max(len(header), body_max) + 2
        widths.append(max(min_w, min(w, max_w)))
    return widths

def style_sheet(writer, engine_name, sheet_name, df):
    if engine_name == "xlsxwriter":
        style_sheet_xlsxwriter(writer, sheet_name, df)
    else:
        style_sheet_openpyxl(writer, sheet_name, df)

# -------- In vivo toggle (before parsing) --------
with st.sidebar:
    st.header("Study type")
    in_vivo_mode = st.checkbox(
        "Analyze as **in vivo** data (names like `A1-<gRNA>-<dose>`; blank dose → 0)",
        value=False,
        help=(
            "When on: Group = first letter at the very start (A/B/…), Replicate = digits after it; "
            "gRNA = text between the 1st and last '-', dose = token after the last '-' (blank → 0). "
            "Rows that don’t match a leading letter+number are dropped."
        ),
    )

def parse_ivv(description: str):
    """
    In vivo parser
    Returns: (group_letter, rep_num, gRNA, dose_str, dose_val, matched)
    - Leading token must start with letter+digits, e.g. 'A1' or 'C12'
    - gRNA = between first '-' and last '-' (if only one '-', gRNA becomes empty)
    - dose = after last '-' (blank -> '0')
    """
    if description is None or (isinstance(description, float) and np.isnan(description)):
        return "NON-MATCH", np.nan, "", "0", 0.0, False

    s = str(description).strip()

    # leading letter+digits
    m0 = re.match(r"^\s*([A-Za-z])(\d+)\b", s)
    if not m0:
        return "NON-MATCH", np.nan, "", "0", 0.0, False
    grp_letter = m0.group(1)
    rep_num = float(m0.group(2))

    # split by '-'
    parts = s.split("-")
    if len(parts) == 1:
        # no '-' -> no gRNA, no dose
        dose_str = "0"
        dose_val = 0.0
        grna = ""
    else:
        first = parts[0]
        last = parts[-1]
        middle = "-".join(parts[1:-1]) if len(parts) > 2 else ""
        grna = middle.strip()

        dose_str = last.strip()
        if dose_str == "":  # trailing '-' -> PBS/0
            dose_str = "0"
        dose_val = numeric_from_string(dose_str)

    return grp_letter, rep_num, grna, dose_str, dose_val, True

def parse_invitro(description: str):
    """
    In vitro fallback: group = text before last '-', dose = text after last '-' (blank -> '0')
    """
    if description is None or (isinstance(description, float) and np.isnan(description)):
        return "Unknown", "0", 0.0
    s = str(description).strip()
    if "-" in s:
        prefix, dose = s.rsplit("-", 1)
    else:
        prefix, dose = s, "0"
    if dose == "":
        dose = "0"
    return prefix, dose, numeric_from_string(dose)

# ---------- Load raw ----------
if uploaded:
    raw = read_csv(uploaded)
elif use_sample:
    raw = load_sample()
    st.info("Using a built-in example dataset. Uncheck to upload your CSV.")
else:
    st.stop()

info = infer_columns(raw)

# ---- Build Excel output name from input file name ----
if uploaded is not None and getattr(uploaded, "name", None):
    _input_name = uploaded.name
elif use_sample:
    _input_name = "example_dataset.csv"
else:
    _input_name = "input.csv"

_file_stem = os.path.splitext(os.path.basename(_input_name))[0]
# Requested style:
excel_filename = f"{_file_stem}_For_GraphPad_Input.xlsx"

# (Optional, clearer variants you can use instead)
# excel_filename_in_vivo  = f"{_file_stem}_InVivo_gRNA_Reps_GraphPad.xlsx"
# excel_filename_in_vitro = f"{_file_stem}_GraphPad_Tables.xlsx"

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
    amp_col = st.selectbox("Amplicon column (optional)", options=[None] + list(raw.columns),
                           index=(1 + raw.columns.get_loc(info["amp"]) if info["amp"] in raw.columns else 0))

# ---------- Clean numerics & parse fields ----------
df = raw.copy()
df = clean_metric_column(df, indel_col)
if oof_col: df = clean_metric_column(df, oof_col)
if rin_col: df[rin_col] = df[rin_col].apply(numeric_from_string)
if raa_col: df[raa_col] = df[raa_col].apply(numeric_from_string)

plates, wells = zip(*df[sid_col].map(parse_plate_well))
df["_Plate"] = list(plates); df["_Well"] = list(wells)

if in_vivo_mode:
    gl, rn, grna, dstr, dval, ok = zip(*df[desc_col].map(parse_ivv))
    df["_Group"] = list(gl)          # Group letter only
    df["_RepNum"] = list(rn)
    df["_gRNA"]   = list(grna)
    df["_Dose"]   = list(dstr)
    df["_DoseVal"]= list(dval)
    df["_InVivoMatch"] = list(ok)
    # Strict drop of non-matching rows
    df = df[df["_InVivoMatch"]].copy()
    if df.empty:
        st.warning("In vivo mode is enabled, but none of the rows match `A1-<gRNA>-<dose>`.")
        st.stop()
else:
    grp, dstr, dval = zip(*df[desc_col].map(parse_invitro))
    df["_Group"] = list(grp)
    df["_Dose"]  = list(dstr)
    df["_DoseVal"]= list(dval)
    df["_RepNum"] = np.nan
    df["_gRNA"]   = ""  # not used in vitro
    df["_InVivoMatch"] = True

# Normalize missing/blank doses → "0", and recompute numeric
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
    df_before_qc = df.copy()
    mask_qc = pd.Series(True, index=df.index)
    if rin_col:
        mask_qc &= df[rin_col].ge(qc_reads_min)
    if rin_col and raa_col:
        mask_qc &= df["alignment%"].ge(qc_align_min)

    removed_df = df_before_qc.loc[~mask_qc].copy()
    cols_show = []
    if sid_col: cols_show.append(sid_col)
    if desc_col: cols_show.append(desc_col)
    if rin_col: cols_show.append(rin_col)
    if raa_col: cols_show.append(raa_col)
    if "alignment%" in removed_df.columns: cols_show.append("alignment%")
    removed_df = removed_df[cols_show] if cols_show else removed_df
    rename_map = {}
    if sid_col: rename_map[sid_col] = "Sample_ID"
    if desc_col: rename_map[desc_col] = "Description"
    if rin_col: rename_map[rin_col] = "Reads_in_input"
    if raa_col: rename_map[raa_col] = "Reads_aligned_all_amplicons"
    removed_df = removed_df.rename(columns=rename_map)

    df = df.loc[mask_qc].copy()
    kept = len(df); total_pre_qc = len(df_before_qc); removed = total_pre_qc - kept
    st.success(f"QC trimming kept {kept}/{total_pre_qc} rows ({removed} removed).")
    with st.expander(f"Show filtered-out samples ({removed})", expanded=False):
        if removed == 0:
            st.caption("No samples were removed by QC filters.")
        else:
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
    st.session_state.selected_amplicons = []
    st.session_state.selected_doses = []

def _on_amplicons_change():
    st.session_state.selected_doses = []

with st.sidebar:
    st.header("2) Filters (to show)")
    for k in ("selected_plates", "selected_amplicons", "selected_doses"):
        st.session_state.setdefault(k, [])

    available_plates = sorted(pd.Series(df["_Plate"].astype(str)).unique().tolist())

    st.write("**Plate(s)**")
    c1, c2 = st.columns(2)
    with c1:
        st.button("Select all", key="plates_all",
                  on_click=lambda: st.session_state.update(
                      selected_plates=available_plates,
                      selected_amplicons=[], selected_doses=[]))
    with c2:
        st.button("Clear", key="plates_clear",
                  on_click=lambda: st.session_state.update(
                      selected_plates=[], selected_amplicons=[], selected_doses=[]))

    with st.expander("Choose plate(s)", expanded=False):
        q_pl = st.text_input("Search plates", key="plate_search",
                             label_visibility="collapsed", placeholder="Search plates")
        plate_opts = [o for o in available_plates if not q_pl or q_pl.lower() in o.lower()]
        for o in plate_opts:
            st.checkbox(o, key=f"plate_opt_{o}", value=(o in st.session_state.selected_plates))
        if st.button("Apply plates", key="apply_plates"):
            st.session_state.selected_plates = [
                o for o in available_plates if st.session_state.get(f"plate_opt_{o}", False)
            ]
            _on_plates_change()

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
    amp_disabled = (not selected_plates)

    c1, c2 = st.columns(2)
    with c1:
        st.button(
            "Select all",
            key="amp_all",
            disabled=amp_disabled,
            on_click=lambda: st.session_state.update(
                selected_amplicons=available_amplicons if not amp_disabled else [],
                selected_doses=[],
            ),
        )
    with c2:
        st.button(
            "Clear",
            key="amp_clear",
            disabled=amp_disabled,
            on_click=lambda: st.session_state.update(
                selected_amplicons=[],
                selected_doses=[],
            ),
        )

    with st.expander("Choose amplicon(s)", expanded=False):
        if amp_disabled:
            st.caption("Select plate(s) first.")
        else:
            q_amp = st.text_input(
                "Search amplicons", key="amp_search",
                label_visibility="collapsed", placeholder="Search amplicons"
            )
            amp_opts = [o for o in available_amplicons if not q_amp or q_amp.lower() in o.lower()]
            for o in amp_opts:
                st.checkbox(o, key=f"amp_opt_{o}", value=(o in st.session_state.selected_amplicons))
            if st.button("Apply amplicons", key="apply_amplicons", disabled=False):
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
        raw_doses = [d for d in dose_pool["_Dose"].astype(str).unique().tolist() if d != "NA"]
        available_doses = sort_doses_for_ui(raw_doses)
    else:
        available_doses = []

    st.write("**Dose(s)**")
    dose_disabled = (not selected_plates)

    c1, c2 = st.columns(2)
    with c1:
        st.button(
            "Select all",
            key="dose_all",
            disabled=dose_disabled,
            on_click=lambda: st.session_state.update(
                selected_doses=available_doses if not dose_disabled else []
            ),
        )
    with c2:
        st.button(
            "Clear",
            key="dose_clear",
            disabled=dose_disabled,
            on_click=lambda: st.session_state.update(selected_doses=[]),
        )

    with st.expander("Choose dose(s)", expanded=False):
        if dose_disabled:
            st.caption("Select plate(s) first.")
        else:
            q_dose = st.text_input(
                "Search doses", key="dose_search",
                label_visibility="collapsed", placeholder="Search doses"
            )
            dose_opts = [o for o in available_doses if not q_dose or q_dose.lower() in o.lower()]
            for o in dose_opts:
                st.checkbox(o, key=f"dose_opt_{o}", value=(o in st.session_state.selected_doses))
            if st.button("Apply doses", key="apply_doses", disabled=False):
                st.session_state.selected_doses = [
                    o for o in available_doses if st.session_state.get(f"dose_opt_{o}", False)
                ]

    selected_doses = st.session_state["selected_doses"]
    st.caption("Selected doses: " + (", ".join(selected_doses) if selected_doses else "none"))

    st.header("3) Plot options")
    show_oof = st.checkbox("Show %OOF chart", value=False, key="show_oof")

    if in_vivo_mode:
        st.caption("In vivo mode: one bar per **(Group letter, gRNA, Dose)**; replicates = animals.")
        vivo_group_order = st.radio(
            "Bar order", ["As input order", "By dose: High → Low", "By dose: Low → High"],
            index=0,
            help="Sort gRNA cohorts by dose; ties keep input order."
        )
    else:
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
clip_nonnegative_inplace(
    fdf_plot,
    [indel_col, oof_col, rin_col, raa_col, "alignment%"]
)

# ---------------- Aggregate for Indel% ----------------
if in_vivo_mode:
    # one bar per (GroupLetter, gRNA, Dose); animals are replicates
    agg_indel = (
        fdf_plot.groupby(["_Group", "_gRNA", "_Dose", "_DoseVal"], dropna=False)[indel_col]
                .agg(mean="mean", std="std", count="count")
                .reset_index()
    )
    # x-axis label: A-<gRNA>-<dose>
    agg_indel["_DescLabel"] = (
        agg_indel["_Group"].astype(str) + "-" +
        agg_indel["_gRNA"].astype(str) + "-" +
        agg_indel["_Dose"].astype(str)
    )
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
    labels_input_order = list(pd.unique(
        (fdf_plot["_Group"].astype(str) + "-" + fdf_plot["_gRNA"].astype(str) + "-" + fdf_plot["_Dose"].astype(str))
    ))
    # default: as input
    if vivo_group_order == "By dose: High → Low":
        # order by dose descending; ties keep input
        dmap = agg_indel.set_index("_DescLabel")["_DoseVal"].to_dict()
        labels_sorted = sorted(labels_input_order, key=lambda x: (dmap.get(x, -np.inf),), reverse=True)
    elif vivo_group_order == "By dose: Low → High":
        dmap = agg_indel.set_index("_DescLabel")["_DoseVal"].to_dict()
        labels_sorted = sorted(labels_input_order, key=lambda x: (dmap.get(x, np.inf),))
    else:
        labels_sorted = labels_input_order
    x_categories = [lbl for lbl in labels_sorted if lbl in set(agg_indel["_DescLabel"].astype(str))]
else:
    # in vitro path (unchanged)
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

# ----- Category assignment & colors -----
desc_vals = agg_indel["_DescLabel"].astype(str).tolist()
present = set(desc_vals)
if 'x_categories' not in locals() or not x_categories:
    x_categories = list(dict.fromkeys(desc_vals))
else:
    x_categories = [x for x in x_categories if x in present] or list(dict.fromkeys(desc_vals))
agg_indel["_DescLabel"] = pd.Categorical(agg_indel["_DescLabel"], categories=x_categories, ordered=True)

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
    # shade by dose within each GROUP letter
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
            lbl = f"{row['_Group']}-{row['_gRNA']}-{row['_Dose']}"
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

fallback = px.colors.qualitative.Plotly
for lbl in pd.unique(agg_indel["_DescLabel"].astype(str)):
    if lbl not in color_discrete_map:
        color_discrete_map[lbl] = fallback[len(color_discrete_map) % len(fallback)]

# ---------------- Plots ----------------
if "png_w" not in locals(): png_w = 1200
if "lock_ratio" not in locals(): lock_ratio = True
if "png_h" not in locals(): png_h = int(round(png_w * 9 / 16)) if lock_ratio else 675
if "png_scale" not in locals(): png_scale = 2

# ---------------- Plot with bars + SD ----------------
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
        xaxis_title=None,  # or a string if you want a title
        yaxis_title="Mean Indel% (± SD)",
        showlegend=False,
        xaxis=dict(
            tickfont=dict(color="black"),
            title=dict(font=dict(color="black"))
        ),
        yaxis=dict(
            tickfont=dict(color="black"),
            title=dict(font=dict(color="black"))
        ),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    config = {
        "toImageButtonOptions": {"format": "png", "filename": "indel_bars",
                                 "width": int(png_w), "height": int(png_h), "scale": int(png_scale)},
        "displaylogo": False,
    }
    st.plotly_chart(fig, use_container_width=True, config=config)

# ---------------- Optional %OOF plot (same order/colors) ----------------
if show_oof and oof_col and (oof_col in fdf_plot.columns):
    if in_vivo_mode:
        agg_oof = (
            fdf_plot.groupby(["_Group", "_gRNA", "_Dose", "_DoseVal"], dropna=False)[oof_col]
                    .agg(mean="mean", std="std", count="count")
                    .reset_index()
        )
        agg_oof["_DescLabel"] = (
            agg_oof["_Group"].astype(str) + "-" +
            agg_oof["_gRNA"].astype(str) + "-" +
            agg_oof["_Dose"].astype(str)
        )
    else:
        agg_oof = (
            fdf_plot.groupby(["_Group", "_Dose", "_DoseVal"], dropna=False)[oof_col]
                    .agg(mean="mean", std="std", count="count")
                    .reset_index()
        )
        agg_oof["_DescLabel"] = agg_oof["_Group"].astype(str) + "-" + agg_oof["_Dose"].astype(str)

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
            xaxis_title=None, yaxis_title="Mean %OOF (± SD)", showlegend=False,
        xaxis=dict(
            tickfont=dict(color="black"),
            title=dict(font=dict(color="black"))
        ),
        yaxis=dict(
            tickfont=dict(color="black"),
            title=dict(font=dict(color="black"))
        ),
        plot_bgcolor="white",
        paper_bgcolor="white"
        )
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
clip_nonnegative_inplace(
    rows_df,
    ["Indel%", "%OOF", "Reads_in_input", "Reads_aligned_all_amplicons", "alignment%"]
)

# ---------- GraphPad tables ----------
# In vitro: keep previous wide tables
# ---------- In vivo: gRNA table WITH replicates (Indel + %OOF, PBS rows last) ----------
if in_vivo_mode:
    # 1) Extract (Group, gRNA) order from plotted x-axis (labels like "A-<gRNA>-<dose>")
    def _label_to_group_grna(lbl: str):
        parts = str(lbl).split("-")
        if len(parts) < 3:
            return ("", "")
        group = parts[0].strip()
        grna  = "-".join(parts[1:-1]).strip()
        return (group, grna)

    pair_order_plot = []
    for lbl in x_categories:
        key = _label_to_group_grna(lbl)
        if key not in pair_order_plot:
            pair_order_plot.append(key)  # e.g., ("A", "XXX")

    # 2) Original input order for (Group, gRNA) (used to tie-break PBS order)
    seen = set()
    pair_order_input = []
    for _, row in fdf_plot.iterrows():
        key = (str(row["_Group"]), str(row["_gRNA"]))
        if key not in seen:
            seen.add(key)
            pair_order_input.append(key)

    # 3) Dose column order (same logic as before)
    if locals().get("vivo_group_order", "As input order") == "By dose: Low \u2192 High":
        dose_columns_order = dose_low_to_high
    else:
        dose_columns_order = dose_high_to_low

    # 4) Replicate sorting (unchanged helper)
    def _rep_sort_key(sub):
        if "_RepNum" in sub.columns and sub["_RepNum"].notna().any():
            return sub.sort_values(by=["_RepNum", sid_col])
        elif "_Well" in sub.columns and sub["_Well"].notna().any():
            def _well_key(w):
                s = str(w)
                m = re.match(r"^([A-Ha-h])\s*[-:]?\s*(\d{1,2})$", s)
                if m:
                    return (ord(m.group(1).upper()) - ord('A'), int(m.group(2)))
                return (99, 99, s)
            return sub.sort_values(by=["_Well", sid_col], key=lambda s: s.map(_well_key) if s.name == "_Well" else s)
        else:
            return sub.sort_values(by=[sid_col])

    # 5) Max replicates across (Group, gRNA, Dose)
    max_reps = (
        fdf_plot.groupby(["_Group", "_gRNA", "_Dose"]).size().max()
        if not fdf_plot.empty else 1
    )
    max_reps = int(max_reps) if pd.notna(max_reps) else 1

    # 6) Build replicate columns: one block per dose, then Indel/OOF by rep
    cols = []
    for d in dose_columns_order:
        for i in range(1, max_reps + 1):
            cols.append(f"{d}-Indel-rep{i}")
    if oof_col and oof_col in fdf_plot.columns:
        for d in dose_columns_order:
            for i in range(1, max_reps + 1):
                cols.append(f"{d}-OOF-rep{i}")

    def _fmt_blank(x):
        try:
            return f"{float(x):.1f}"
        except Exception:
            return "" if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)

    # Small helpers
    def _is_pbs(d):
        ds = str(d).strip().lower()
        return (ds == "" or ds == "0" or ds == "0.0" or ds == "pbs")

    # 7) Build rows keyed by (Group, gRNA)
    nonpbs_rows, pbs_rows = [], []

    for (G, g) in pair_order_plot:
        doses_here = pd.unique(
            fdf_plot[(fdf_plot["_Group"] == G) & (fdf_plot["_gRNA"] == g)]["_Dose"].astype(str)
        )
        has_pbs = any(_is_pbs(d) for d in doses_here)

        # PBS row (per group)
        if has_pbs:
            row_pbs = {"gRNA": f"PBS-{G}-{g}"}
            for d in dose_columns_order:
                sub = fdf_plot[(fdf_plot["_Group"] == G) &
                            (fdf_plot["_gRNA"] == g) &
                            (fdf_plot["_Dose"].astype(str) == str(d))]
                if _is_pbs(d):
                    sub = _rep_sort_key(sub)
                    vals_indel = [_fmt_blank(v) for v in sub[indel_col].tolist()]
                    for i in range(1, max_reps + 1):
                        row_pbs[f"{d}-Indel-rep{i}"] = (vals_indel[i-1] if i-1 < len(vals_indel) else "")
                    if oof_col and oof_col in fdf_plot.columns:
                        vals_oof = [_fmt_blank(v) for v in sub[oof_col].tolist()]
                        for i in range(1, max_reps + 1):
                            row_pbs[f"{d}-OOF-rep{i}"] = (vals_oof[i-1] if i-1 < len(vals_oof) else "")
                else:
                    # empty cells for non-PBS doses on the PBS row
                    for i in range(1, max_reps + 1):
                        row_pbs[f"{d}-Indel-rep{i}"] = ""
                        if oof_col and oof_col in fdf_plot.columns:
                            row_pbs[f"{d}-OOF-rep{i}"] = ""
            pbs_rows.append(((G, g), row_pbs))

        # Non-PBS row (per group) — only if at least one non-PBS dose exists
        has_nonpbs = any(
            (not _is_pbs(d)) and
            (not fdf_plot[(fdf_plot["_Group"] == G) &
                        (fdf_plot["_gRNA"] == g) &
                        (fdf_plot["_Dose"].astype(str) == str(d))].empty)
            for d in dose_columns_order
        )
        if has_nonpbs:
            row = {"gRNA": f"{G}-{g}"}
            for d in dose_columns_order:
                sub = fdf_plot[(fdf_plot["_Group"] == G) &
                            (fdf_plot["_gRNA"] == g) &
                            (fdf_plot["_Dose"].astype(str) == str(d))]
                if _is_pbs(d):
                    # leave blanks for PBS on the non-PBS row
                    for i in range(1, max_reps + 1):
                        row[f"{d}-Indel-rep{i}"] = ""
                        if oof_col and oof_col in fdf_plot.columns:
                            row[f"{d}-OOF-rep{i}"] = ""
                    continue

                sub = _rep_sort_key(sub)
                vals_indel = [_fmt_blank(v) for v in sub[indel_col].tolist()]
                for i in range(1, max_reps + 1):
                    row[f"{d}-Indel-rep{i}"] = (vals_indel[i-1] if i-1 < len(vals_indel) else "")
                if oof_col and oof_col in fdf_plot.columns:
                    vals_oof = [_fmt_blank(v) for v in sub[oof_col].tolist()]
                    for i in range(1, max_reps + 1):
                        row[f"{d}-OOF-rep{i}"] = (vals_oof[i-1] if i-1 < len(vals_oof) else "")
            nonpbs_rows.append(row)
        # else: skip creating a blank non-PBS row entirely

    # 8) Sort PBS rows by original input (Group, gRNA) order
    pbs_rows_sorted = []
    for key in pair_order_input:
        for (k2, row) in pbs_rows:
            if key == k2:
                pbs_rows_sorted.append(row)

    # Final table: non-PBS (plot order) then PBS (input order)
    gp_table = pd.DataFrame(nonpbs_rows + pbs_rows_sorted, columns=["gRNA"] + cols)

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
            # write sheets
            gp_table.to_excel(writer, index=False, sheet_name="group_gRNA_by_dose_reps")
            rows_df.to_excel(writer, index=False, sheet_name="rows_used")

            # style sheets
            style_sheet(writer, excel_engine, "group_gRNA_by_dose_reps", gp_table)
            style_sheet(writer, excel_engine, "rows_used", rows_df)

        st.download_button(
            "Download Excel (.xlsx) — rows + gRNA table (Indel + OOF, replicates)",
            data=buf.getvalue(),
            file_name=excel_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("Install `openpyxl` or `xlsxwriter` to enable .xlsx downloads.")

    # ---------- Preview ----------
    st.subheader("gRNA table (in vivo, replicates with Indel + OOF)")
    st.dataframe(gp_table, use_container_width=True, hide_index=True)

else:
    # ------ original GraphPad wide tables for in vitro ------
    # Build order for GraphPad tables from plotted categories
    groups_in_order = []
    for lbl in x_categories:
        g = str(lbl).rsplit("-", 1)[0]
        if g not in groups_in_order:
            groups_in_order.append(g)

    doses_high_to_low = list(reversed(dose_low_to_high))

    def well_sort_key(w):
        s = str(w)
        m = re.match(r"^([A-Ha-h])\s*[-:]?\s*(\d{1,2})$", s)
        if m:
            return (ord(m.group(1).upper()) - ord('A'), int(m.group(2)))
        return (99, 99, s)

    gp_key = "_Group"
    src_df = fdf_plot.copy()
    if "_Well" in src_df.columns and src_df["_Well"].notna().any():
        fdf_sorted = src_df.sort_values(
            by=[gp_key, "_Dose", "_Well"],
            key=lambda s: s.map(well_sort_key) if s.name == "_Well" else s
        )
    else:
        fdf_sorted = src_df.sort_values(by=[gp_key, "_Dose", sid_col])
    fdf_sorted["_rep_idx"] = fdf_sorted.groupby([gp_key, "_Dose"]).cumcount() + 1

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
                vals = [fmt1(v) for v in subgd.sort_values(["_rep_idx"])[metric_src_col].tolist()]
                for i in range(1, max_reps_global + 1):
                    row[f"{d}_{metric_label}_rep{i}"] = (vals[i-1] if i-1 < len(vals) else np.nan)
            rows.append(row)
        return pd.DataFrame(rows, columns=["Description"] + cols)

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
                vals = [fmt1(v) for v in subdg.sort_values(["_rep_idx"])[metric_src_col].tolist()]
                for i in range(1, max_reps_global + 1):
                    row[f"{g}_{metric_label}_rep{i}"] = (vals[i-1] if i-1 < len(vals) else np.nan)
            rows.append(row)
        return pd.DataFrame(rows, columns=["Dose"] + cols)

    def make_type1_mean(metric_label: str, metric_src_col: str) -> pd.DataFrame:
        """
        Rows = Group, Cols = Dose (means over replicates), same order as Type 1 table.
        """
        cols = [f"{d}_{metric_label}_mean" for d in doses_high_to_low]
        rows = []
        for g in groups_in_order:
            row = {"Description": g}
            subg = fdf_sorted[fdf_sorted[gp_key] == g]
            for d in doses_high_to_low:
                subgd = subg[subg["_Dose"].astype(str) == d]
                if subgd.empty or metric_src_col not in subgd.columns:
                    row[f"{d}_{metric_label}_mean"] = np.nan
                else:
                    row[f"{d}_{metric_label}_mean"] = fmt1(subgd[metric_src_col].mean())
            rows.append(row)
        return pd.DataFrame(rows, columns=["Description"] + cols)

    type1_indel = (make_type1("Indel%", indel_col)
                   if indel_col in fdf_sorted.columns else pd.DataFrame({"Description": groups_in_order}))
    type1_df = type1_indel
    if oof_col and oof_col in fdf_sorted.columns:
        type1_oof = make_type1("%OOF", oof_col)
        type1_df = type1_indel.merge(type1_oof.drop(columns=["Description"]),
                                     left_index=True, right_index=True)

    type2_indel = (make_type2("Indel%", indel_col)
                   if indel_col in fdf_sorted.columns else pd.DataFrame({"Dose": doses_high_to_low}))
    type2_df = type2_indel
    if oof_col and oof_col in fdf_sorted.columns:
        type2_oof = make_type2("%OOF", oof_col)
        type2_df = type2_indel.merge(type2_oof.drop(columns=["Dose"]),
                                     left_index=True, right_index=True)

# --- Type 1 MEAN (rows=Group, cols=Dose means) ---
    type1_mean_indel = (make_type1_mean("Indel%", indel_col)
                        if indel_col in fdf_sorted.columns else pd.DataFrame({"Description": groups_in_order}))
    type1_mean_df = type1_mean_indel
    if oof_col and oof_col in fdf_sorted.columns:
        type1_mean_oof = make_type1_mean("%OOF", oof_col)
        type1_mean_df = type1_mean_indel.merge(
            type1_mean_oof.drop(columns=["Description"]),
            left_index=True, right_index=True
        )

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
            type1_df.to_excel(writer, index=False, sheet_name="graphpad_type1_groups")
            type2_df.to_excel(writer, index=False, sheet_name="graphpad_type2_doses")
            type1_mean_df.to_excel(writer, index=False, sheet_name="graphpad_type1_groups_mean")
            rows_df.to_excel(writer, index=False, sheet_name="rows_used")

            # style each sheet
            style_sheet(writer, excel_engine, "graphpad_type1_groups", type1_df)
            style_sheet(writer, excel_engine, "graphpad_type2_doses", type2_df)
            style_sheet(writer, excel_engine, "graphpad_type1_groups_mean", type1_mean_df)
            style_sheet(writer, excel_engine, "rows_used", rows_df)

        st.download_button(
            "Download Excel (.xlsx) — rows + both GraphPad tables",
            data=buf.getvalue(),
            file_name=excel_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("Install `openpyxl` or `xlsxwriter` to enable .xlsx downloads.")

    st.subheader("GraphPad-friendly wide table (rounded to 1 decimal)")
    which_gp = st.radio(
        "Layout to preview",
        options=[
            "Type 1: rows=Group, cols=Dose",
            "Type 2: rows=Dose, cols=Group",
            "Type 1 (MEAN): rows=Group, cols=Dose means"  # <— NEW
        ],
        index=0, horizontal=True,
    )

    if which_gp.startswith("Type 1 (MEAN)"):
        st.dataframe(type1_mean_df, use_container_width=True, hide_index=True)
    elif which_gp.startswith("Type 1"):
        st.dataframe(type1_df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(type2_df, use_container_width=True, hide_index=True)

# ---------- Rows used ----------
st.subheader("Rows used for this plot (filtered)")
st.dataframe(rows_df, use_container_width=True, hide_index=True)
