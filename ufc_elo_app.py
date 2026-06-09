"""Streamlit front-end for UFC peak Elo rankings."""

from __future__ import annotations

import hashlib
import re
import unicodedata

import altair as alt
import networkx as nx
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
import streamlit_shadcn_ui as ui
from streamlit_echarts import st_echarts, JsCode
from streamlit_option_menu import option_menu

from elo_engine import (
    UFCEloEngine,
    BASE_RATING,
    K_FACTOR,
    FINISH_MULTIPLIER,
    DECISION_MULTIPLIER,
)
from cardio_score import compute_baselines, score_fighter

FIGHT_RESULTS_CSV = "ufc_fight_results.csv"
EVENT_DETAILS_CSV = "ufc_event_details.csv"
FIGHTER_TOTT_CSV = "ufc_fighter_tott.csv"
FIGHT_STATS_CSV = "ufc_fight_stats.csv"
FIGHTER_PROFILES_CSV = "Fighter_Profiles.csv"

OUTCOME_MAP = {
    "W/L": "W",
    "L/W": "L",
    "D/D": "D",   # filtered out below per spec
    "NC/NC": "NC",
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().upper().replace(" ", "") for c in df.columns]
    # Common rename: WEIGHTCLASS variants land on WEIGHTCLASS already via the
    # space-stripping above. Same for TIMEFORMAT, etc.
    return df


# ---------------------------------------------------------------------------
# Weight-class cleaning (ufc_fight_results.csv)
# ---------------------------------------------------------------------------

# Canonical UFC divisions, ordered lightest → heaviest. Catch Weight is now
# a valid bucket since the engine no longer filters those bouts out.
CANONICAL_WEIGHT_CLASSES: tuple[str, ...] = (
    "Strawweight",
    "Flyweight",
    "Bantamweight",
    "Featherweight",
    "Lightweight",
    "Welterweight",
    "Middleweight",
    "Light Heavyweight",
    "Heavyweight",
    "Open Weight",
    "Catch Weight",
)


def _clean_weight_class(raw: object) -> tuple[str, str, bool]:
    """Normalise a raw WEIGHTCLASS string into (clean, title_type, is_womens).

    - Matching is case- and whitespace-insensitive.
    - Strips the keywords 'Bout', 'Title Bout', 'Interim Title Bout',
      'Tournament', the 'UFC' organisation prefix, and the 'Women's'
      prefix.
    - title_type is one of 'None', 'Title', 'Interim'.
    - is_womens is True iff the raw string referenced a women's division.
      Women's divisions get a 'Womens ' prefix on the returned canonical
      name (e.g. 'Womens Bantamweight') so they sort and group as their
      own division throughout the app.
    - If the cleaned token doesn't match any canonical division, the raw
      string is kept verbatim and a warning is printed to the terminal.
    """
    if not isinstance(raw, str) or not raw.strip():
        return ("", "None", False)

    s = re.sub(r"\s+", " ", raw).strip()
    s_lower = s.lower()

    # Title detection — interim is more specific so it's checked first.
    if "interim" in s_lower and "title" in s_lower:
        title_type = "Interim"
    elif "title" in s_lower:
        title_type = "Title"
    else:
        title_type = "None"

    is_womens = "women" in s_lower

    # Strip prefix + suffix tokens. Order matters: longest first so
    # 'interim title bout' is removed before 'title bout' / 'bout'.
    cleaned = s_lower
    cleaned = re.sub(r"women['’]?s\s*", "", cleaned)
    for token in (
        "interim title bout",
        "title bout",
        "tournament",
        "bout",
        "ufc",  # organisation prefix sometimes attached to the weight class
    ):
        cleaned = cleaned.replace(token, "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    canonical_lower = {c.lower(): c for c in CANONICAL_WEIGHT_CLASSES}
    if cleaned in canonical_lower:
        clean_name = canonical_lower[cleaned]
        if is_womens:
            clean_name = f"Womens {clean_name}"
        return (clean_name, title_type, is_womens)

    print(
        f"[load_fights] Unknown weight class: {raw!r} "
        f"(after cleaning: {cleaned!r}) — keeping raw value"
    )
    return (s, title_type, is_womens)


def _format_wc_with_title(clean_class: object, title_type: object) -> str:
    """Append a '(Title Fight)' / '(Interim Title Fight)' suffix when applicable."""
    wc = "" if clean_class is None else str(clean_class)
    tt = "" if title_type is None else str(title_type)
    if tt == "Title":
        return f"{wc} (Title Fight)" if wc else "Title Fight"
    if tt == "Interim":
        return f"{wc} (Interim Title Fight)" if wc else "Interim Title Fight"
    return wc


# Cache-bust counter for load_fights(). Bump this any time the cleaning /
# transformation logic inside load_fights changes (or any helper it calls,
# such as _clean_weight_class). @st.cache_data keys on function arguments,
# so threading this version through the signature forces a rebuild on the
# next rerun — no manual "Clear cache" needed.
_LOAD_FIGHTS_VERSION = 2


@st.cache_data(show_spinner="Loading UFC fight data…")
def load_fights(_version: int = _LOAD_FIGHTS_VERSION) -> pd.DataFrame:
    results = _normalise_columns(pd.read_csv(FIGHT_RESULTS_CSV))
    events = _normalise_columns(pd.read_csv(EVENT_DETAILS_CSV))

    required_results = {"BOUT", "OUTCOME", "WEIGHTCLASS"}
    missing = required_results - set(results.columns)
    if missing:
        st.error(f"`{FIGHT_RESULTS_CSV}` is missing columns: {sorted(missing)}. Found: {list(results.columns)}")
        return pd.DataFrame()

    if "DATE" not in events.columns:
        st.error(f"`{EVENT_DETAILS_CSV}` is missing a DATE column. Found: {list(events.columns)}")
        return pd.DataFrame()

    events["DATE"] = pd.to_datetime(events["DATE"], errors="coerce")

    # Pick a join key. Prefer EVENT, fall back to URL.
    join_key = None
    for candidate in ("EVENT", "URL"):
        if candidate in results.columns and candidate in events.columns:
            results[candidate] = results[candidate].astype(str).str.strip()
            events[candidate] = events[candidate].astype(str).str.strip()
            join_key = candidate
            break
    if join_key is None:
        st.error(
            "Could not find a shared join column between the two CSVs.\n"
            f"Results cols: {list(results.columns)}\nEvents cols: {list(events.columns)}"
        )
        return pd.DataFrame()

    df = results.merge(events[[join_key, "DATE"]], on=join_key, how="left")

    matched_dates = df["DATE"].notna().sum()
    if matched_dates == 0:
        st.error(
            f"No rows matched between results and events on `{join_key}`.\n"
            f"Sample results {join_key}: {results[join_key].head(3).tolist()}\n"
            f"Sample events  {join_key}: {events[join_key].head(3).tolist()}"
        )
        return pd.DataFrame()

    df = df.dropna(subset=["BOUT", "OUTCOME", "WEIGHTCLASS"])

    # Disregard D/D outcomes. Catch Weight bouts are now kept (the
    # cleaning pass below normalises them into the 'Catch Weight' bucket).
    df = df[df["OUTCOME"].astype(str).str.strip() != "D/D"]

    # Clean WEIGHTCLASS into a canonical division and extract title-fight
    # type ('None' / 'Title' / 'Interim') + a women's-division flag.
    cleaned = df["WEIGHTCLASS"].apply(_clean_weight_class)
    df = df.assign(
        WEIGHTCLASS=cleaned.apply(lambda t: t[0]),
        TITLE_TYPE=cleaned.apply(lambda t: t[1]),
        IS_WOMENS=cleaned.apply(lambda t: t[2]),
    )
    df = df[df["WEIGHTCLASS"].astype(str).str.strip() != ""]

    if df.empty:
        st.warning("All fights were filtered out. Check OUTCOME / WEIGHTCLASS values.")
        return df

    # Flexible split: handles "vs.", "vs", and odd whitespace/casing.
    bout_split = (
        df["BOUT"].astype(str)
        .str.split(r"\s+vs\.?\s+", n=1, expand=True, regex=True)
    )
    if bout_split.shape[1] < 2:
        st.error(
            "Could not split BOUT into two fighters. Sample BOUT values: "
            f"{df['BOUT'].head(3).tolist()}"
        )
        return pd.DataFrame()

    df = df.assign(
        FIGHTER_A=bout_split[0].str.strip(),
        FIGHTER_B=bout_split[1].str.strip(),
    )
    df = df.dropna(subset=["FIGHTER_A", "FIGHTER_B"])
    df = df[(df["FIGHTER_A"] != "") & (df["FIGHTER_B"] != "")]

    df["OUTCOME"] = df["OUTCOME"].astype(str).str.strip().map(OUTCOME_MAP).fillna("NC")
    df = df[df["OUTCOME"] != "NC"]

    df["METHOD"] = df["METHOD"].fillna("") if "METHOD" in df.columns else ""

    # Drop rows that didn't get a date — we need chronological ordering.
    df = df.dropna(subset=["DATE"])

    df = df.sort_values("DATE").reset_index(drop=True)
    keep = [
        "DATE", "FIGHTER_A", "FIGHTER_B", "OUTCOME", "METHOD",
        "WEIGHTCLASS", "TITLE_TYPE", "IS_WOMENS",
    ]
    for extra in ("EVENT", "BOUT"):
        if extra in df.columns:
            keep.append(extra)
    return df[keep]


@st.cache_resource(show_spinner="Computing Elo ratings…")
def build_engine(
    k: float = K_FACTOR,
    base: float = BASE_RATING,
    finish_mult: float = FINISH_MULTIPLIER,
    decision_mult: float = DECISION_MULTIPLIER,
) -> UFCEloEngine:
    # Defaults pull from the ELO CONFIGURATION block in elo_engine.py.
    # The Settings page can override any of them at runtime. Streamlit's
    # cache_resource keys on the args, so each unique combination is
    # computed once and reused — this fixes the original "no-arg cache
    # never invalidates" gotcha.
    engine = UFCEloEngine(
        k=k,
        base=base,
        finish_multiplier=finish_mult,
        decision_multiplier=decision_mult,
    )
    engine.run(load_fights())
    return engine


@st.cache_resource(show_spinner="Computing cardio baselines…")
def build_cardio_baselines():
    """Frozen dataset-wide baselines for the cardio score.

    Cached as a resource so the per-component mu/sigma and per-fighter
    lambda_bar dictionary are computed once and reused across reruns.
    Returns None if the underlying CSVs are missing.
    """
    stats_by_round = load_stats_by_round()
    try:
        results = pd.read_csv(FIGHT_RESULTS_CSV)
    except FileNotFoundError:
        return None
    return compute_baselines(stats_by_round, results)


def render_top_n(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    out = df.head(n).copy()
    out.insert(0, "Rank", range(1, len(out) + 1))
    out["Peak Elo"] = out["Peak Elo"].round(1)
    return out[["Rank", "Fighter", "Peak Elo"]]


# Stable categorical palette used to colour fighter lines on the trajectory
# chart. A hash of the fighter's name picks an index, so the same fighter
# always gets the same colour across reloads.
FIGHTER_PALETTE: tuple[str, ...] = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#aec7e8",
    "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
    "#f7b6d2", "#dbdb8d", "#9edae5",
)


def fighter_color(name: str) -> str:
    """Deterministic colour for a fighter, derived from a hash of their name."""
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return FIGHTER_PALETTE[int(digest, 16) % len(FIGHTER_PALETTE)]


# ---------------------------------------------------------------------------
# Global styling (Bebas Neue headings) + per-fighter accent banner helpers.
# ---------------------------------------------------------------------------

def inject_global_css() -> None:
    """Inject Bebas Neue + accent-banner styles. Called once in main().

    Bebas Neue is a condensed display font (gorgeous for headings,
    unreadable for body text), so it's scoped to headings + Streamlit's
    metric labels + tab labels only. Body text and tables stay in the
    config.toml-configured sans-serif.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap');
        h1, h2, h3, h4, h5, h6,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] p,
        .tot-title,
        .stTabs [data-baseweb="tab"] {
            font-family: 'Bebas Neue', 'Arial Narrow', sans-serif !important;
            letter-spacing: 0.04em;
        }
        h1 { font-size: 3rem !important; }
        h2 { font-size: 2.2rem !important; }
        h3 { font-size: 1.7rem !important; }
        .tot-title {
            font-size: 1.5rem;
            margin: 0 0 0.6rem 0;
            letter-spacing: 0.05em;
        }
        .fighter-banner {
            height: 6px;
            width: 100%;
            margin: 0 0 1.2rem 0;
            border-radius: 3px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_fighter_banner(color: str) -> None:
    """Thin gradient stripe across the top of a section, tinted fighter colour."""
    st.markdown(
        f"<div class='fighter-banner' style='background: linear-gradient("
        f"90deg, {color} 0%, {color}88 50%, transparent 100%);'></div>",
        unsafe_allow_html=True,
    )


def render_two_fighter_banner(color_a: str, color_b: str) -> None:
    """Gradient stripe blending two fighters' colours (Fight Finder)."""
    st.markdown(
        f"<div class='fighter-banner' style='background: linear-gradient("
        f"90deg, {color_a} 0%, {color_a}cc 30%, {color_b}cc 70%, "
        f"{color_b} 100%);'></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Fighter bio parsing & formatting (ufc_fighter_tott.csv)
# ---------------------------------------------------------------------------

def _normalise_name(name: object) -> str:
    """Normalise a fighter name for cross-CSV matching.

    Strips whitespace, collapses internal whitespace, NFKC-normalises
    unicode (so accented characters compare equal across files), and
    casefolds. The result is only used as a dictionary key — the
    original display name is preserved separately.
    """
    if not isinstance(name, str):
        return ""
    n = unicodedata.normalize("NFKC", name).strip()
    n = re.sub(r"\s+", " ", n)
    return n.casefold()


def _parse_height(s: object) -> float | None:
    """Parse a height string like '5\\' 11"' into total inches."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s in ("", "--"):
        return None
    m = re.match(r"(\d+)'\s*(\d+)", s)
    if not m:
        return None
    return float(int(m.group(1)) * 12 + int(m.group(2)))


def _parse_lbs(s: object) -> float | None:
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s in ("", "--"):
        return None
    m = re.match(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def _parse_inches(s: object) -> float | None:
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s in ("", "--"):
        return None
    m = re.match(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def _parse_stance(s: object) -> str | None:
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s in ("", "--"):
        return None
    return s


def _parse_dob(s: object) -> pd.Timestamp | None:
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s in ("", "--"):
        return None
    ts = pd.to_datetime(s, errors="coerce")
    return None if pd.isna(ts) else ts


def _format_height(inches: float | None) -> str:
    if inches is None or pd.isna(inches):
        return "—"
    feet = int(inches) // 12
    rem = int(inches) % 12
    cm = round(inches * 2.54)
    return f"{feet}'{rem}\" ({cm} cm)"


def _format_weight(lbs: float | None) -> str:
    if lbs is None or pd.isna(lbs):
        return "—"
    kg = round(lbs * 0.45359237, 1)
    return f"{lbs:.0f} lbs ({kg} kg)"


def _format_reach(inches: float | None) -> str:
    if inches is None or pd.isna(inches):
        return "—"
    cm = round(inches * 2.54)
    return f"{inches:.0f}\" ({cm} cm)"


def _format_age_today(dob: pd.Timestamp | None) -> str:
    """Live age in years, computed from today's date."""
    if dob is None or pd.isna(dob):
        return "—"
    today = pd.Timestamp.today().normalize()
    years = today.year - dob.year - (
        (today.month, today.day) < (dob.month, dob.day)
    )
    return str(years)


@st.cache_data(show_spinner="Loading fighter bios…")
def load_bios() -> dict:
    """Map normalised fighter name → bio dict.

    Returns an empty dict (with a warning) if the bio CSV can't be loaded.
    Each bio is { name, height_in, weight_lbs, reach_in, stance, dob }.
    """
    try:
        df = _normalise_columns(pd.read_csv(FIGHTER_TOTT_CSV))
    except FileNotFoundError:
        st.warning(
            f"`{FIGHTER_TOTT_CSV}` not found — bio info will be unavailable."
        )
        return {}

    if "FIGHTER" in df.columns:
        names = df["FIGHTER"].astype(str)
    elif {"FIRST", "LAST"} <= set(df.columns):
        names = (
            df["FIRST"].fillna("").astype(str)
            + " "
            + df["LAST"].fillna("").astype(str)
        ).str.strip()
    else:
        st.warning(
            f"`{FIGHTER_TOTT_CSV}` has no FIGHTER (or FIRST/LAST) column. "
            f"Found: {list(df.columns)}"
        )
        return {}

    # DOB column name varies across scrape variants.
    dob_col = next(
        (c for c in ("DOB", "DATEOFBIRTH", "BIRTHDATE") if c in df.columns),
        None,
    )

    bios: dict = {}
    for name, (_, row) in zip(names, df.iterrows()):
        key = _normalise_name(name)
        if not key:
            continue
        bios[key] = {
            "name": str(name).strip(),
            "height_in": _parse_height(row.get("HEIGHT")),
            "weight_lbs": _parse_lbs(row.get("WEIGHT")),
            "reach_in": _parse_inches(row.get("REACH")),
            "stance": _parse_stance(row.get("STANCE")),
            "dob": _parse_dob(row.get(dob_col)) if dob_col else None,
        }
    return bios


# ---------------------------------------------------------------------------
# Wikipedia profile loading (Fighter_Profiles.csv produced by build_fighter_profiles.py)
# ---------------------------------------------------------------------------

# (column key, display label) pairs rendered in the Wikipedia profile card.
WIKI_BIO_FIELDS: tuple[tuple[str, str], ...] = (
    ("wiki_other_names", "Also known as"),
    ("wiki_birth_place", "Born in"),
    ("wiki_fighting_out_of", "Fighting out of"),
    ("wiki_team", "Team"),
    ("wiki_style", "Style"),
    ("wiki_rank", "Rank"),
    ("wiki_years_active", "Years active"),
)


@st.cache_data(show_spinner="Loading Wikipedia profiles…")
def load_wiki_profiles() -> dict:
    """Map normalised fighter name → dict of Wikipedia-sourced fields, read
    from Fighter_Profiles.csv. Returns an empty dict if the file is missing.
    Re-run `build_fighter_profiles.py` to refresh; restart Streamlit (or
    clear cache) so the app picks up the new CSV."""
    try:
        df = pd.read_csv(FIGHTER_PROFILES_CSV)
    except FileNotFoundError:
        return {}
    if "name" not in df.columns:
        return {}
    if "wiki_matched" in df.columns:
        df = df[
            df["wiki_matched"].astype(str).str.lower().isin(["true", "1", "yes"])
        ]
    keep = [k for k, _ in WIKI_BIO_FIELDS] + ["wiki_url", "wiki_title"]
    out: dict = {}
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        info: dict = {}
        for col in keep:
            if col in df.columns:
                v = row.get(col)
                if pd.notna(v) and str(v).strip() not in ("", "nan", "None"):
                    info[col] = str(v).strip()
        if info:
            out[_normalise_name(name)] = info
    return out


def render_wiki_profile_card(wiki_info: dict | None) -> None:
    """Bordered card showing Wikipedia bio fields beneath the Tale of the Tape.
    Renders nothing if there's no Wikipedia data for this fighter."""
    if not wiki_info:
        return
    rows = [(label, wiki_info[key]) for key, label in WIKI_BIO_FIELDS if wiki_info.get(key)]
    if not rows:
        return
    with st.container(border=True):
        st.markdown(
            "<div class='tot-title'>\U0001f4d6 WIKIPEDIA PROFILE</div>",
            unsafe_allow_html=True,
        )
        for label, value in rows:
            st.markdown(
                f"<div style='display:flex;gap:1rem;padding:6px 0;"
                f"border-bottom:1px solid rgba(255,255,255,0.06);'>"
                f"<div style='min-width:160px;font-size:0.72rem;"
                f"letter-spacing:0.06em;text-transform:uppercase;"
                f"opacity:0.65;font-weight:700;padding-top:2px;'>{label}</div>"
                f"<div style='flex:1;font-size:0.95rem;'>{value}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        wiki_url = wiki_info.get("wiki_url")
        if wiki_url:
            st.markdown(
                f"<div style='margin-top:0.8rem;font-size:0.75rem;opacity:0.6;'>"
                f"Source: <a href='{wiki_url}' target='_blank' "
                f"style='color:#E03131;text-decoration:none;'>Wikipedia ↗</a>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Per-fight stats parsing & loading (ufc_fight_stats.csv)
# ---------------------------------------------------------------------------

def _parse_x_of_y(value: object) -> tuple[int, int]:
    """Parse 'X of Y' strings (e.g. '12 of 30') into (landed, attempted)."""
    if not isinstance(value, str):
        return 0, 0
    m = re.match(r"\s*(\d+)\s+of\s+(\d+)", value)
    if not m:
        return 0, 0
    return int(m.group(1)), int(m.group(2))


def _parse_ctrl_seconds(value: object) -> int:
    """Parse a 'M:SS' control-time string into total seconds."""
    if not isinstance(value, str):
        return 0
    m = re.match(r"\s*(\d+):(\d+)", value)
    if not m:
        return 0
    return int(m.group(1)) * 60 + int(m.group(2))


def _format_ctrl(seconds: int) -> str:
    """Render seconds as 'M:SS' for display."""
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


# Counted-stat columns: parsed into '<prefix>_landed' / '<prefix>_attempted'
# and summed across rounds when aggregating to per-fight totals.
_STAT_SPLIT_COLS: tuple[tuple[str, str], ...] = (
    ("SIG.STR.", "sig"),
    ("TOTAL STR.", "total"),
    ("TD", "td"),
    ("HEAD", "head"),
    ("BODY", "body"),
    ("LEG", "leg"),
    ("DISTANCE", "distance"),
    ("CLINCH", "clinch"),
    ("GROUND", "ground"),
)


@st.cache_data(show_spinner="Loading per-fight stats…")
def load_stats() -> pd.DataFrame:
    """Load ufc_fight_stats.csv and aggregate to one row per (event, bout, fighter)."""
    try:
        raw = pd.read_csv(FIGHT_STATS_CSV)
    except FileNotFoundError:
        st.warning(
            f"`{FIGHT_STATS_CSV}` not found — per-fight stats will be unavailable."
        )
        return pd.DataFrame()

    raw.columns = [str(c).strip() for c in raw.columns]
    needed = {"EVENT", "BOUT", "FIGHTER", "ROUND"}
    if not needed <= set(raw.columns):
        st.warning(
            f"`{FIGHT_STATS_CSV}` is missing expected columns. "
            f"Found: {list(raw.columns)}"
        )
        return pd.DataFrame()

    for col, prefix in _STAT_SPLIT_COLS:
        if col not in raw.columns:
            raw[f"{prefix}_landed"] = 0
            raw[f"{prefix}_attempted"] = 0
            continue
        parsed = raw[col].apply(_parse_x_of_y)
        raw[f"{prefix}_landed"] = parsed.apply(lambda t: t[0]).astype(int)
        raw[f"{prefix}_attempted"] = parsed.apply(lambda t: t[1]).astype(int)

    if "CTRL" in raw.columns:
        raw["ctrl_sec"] = raw["CTRL"].apply(_parse_ctrl_seconds).astype(int)
    else:
        raw["ctrl_sec"] = 0
    raw["kd"] = pd.to_numeric(raw.get("KD"), errors="coerce").fillna(0).astype(int)
    raw["sub_att"] = pd.to_numeric(raw.get("SUB.ATT"), errors="coerce").fillna(0).astype(int)

    sum_cols = ["kd", "sub_att", "ctrl_sec"] + [
        f"{prefix}_{kind}"
        for _, prefix in _STAT_SPLIT_COLS
        for kind in ("landed", "attempted")
    ]
    grouped = (
        raw.groupby(["EVENT", "BOUT", "FIGHTER"], sort=False)[sum_cols]
        .sum()
        .reset_index()
    )
    grouped["fighter_norm"] = grouped["FIGHTER"].apply(_normalise_name)
    return grouped


def lookup_fight_stats(
    stats: pd.DataFrame,
    event: str,
    bout: str,
    fighter: str,
) -> dict | None:
    """Return per-fight stat totals for one fighter in one bout, or None."""
    if stats.empty or not event or not bout:
        return None
    norm = _normalise_name(fighter)
    match = stats[
        (stats["EVENT"] == event)
        & (stats["BOUT"] == bout)
        & (stats["fighter_norm"] == norm)
    ]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


@st.cache_data(show_spinner="Loading per-round stats…")
def load_stats_by_round() -> pd.DataFrame:
    """Load ufc_fight_stats.csv with one row per (event, bout, fighter, round).

    Same parsing as load_stats() but **without** the per-fight groupby —
    every round stays as its own row so we can render round-by-round
    breakdowns in Fight Finder.
    """
    try:
        raw = pd.read_csv(FIGHT_STATS_CSV)
    except FileNotFoundError:
        return pd.DataFrame()

    raw.columns = [str(c).strip() for c in raw.columns]
    needed = {"EVENT", "BOUT", "FIGHTER", "ROUND"}
    if not needed <= set(raw.columns):
        return pd.DataFrame()

    for col, prefix in _STAT_SPLIT_COLS:
        if col not in raw.columns:
            raw[f"{prefix}_landed"] = 0
            raw[f"{prefix}_attempted"] = 0
            continue
        parsed = raw[col].apply(_parse_x_of_y)
        raw[f"{prefix}_landed"] = parsed.apply(lambda t: t[0]).astype(int)
        raw[f"{prefix}_attempted"] = parsed.apply(lambda t: t[1]).astype(int)

    if "CTRL" in raw.columns:
        raw["ctrl_sec"] = raw["CTRL"].apply(_parse_ctrl_seconds).astype(int)
    else:
        raw["ctrl_sec"] = 0
    raw["kd"] = pd.to_numeric(raw.get("KD"), errors="coerce").fillna(0).astype(int)
    raw["sub_att"] = pd.to_numeric(raw.get("SUB.ATT"), errors="coerce").fillna(0).astype(int)
    # ROUND values look like "Round 1", "Round 2", … — extract the integer.
    # pd.to_numeric on the raw string returns all NaN, which broke every
    # per-round lookup until we extracted the digit explicitly.
    raw["round_num"] = (
        raw["ROUND"].astype(str).str.extract(r"(\d+)", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
        .astype("Int64")
    )
    raw["fighter_norm"] = raw["FIGHTER"].apply(_normalise_name)
    return raw


def lookup_round_stats(
    stats_by_round: pd.DataFrame,
    event: str,
    bout: str,
    fighter: str,
) -> pd.DataFrame:
    """Return a DataFrame of per-round stat rows for one fighter in one bout,
    sorted by round number. Empty DataFrame if no rows match."""
    if stats_by_round.empty or not event or not bout:
        return pd.DataFrame()
    norm = _normalise_name(fighter)
    match = stats_by_round[
        (stats_by_round["EVENT"] == event)
        & (stats_by_round["BOUT"] == bout)
        & (stats_by_round["fighter_norm"] == norm)
    ].sort_values("round_num")
    return match


# ---------------------------------------------------------------------------
# Career-aggregate stats (Fighter Search → Stats & Records dashboard)
# ---------------------------------------------------------------------------

def _method_tag(method: object) -> str:
    """Classify a fight method into KO/TKO, SUB, DEC, or Other."""
    if not isinstance(method, str):
        return "Other"
    m = method.lower()
    if "ko" in m or "tko" in m:
        return "KO/TKO"
    if "sub" in m:
        return "SUB"
    if "dec" in m or "decision" in m:
        return "DEC"
    return "Other"


def _parse_round_time_to_minutes(round_value: object, time_value: object) -> float | None:
    """Convert (final round, M:SS in final round) to total fight minutes.

    Assumes 5-minute rounds (true for >99% of UFC bouts).
    """
    try:
        final_round = int(float(str(round_value).strip()))
    except (ValueError, TypeError):
        return None
    if final_round < 1:
        return None
    m = re.match(r"\s*(\d+):(\d+)", str(time_value))
    if not m:
        return None
    minutes_in_final = int(m.group(1)) + int(m.group(2)) / 60
    return (final_round - 1) * 5.0 + minutes_in_final


@st.cache_data(show_spinner="Loading fight durations…")
def load_fight_durations() -> dict:
    """Map (EVENT, BOUT) → total fight duration in minutes."""
    try:
        df = pd.read_csv(FIGHT_RESULTS_CSV)
    except FileNotFoundError:
        return {}
    df.columns = [str(c).strip().upper().replace(" ", "") for c in df.columns]
    if not {"EVENT", "BOUT", "ROUND", "TIME"} <= set(df.columns):
        return {}
    durations: dict = {}
    for _, row in df.iterrows():
        minutes = _parse_round_time_to_minutes(row.get("ROUND"), row.get("TIME"))
        if minutes is None:
            continue
        key = (str(row["EVENT"]).strip(), str(row["BOUT"]).strip())
        durations[key] = minutes
    return durations


def compute_career_stats(
    fighter: str,
    engine: UFCEloEngine,
    stats_df: pd.DataFrame,
    durations: dict,
) -> dict:
    """Aggregate every available career stat for one fighter.

    Returns a flat dict with raw totals + derived rates / percentages.
    Per-fight-stat fields will be 0 for fighters whose bouts pre-date the
    stats CSV's coverage window (roughly UFC 28+).
    """
    invert = {"W": "L", "L": "W", "D": "D"}
    s = {
        "wins_ko": 0,
        "first_round_finishes": 0,
        "wins_by_method": {"KO/TKO": 0, "DEC": 0, "SUB": 0, "Other": 0},
        "total_wins": 0,
        "sig_landed": 0, "sig_attempted": 0,
        "sig_absorbed": 0, "sig_attempted_against": 0,
        "td_landed": 0, "td_attempted": 0,
        "td_against_landed": 0, "td_against_attempted": 0,
        "kd": 0, "sub_att": 0,
        "distance_landed": 0, "clinch_landed": 0, "ground_landed": 0,
        "head_landed": 0, "body_landed": 0, "leg_landed": 0,
        "total_minutes": 0.0, "fights_with_duration": 0,
    }

    for f in engine.fight_deltas:
        if f["fighter_a"] == fighter:
            opp = f["fighter_b"]
            outcome = f["outcome"]
        elif f["fighter_b"] == fighter:
            opp = f["fighter_a"]
            outcome = invert.get(f["outcome"], f["outcome"])
        else:
            continue

        tag = _method_tag(f.get("method", ""))
        event = f.get("event", "")
        bout = f.get("bout", "")
        duration = durations.get((event, bout))

        if outcome == "W":
            s["total_wins"] += 1
            s["wins_by_method"][tag] = s["wins_by_method"].get(tag, 0) + 1
            if tag == "KO/TKO":
                s["wins_ko"] += 1
            # First-round finish: KO/TKO or SUB win ending inside 5:00.
            if tag in ("KO/TKO", "SUB") and duration is not None and duration <= 5.0:
                s["first_round_finishes"] += 1

        my = lookup_fight_stats(stats_df, event, bout, fighter)
        opp_s = lookup_fight_stats(stats_df, event, bout, opp)

        if my:
            for k in ("sig_landed", "sig_attempted", "td_landed", "td_attempted",
                      "kd", "sub_att", "distance_landed", "clinch_landed",
                      "ground_landed", "head_landed", "body_landed", "leg_landed"):
                s[k] += int(my.get(k, 0) or 0)
        if opp_s:
            s["sig_absorbed"] += int(opp_s.get("sig_landed", 0) or 0)
            s["sig_attempted_against"] += int(opp_s.get("sig_attempted", 0) or 0)
            s["td_against_landed"] += int(opp_s.get("td_landed", 0) or 0)
            s["td_against_attempted"] += int(opp_s.get("td_attempted", 0) or 0)

        if duration is not None:
            s["total_minutes"] += duration
            s["fights_with_duration"] += 1

    # Derived rates & percentages.
    s["sig_acc"] = (s["sig_landed"] / s["sig_attempted"]) if s["sig_attempted"] else None
    s["td_acc"] = (s["td_landed"] / s["td_attempted"]) if s["td_attempted"] else None
    s["sig_def"] = (
        1 - s["sig_absorbed"] / s["sig_attempted_against"]
    ) if s["sig_attempted_against"] else None
    s["td_def"] = (
        1 - s["td_against_landed"] / s["td_against_attempted"]
    ) if s["td_against_attempted"] else None
    if s["total_minutes"] > 0:
        m = s["total_minutes"]
        s["slpm"] = s["sig_landed"] / m
        s["sapm"] = s["sig_absorbed"] / m
        s["td_per15"] = s["td_landed"] / m * 15
        s["sub_per15"] = s["sub_att"] / m * 15
        s["kd_per15"] = s["kd"] / m * 15
    else:
        s["slpm"] = s["sapm"] = s["td_per15"] = None
        s["sub_per15"] = s["kd_per15"] = None
    s["avg_fight_time"] = (
        s["total_minutes"] / s["fights_with_duration"]
        if s["fights_with_duration"] else None
    )
    return s


def _format_minutes_as_mmss(minutes: float | None) -> str:
    if minutes is None:
        return "—"
    total_seconds = int(round(minutes * 60))
    return f"{total_seconds // 60}:{total_seconds % 60:02d}"


def _fmt_rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def _stat_tile_html(label: str, value: str, sub: str | None = None) -> str:
    sub_html = (
        f"<div style='font-size:0.62rem;opacity:0.45;text-transform:uppercase;"
        f"letter-spacing:0.06em;margin-top:2px;'>{sub}</div>"
        if sub else ""
    )
    return (
        f"<div style='padding:4px 0;'>"
        f"<div style='font-size:1.6rem;font-weight:700;line-height:1.1;'>{value}</div>"
        f"<div style='font-size:0.7rem;opacity:0.6;text-transform:uppercase;"
        f"letter-spacing:0.06em;margin-top:4px;'>{label}</div>"
        f"{sub_html}"
        f"</div>"
    )


def render_donut_gauge(percent: float | None, *, color: str = "#E03131", size: int = 170) -> None:
    """A thin donut gauge for accuracy / defense percentages."""
    if percent is None:
        st.markdown(
            f"<div style='height:{size}px;display:flex;align-items:center;"
            f"justify-content:center;color:#888;font-weight:600;'>—</div>",
            unsafe_allow_html=True,
        )
        return
    p = max(0.0, min(1.0, percent))
    fig = go.Figure(
        go.Pie(
            values=[p, 1 - p],
            hole=0.72,
            marker=dict(colors=[color, "rgba(255,255,255,0.08)"], line=dict(width=0)),
            textinfo="none",
            sort=False,
            direction="clockwise",
            rotation=0,
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        annotations=[dict(
            text=f"<b>{p:.0%}</b>",
            x=0.5, y=0.5,
            font=dict(size=26, color="#FAFAFA"),
            showarrow=False,
        )],
        margin=dict(t=8, b=8, l=8, r=8),
        height=size,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_body_silhouette(
    head_data,
    body_data,
    leg_data,
    *,
    color: str = "#E03131",
) -> None:
    """Inline SVG body silhouette with strike counts + percentages overlaid.

    Each *_data is a (count, fraction) tuple where fraction is in [0, 1].
    The fill colour and percent-text colour both use `color` so the
    silhouette matches the fighter's per-fighter palette colour.
    """
    head_n, head_p = head_data
    body_n, body_p = body_data
    leg_n, leg_p = leg_data
    svg = (
        "<div style='display:flex;justify-content:center;padding:4px 0 8px 0;'>"
        "<svg width='240' height='280' viewBox='0 0 240 280' "
        "xmlns='http://www.w3.org/2000/svg'>"
        # Head
        f"<ellipse cx='120' cy='40' rx='22' ry='26' fill='{color}' opacity='0.85'/>"
        # Neck
        f"<rect x='112' y='62' width='16' height='10' fill='{color}' opacity='0.6'/>"
        # Torso
        f"<path d='M 90 72 Q 120 66 150 72 L 158 170 Q 120 180 82 170 Z' "
        f"fill='{color}' opacity='0.6'/>"
        # Arms
        f"<path d='M 88 76 L 65 158 L 75 162 L 95 86 Z' fill='{color}' opacity='0.55'/>"
        f"<path d='M 152 76 L 175 158 L 165 162 L 145 86 Z' fill='{color}' opacity='0.55'/>"
        # Legs
        f"<path d='M 90 170 L 86 268 L 110 268 L 116 170 Z' fill='{color}' opacity='0.45'/>"
        f"<path d='M 124 170 L 130 268 L 154 268 L 150 170 Z' fill='{color}' opacity='0.45'/>"
        # Leader lines + numbers (right side)
        "<line x1='142' y1='40' x2='200' y2='40' stroke='#666' stroke-width='1'/>"
        f"<text x='205' y='36' fill='#FAFAFA' font-size='14' font-weight='700' "
        f"font-family='sans-serif'>{head_n}</text>"
        f"<text x='205' y='52' fill='{color}' font-size='11' "
        f"font-family='sans-serif'>{head_p:.0%}</text>"
        "<line x1='158' y1='120' x2='200' y2='120' stroke='#666' stroke-width='1'/>"
        f"<text x='205' y='116' fill='#FAFAFA' font-size='14' font-weight='700' "
        f"font-family='sans-serif'>{body_n}</text>"
        f"<text x='205' y='132' fill='{color}' font-size='11' "
        f"font-family='sans-serif'>{body_p:.0%}</text>"
        "<line x1='156' y1='220' x2='200' y2='220' stroke='#666' stroke-width='1'/>"
        f"<text x='205' y='216' fill='#FAFAFA' font-size='14' font-weight='700' "
        f"font-family='sans-serif'>{leg_n}</text>"
        f"<text x='205' y='232' fill='{color}' font-size='11' "
        f"font-family='sans-serif'>{leg_p:.0%}</text>"
        # Region labels (left side)
        "<text x='10' y='44' fill='#888' font-size='11' font-weight='600' "
        "font-family='sans-serif'>HEAD</text>"
        "<text x='10' y='124' fill='#888' font-size='11' font-weight='600' "
        "font-family='sans-serif'>BODY</text>"
        "<text x='10' y='224' fill='#888' font-size='11' font-weight='600' "
        "font-family='sans-serif'>LEG</text>"
        "</svg></div>"
    )
    st.markdown(svg, unsafe_allow_html=True)


def _row_kv(label: str, count: int, total: int) -> str:
    pct_str = f"{count / total:.0%}" if total else "—"
    return (
        "<div style='display:flex;justify-content:space-between;"
        "padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>"
        f"<span style='font-size:0.78rem;letter-spacing:0.05em;"
        f"text-transform:uppercase;opacity:0.75;'>{label}</span>"
        f"<span style='font-weight:700;'>{count} "
        f"<span style='opacity:0.55;font-weight:500;margin-left:6px;'>"
        f"({pct_str})</span></span></div>"
    )


def render_stats_records(name: str, stats: dict, *, color: str = "#E03131") -> None:
    """UFC.com-style Stats & Records dashboard. Lives below the Tale of the Tape.

    `color` defaults to UFC red but is typically passed the fighter's
    deterministic palette colour so the dashboard visually matches the
    Elo trajectory line on the same page.
    """
    st.markdown(
        "<h3 style='letter-spacing:0.08em;font-weight:800;"
        "margin-top:1.5rem;margin-bottom:0.6rem;'>STATS &amp; RECORDS</h3>",
        unsafe_allow_html=True,
    )

    # ---- Row 1: banner — wins by KO + first-round finishes -----------
    with st.container(border=True):
        b_left, b_right = st.columns(2)
        b_left.markdown(
            f"<div style='font-size:1.6rem;font-weight:700;'>"
            f"{stats['wins_ko']} <span style='font-size:0.85rem;font-weight:500;"
            f"letter-spacing:0.05em;opacity:0.7;text-transform:uppercase;"
            f"margin-left:6px;'>Wins by Knockout</span></div>",
            unsafe_allow_html=True,
        )
        b_right.markdown(
            f"<div style='font-size:1.6rem;font-weight:700;text-align:right;'>"
            f"{stats['first_round_finishes']} <span style='font-size:0.85rem;"
            f"font-weight:500;letter-spacing:0.05em;opacity:0.7;"
            f"text-transform:uppercase;margin-left:6px;'>First Round Finishes</span></div>",
            unsafe_allow_html=True,
        )

    # ---- Row 2: striking accuracy + takedown accuracy donuts ---------
    acc_left, acc_right = st.columns(2)

    def _accuracy_card(title: str, percent: float | None,
                       landed_label: str, landed_n: int,
                       attempted_label: str, attempted_n: int) -> None:
        with st.container(border=True):
            st.markdown(
                f"<div style='font-size:0.85rem;font-weight:700;"
                f"letter-spacing:0.06em;text-transform:uppercase;"
                f"margin-bottom:0.4rem;'>{title}</div>",
                unsafe_allow_html=True,
            )
            gauge_col, info_col = st.columns([1, 1])
            with gauge_col:
                render_donut_gauge(percent, color=color)
            with info_col:
                st.markdown(
                    f"<div style='padding-top:1.2rem;'>"
                    f"<div style='font-size:0.7rem;opacity:0.6;"
                    f"text-transform:uppercase;letter-spacing:0.06em;'>{landed_label}</div>"
                    f"<div style='font-size:1.1rem;font-weight:700;margin-bottom:0.6rem;'>"
                    f"{landed_n:,}</div>"
                    f"<div style='font-size:0.7rem;opacity:0.6;"
                    f"text-transform:uppercase;letter-spacing:0.06em;'>{attempted_label}</div>"
                    f"<div style='font-size:1.1rem;font-weight:700;'>{attempted_n:,}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    with acc_left:
        _accuracy_card(
            "Striking Accuracy", stats["sig_acc"],
            "Sig. Strikes Landed", stats["sig_landed"],
            "Sig. Strikes Attempted", stats["sig_attempted"],
        )
    with acc_right:
        _accuracy_card(
            "Takedown Accuracy", stats["td_acc"],
            "Takedowns Landed", stats["td_landed"],
            "Takedowns Attempted", stats["td_attempted"],
        )

    # ---- Row 3: per-min / per-15-min rates + defense ----------------
    rate_left, rate_right = st.columns(2)
    with rate_left:
        with st.container(border=True):
            r1, r2 = st.columns(2)
            r1.markdown(_stat_tile_html("Sig. Str. Landed", _fmt_rate(stats["slpm"]), "Per Min"),
                        unsafe_allow_html=True)
            r2.markdown(_stat_tile_html("Sig. Str. Absorbed", _fmt_rate(stats["sapm"]), "Per Min"),
                        unsafe_allow_html=True)
            r3, r4 = st.columns(2)
            r3.markdown(_stat_tile_html("Takedown Avg", _fmt_rate(stats["td_per15"]), "Per 15 Min"),
                        unsafe_allow_html=True)
            r4.markdown(_stat_tile_html("Submission Avg", _fmt_rate(stats["sub_per15"]), "Per 15 Min"),
                        unsafe_allow_html=True)
    with rate_right:
        with st.container(border=True):
            r1, r2 = st.columns(2)
            r1.markdown(_stat_tile_html("Sig. Str. Defense", _fmt_pct(stats["sig_def"])),
                        unsafe_allow_html=True)
            r2.markdown(_stat_tile_html("Takedown Defense", _fmt_pct(stats["td_def"])),
                        unsafe_allow_html=True)
            r3, r4 = st.columns(2)
            r3.markdown(_stat_tile_html("Knockdown Avg", _fmt_rate(stats["kd_per15"]), "Per 15 Min"),
                        unsafe_allow_html=True)
            r4.markdown(_stat_tile_html("Average Fight Time",
                                        _format_minutes_as_mmss(stats["avg_fight_time"])),
                        unsafe_allow_html=True)

    # ---- Row 4: position / target silhouette / win-by-method --------
    bot_a, bot_b, bot_c = st.columns(3)
    pos_total = (
        stats["distance_landed"] + stats["clinch_landed"] + stats["ground_landed"]
    )
    target_total = stats["head_landed"] + stats["body_landed"] + stats["leg_landed"]

    with bot_a:
        with st.container(border=True):
            st.markdown(
                "<div style='font-size:0.85rem;font-weight:700;"
                "letter-spacing:0.06em;text-transform:uppercase;"
                "text-align:center;margin-bottom:0.6rem;'>Sig. Str. by Position</div>",
                unsafe_allow_html=True,
            )
            for label, count in (
                ("Standing", stats["distance_landed"]),
                ("Clinch", stats["clinch_landed"]),
                ("Ground", stats["ground_landed"]),
            ):
                st.markdown(_row_kv(label, count, pos_total), unsafe_allow_html=True)

    with bot_b:
        with st.container(border=True):
            st.markdown(
                "<div style='font-size:0.85rem;font-weight:700;"
                "letter-spacing:0.06em;text-transform:uppercase;"
                "text-align:center;margin-bottom:0.4rem;'>Sig. Str. by Target</div>",
                unsafe_allow_html=True,
            )
            head_p = stats["head_landed"] / target_total if target_total else 0
            body_p = stats["body_landed"] / target_total if target_total else 0
            leg_p = stats["leg_landed"] / target_total if target_total else 0
            render_body_silhouette(
                (stats["head_landed"], head_p),
                (stats["body_landed"], body_p),
                (stats["leg_landed"], leg_p),
                color=color,
            )

    with bot_c:
        with st.container(border=True):
            st.markdown(
                "<div style='font-size:0.85rem;font-weight:700;"
                "letter-spacing:0.06em;text-transform:uppercase;"
                "text-align:center;margin-bottom:0.6rem;'>Win by Method</div>",
                unsafe_allow_html=True,
            )
            wins = stats["wins_by_method"]
            total_wins = stats["total_wins"]
            for label in ("KO/TKO", "DEC", "SUB"):
                st.markdown(
                    _row_kv(label, wins.get(label, 0), total_wins),
                    unsafe_allow_html=True,
                )


def render_tale_of_the_tape(
    name: str,
    bio: dict | None,
    peak_elo: float,
    peak_date: pd.Timestamp,
    peak_weight_class: str,
    fights_count: int,
    *,
    key_prefix: str = "",
    cardio_info=None,
) -> None:
    """Tale of the Tape using streamlit-shadcn-ui.

    Bordered card with:
      - title row
      - shadcn badges (weight class + stance / age / height / weight / reach)
      - three shadcn metric_cards (Peak Elo, Peak Date, UFC Fights)

    `key_prefix` lets us render two TotT cards side-by-side in Fight Finder
    without colliding on widget keys.
    """
    safe_key = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_") or "fighter"

    with st.container(border=True):
        st.markdown(
            f"<div class='tot-title'>🥊 TALE OF THE TAPE — "
            f"<strong>{name}</strong></div>",
            unsafe_allow_html=True,
        )

        # Bio attributes as shadcn badges. Weight class is the headline
        # (default variant); the rest are outline style for a calmer look.
        badge_list: list = [(str(peak_weight_class), "default")]
        if bio:
            stance_val = bio["stance"] or "—"
            badge_list.extend([
                (f"Stance: {stance_val}", "default"),
                (f"Age {_format_age_today(bio['dob'])}", "default"),
                (_format_height(bio["height_in"]), "default"),
                (_format_weight(bio["weight_lbs"]), "default"),
                (f"Reach {_format_reach(bio['reach_in'])}", "default"),
            ])
        ui.badges(
            badge_list=badge_list,
            class_name="tot-badges",
            key=f"{key_prefix}badges_{safe_key}",
        )

        if not bio:
            st.caption(
                "No bio data found for this fighter in "
                "`ufc_fighter_tott.csv`."
            )

        # Four shadcn metric_cards: Peak Elo, Peak Date, UFC Fights, Cardio Score.
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            ui.metric_card(
                title="Peak Elo",
                content=f"{peak_elo:.1f}",
                description="Career-best post-fight rating",
                key=f"{key_prefix}m_peak_elo_{safe_key}",
            )
        with m2:
            ui.metric_card(
                title="Peak Date",
                content=(
                    peak_date.strftime("%Y-%m-%d")
                    if pd.notna(peak_date) else "—"
                ),
                description="When the peak was hit",
                key=f"{key_prefix}m_peak_date_{safe_key}",
            )
        with m3:
            ui.metric_card(
                title="UFC Fights",
                content=str(int(fights_count)),
                description="Total processed bouts",
                key=f"{key_prefix}m_fights_{safe_key}",
            )
        with m4:
            # Cardio Score — 0–100 index from cardio_score.score_fighter().
            # Falls back to an 'Insufficient sample' card when the fighter
            # has fewer than 3 qualifying fights (or no per-round stats).
            if cardio_info and cardio_info.qualified:
                ui.metric_card(
                    title="Cardio Score",
                    content=f"{cardio_info.score:.0f}",
                    description=f"0–100 (n={cardio_info.n_fights} fights)",
                    key=f"{key_prefix}m_cardio_{safe_key}",
                )
            else:
                n = cardio_info.n_fights if cardio_info else 0
                ui.metric_card(
                    title="Cardio Score",
                    content="—",
                    description=f"Insufficient sample (n={n})",
                    key=f"{key_prefix}m_cardio_{safe_key}",
                )


def page_overall(
    peaks: pd.DataFrame,
    engine: UFCEloEngine,
    *,
    show_header: bool = True,
) -> None:
    # show_header is False when this page renders inside a tab on the
    # Top 10 Rankings page — the tab label already names the view.
    if show_header:
        st.title("Top 10 Fighters — Peak Elo (All-Time)")
    st.caption("K = 32, base = 1500. Finishes (KO/TKO/SUB) amplify K by ×1.20.")

    top10_table = render_top_n(peaks)
    top10_names = top10_table["Fighter"].tolist()
    history = engine.history_dataframe(top10_names)

    # Side-by-side: table on the left, chart on the right.
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.dataframe(top10_table, hide_index=True, use_container_width=True)

    with right:
        st.subheader("Elo trajectory by fight number")
        st.caption("Each line tracks one fighter's post-fight Elo across their UFC career.")
        if history.empty:
            st.info("No Elo history available to plot.")
        else:
            # Pin the y-axis floor to 1000 so the trajectories aren't squashed.
            y_max = float(history["Elo"].max())
            chart = (
                alt.Chart(history)
                .mark_line()
                .encode(
                    x=alt.X("Fight #:Q", title="Fight #"),
                    y=alt.Y(
                        "Elo:Q",
                        title="Elo rating",
                        scale=alt.Scale(domain=[1000, y_max + 20]),
                    ),
                    color=alt.Color("Fighter:N", sort=top10_names, title="Fighter"),
                )
                .properties(height=420)
            )
            st.altair_chart(chart, use_container_width=True)

    # Side-by-side delta tables: peak deltas (per fighter) and fight deltas
    # (per bout). Both computed across the entire dataset.
    st.divider()
    delta_left, delta_right = st.columns(2, gap="large")

    with delta_left:
        st.subheader("Top 10 biggest peak Elo deltas")
        st.caption(
            f"Peak Delta = Peak Elo − base rating ({engine.base:.0f}). "
            "Computed for every fighter; the 10 largest are shown."
        )
        deltas_all = peaks.copy()
        deltas_all["Peak Delta"] = (deltas_all["Peak Elo"] - engine.base).round(1)
        top10_peak_deltas = (
            deltas_all.sort_values("Peak Delta", ascending=False)
            .head(10)
            .copy()
        )
        top10_peak_deltas.insert(0, "Rank", range(1, len(top10_peak_deltas) + 1))
        top10_peak_deltas["Peak Elo"] = top10_peak_deltas["Peak Elo"].round(1)
        st.dataframe(
            top10_peak_deltas[["Rank", "Fighter", "Peak Elo", "Peak Delta"]],
            hide_index=True,
            use_container_width=True,
        )

    with delta_right:
        st.subheader("Top 10 biggest single-fight Elo deltas")
        st.caption(
            "The 10 individual bouts that produced the biggest rating swing. "
            "Delta shows winner / loser change."
        )
        top_fights = engine.top_fight_deltas(10)
        if top_fights.empty:
            st.info("No fights available.")
        else:
            top_fights.insert(0, "Rank", range(1, len(top_fights) + 1))
            st.dataframe(
                top_fights[["Rank", "Date", "Matchup", "Elo delta"]],
                hide_index=True,
                use_container_width=True,
            )


def page_by_weight_class(peaks: pd.DataFrame, *, show_header: bool = True) -> None:
    # See note on page_overall: title is suppressed when tabbed.
    if show_header:
        st.title("Top 10 per Weight Class — Peak Elo (All-Time)")
    st.caption("Weight class is the class of the bout in which a fighter hit their peak rating.")

    # Build an explicit canonical display order: men's lightest → heaviest,
    # then women's lightest → heaviest. Anything not in this list (e.g. raw
    # weight-class strings that didn't match a canonical division) is dropped
    # from the per-weight-class view.
    canonical_order: list[str] = (
        list(CANONICAL_WEIGHT_CLASSES)
        + [f"Womens {c}" for c in CANONICAL_WEIGHT_CLASSES]
    )
    classes_present = set(peaks["Weight Class"].dropna().astype(str))
    classes = [c for c in canonical_order if c in classes_present]

    if not classes:
        st.info("No weight classes found in the data.")
        return

    # Render in a 2-column grid (so 8 weight classes → 2 columns × 4 rows).
    for i in range(0, len(classes), 2):
        row = classes[i : i + 2]
        cols = st.columns(2)
        for col, wc in zip(cols, row):
            group = peaks[peaks["Weight Class"] == wc].sort_values("Peak Elo", ascending=False)
            with col:
                st.subheader(wc)
                if group.empty:
                    st.info("No fighters in this class.")
                else:
                    st.dataframe(
                        render_top_n(group),
                        hide_index=True,
                        use_container_width=True,
                    )


def render_closest_matchups(
    selected: str,
    peaks: pd.DataFrame,
    *,
    n: int = 10,
    key_prefix: str = "fs_",
) -> None:
    """Render a 'Closest matchups in <weight class>' section.

    Finds the n fighters in the same peak weight class closest in peak Elo
    to `selected`, displays them as a ranked table, and offers a dropdown +
    button to open any of them in Fighter Search. No-op if the fighter isn't
    in `peaks` or has no division peers.
    """
    selected_row_df = peaks[peaks["Fighter"] == selected]
    if selected_row_df.empty:
        return
    selected_row = selected_row_df.iloc[0]
    weight_class = str(selected_row["Weight Class"])
    selected_elo = float(selected_row["Peak Elo"])

    # Same peak weight class, excluding the selected fighter themselves.
    division = peaks[
        (peaks["Weight Class"] == weight_class) & (peaks["Fighter"] != selected)
    ].copy()
    if division.empty:
        st.info(f"No other fighters in {weight_class} to compare against.")
        return

    division["Peak Elo"] = division["Peak Elo"].astype(float)
    division["ΔElo"] = (division["Peak Elo"] - selected_elo).round(1)
    division["abs_delta"] = division["ΔElo"].abs()
    # Pick the n with the smallest absolute Elo gap, then re-sort the
    # surviving rows by peak Elo descending so the table reads top-to-bottom
    # like a leaderboard.
    closest = (
        division.sort_values("abs_delta")
        .head(n)
        .copy()
        .sort_values("Peak Elo", ascending=False)
        .reset_index(drop=True)
    )

    st.markdown(
        f"<h3 style='letter-spacing:0.08em;font-weight:800;"
        f"margin-top:1.5rem;margin-bottom:0.3rem;'>CLOSEST MATCHUPS IN "
        f"{weight_class.upper()}</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"The {len(closest)} fighter"
        f"{'s' if len(closest) != 1 else ''} closest in peak Elo to "
        f"**{selected}** ({selected_elo:.1f}) within {weight_class}, "
        "ordered by peak Elo (highest first)."
    )

    display = closest[["Fighter", "Peak Elo", "ΔElo"]].copy()
    display["Peak Elo"] = display["Peak Elo"].round(1)
    display["ΔElo"] = display["ΔElo"].apply(lambda d: f"{d:+.1f}")
    display.insert(0, "Rank", range(1, len(display) + 1))
    st.dataframe(display, hide_index=True, use_container_width=True)

    # Quick jump dropdown — pick a peer and open them in Fighter Search.
    jump_col, btn_col = st.columns([4, 1], gap="medium")
    with jump_col:
        jump_choice = st.selectbox(
            "Open another fighter",
            options=closest["Fighter"].tolist(),
            index=None,
            placeholder="Pick a peer to open in Fighter Search…",
            label_visibility="collapsed",
            key=f"{key_prefix}closest_jump_{selected}",
        )
    with btn_col:
        if st.button(
            "Open →",
            disabled=jump_choice is None,
            type="primary",
            use_container_width=True,
            key=f"{key_prefix}closest_open_{selected}",
        ):
            st.session_state["fs_selected_fighter"] = jump_choice
            st.rerun()


def page_fighter_search(peaks: pd.DataFrame, engine: UFCEloEngine) -> None:
    record_nav_step("Fighter Search")
    st.title("Fighter Search")
    st.caption(
        "Search any fighter in the dataset and inspect their full UFC fight "
        "history."
    )

    fight_count_map = engine.fight_counts()

    # Search field on the left, filters tucked into a popover on the right.
    # Rendering the popover first ensures min_fights / sort_mode are available
    # before the eligible list is built.
    search_col, filter_col = st.columns([4, 1], gap="medium")

    with filter_col:
        # Invisible spacer that matches the height of a selectbox label so
        # the Filters button lines up with the search input below.
        st.markdown(
            "<div style='height:1.75rem'></div>",
            unsafe_allow_html=True,
        )
        with st.popover("Filters", use_container_width=True):
            min_fights = st.slider(
                "Minimum UFC fights",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
                help=(
                    "Hide fighters with fewer than this many UFC fights. "
                    "Set to 0 to include everyone."
                ),
            )
            sort_mode = st.selectbox(
                "Sort by",
                options=[
                    "Alphabetical",
                    "Peak Elo (high → low)",
                    "Peak Elo (low → high)",
                    "Cardio Score (high → low)",
                    "Cardio Score (low → high)",
                ],
                index=0,
            )

    # Build the eligible pool: full peaks table, with a Fights column attached.
    eligible = peaks.copy()
    eligible["Fights"] = (
        eligible["Fighter"].map(fight_count_map).fillna(0).astype(int)
    )
    eligible = eligible[eligible["Fights"] >= min_fights]

    # Apply the chosen sort order to the dropdown list.
    if sort_mode == "Peak Elo (high → low)":
        eligible = eligible.sort_values("Peak Elo", ascending=False)
    elif sort_mode == "Peak Elo (low → high)":
        eligible = eligible.sort_values("Peak Elo", ascending=True)
    elif sort_mode in ("Cardio Score (high → low)", "Cardio Score (low → high)"):
        # Score every eligible fighter once. Unqualified fighters get NaN so
        # they sink to the bottom regardless of sort direction (na_position).
        cardio_baselines = build_cardio_baselines()
        eligible["Cardio Score"] = [
            score_fighter(f, cardio_baselines).score for f in eligible["Fighter"]
        ]
        ascending = sort_mode == "Cardio Score (low → high)"
        eligible = eligible.sort_values(
            "Cardio Score", ascending=ascending, na_position="last"
        )
    else:  # Alphabetical (case-insensitive)
        eligible = eligible.sort_values(
            "Fighter", key=lambda s: s.str.lower()
        )
    eligible = eligible.reset_index(drop=True)
    names = eligible["Fighter"].tolist()

    with search_col:
        if not names:
            st.info(f"No fighters meet the minimum of {min_fights} fight(s).")
            return
        selected = st.selectbox(
            f"Search for a fighter ({len(names):,} eligible)",
            options=names,
            index=None,
            placeholder="Start typing a name…",
            key="fs_selected_fighter",
        )

    if not selected:
        st.info("Pick a fighter from the dropdown above to view their fight history.")
        return

    # Per-fighter accent banner across the top of this section.
    render_fighter_banner(fighter_color(selected))

    # Headline metrics + bio combined into the Tale of the Tape card.
    peak_row = eligible[eligible["Fighter"] == selected].iloc[0]
    peak_date = pd.to_datetime(peak_row["Peak Date"])
    bios = load_bios()
    bio = bios.get(_normalise_name(selected))
    cardio_baselines = build_cardio_baselines()
    cardio = score_fighter(selected, cardio_baselines)
    render_tale_of_the_tape(
        name=selected,
        bio=bio,
        peak_elo=float(peak_row["Peak Elo"]),
        peak_date=peak_date,
        peak_weight_class=str(peak_row["Weight Class"]),
        fights_count=int(peak_row["Fights"]),
        key_prefix="fs_",
        cardio_info=cardio,
    )

    # Wikipedia bio fields scraped by build_fighter_profiles.py.
    wiki_profiles = load_wiki_profiles()
    render_wiki_profile_card(wiki_profiles.get(_normalise_name(selected)))

    # UFC.com-style Stats & Records dashboard, computed from career totals
    # in ufc_fight_stats.csv + fight durations parsed from ufc_fight_results.csv.
    stats_df = load_stats()
    durations = load_fight_durations()
    career = compute_career_stats(selected, engine, stats_df, durations)
    render_stats_records(selected, career, color=fighter_color(selected))

    # Peer context — closest fighters in peak Elo within the same weight class.
    render_closest_matchups(selected, peaks)

    fights = engine.fighter_fights(selected)
    if fights.empty:
        st.info("No fights found for this fighter.")
        return

    fights = fights.copy()
    fights["Date"] = pd.to_datetime(fights["Date"]).dt.strftime("%Y-%m-%d")

    # Reach differential vs each opponent (selected fighter's reach minus
    # opponent's reach, in inches). Renders as "—" if either reach is missing.
    fighter_reach = bio.get("reach_in") if bio else None
    reach_diffs = []
    for opp in fights["Opponent"]:
        opp_bio = bios.get(_normalise_name(opp), {})
        opp_reach = opp_bio.get("reach_in")
        if fighter_reach is None or opp_reach is None:
            reach_diffs.append("—")
        else:
            reach_diffs.append(f"{round(fighter_reach - opp_reach):+d}\"")
    fights["Reach Δ"] = reach_diffs

    st.subheader(
        f"{selected} — {len(fights)} UFC fight"
        f"{'s' if len(fights) != 1 else ''} (most recent first)"
    )
    st.dataframe(
        fights[["Date", "Opponent", "Result", "Weight Class", "Reach Δ", "ΔElo"]],
        hide_index=True,
        use_container_width=True,
    )

    # Elo trajectory across the selected fighter's UFC career.
    # Built as a Plotly line chart wrapped in streamlit-plotly-events so
    # clicking any point reveals that bout inline + an "Open in Fight
    # Finder" button.
    elo_series = engine.history.get(selected, [])
    if elo_series:
        # Walk the fight history in chronological order and pair each fight
        # with its post-fight Elo (engine.history is appended in process order,
        # which matches chronological order).
        trajectory = (
            fights.sort_values("Date", ascending=True)
            .reset_index(drop=True)
            .copy()
        )
        n = min(len(trajectory), len(elo_series))
        trajectory = trajectory.iloc[:n].copy()
        trajectory.insert(0, "Fight #", range(1, n + 1))
        trajectory["Elo"] = [round(e, 1) for e in elo_series[:n]]
        trajectory = trajectory.rename(columns={"ΔElo": "Elo delta"})

        st.subheader("Elo trajectory by fight number")
        st.caption(
            "Post-fight Elo after each UFC bout, in chronological order. "
            "**Click any point** to see bout details and jump to Fight Finder."
        )

        traj_color = fighter_color(selected)
        y_min = 1400
        y_max_data = float(trajectory["Elo"].max())

        traj_fig = go.Figure()
        traj_fig.add_trace(
            go.Scatter(
                x=trajectory["Fight #"],
                y=trajectory["Elo"],
                mode="lines+markers",
                line=dict(color=traj_color, width=2.5),
                marker=dict(
                    size=12,
                    color=traj_color,
                    line=dict(color="white", width=1),
                ),
                customdata=list(zip(
                    trajectory["Date"].astype(str),
                    trajectory["Opponent"].astype(str),
                    trajectory["Result"].astype(str),
                    trajectory["Elo delta"],
                )),
                hovertemplate=(
                    "<b>Fight #%{x}</b><br>"
                    "Date: %{customdata[0]}<br>"
                    "Opponent: %{customdata[1]}<br>"
                    "Result: %{customdata[2]}<br>"
                    "Elo delta: %{customdata[3]:+.1f}<br>"
                    "Post-fight Elo: %{y:.1f}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        traj_fig.update_layout(
            height=360,
            xaxis_title="Fight #",
            yaxis_title="Elo rating",
            yaxis=dict(range=[y_min, y_max_data + 20]),
            margin=dict(t=30, b=40, l=50, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAFAFA"),
            hovermode="closest",
        )
        traj_fig.update_xaxes(gridcolor="#1A1D24", zerolinecolor="#1A1D24")
        traj_fig.update_yaxes(gridcolor="#1A1D24", zerolinecolor="#1A1D24")

        # Native Streamlit plotly click events (Streamlit >= 1.39).
        traj_event = st.plotly_chart(
            traj_fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key=f"trajectory_chart_{selected}",
        )

        clicked_points = []
        if traj_event and getattr(traj_event, "selection", None):
            clicked_points = traj_event.selection.get("points", []) or []

        if clicked_points:
            point = clicked_points[0]
            idx = point.get("point_index")
            if idx is None:
                idx = point.get("pointIndex")
            if idx is not None and 0 <= int(idx) < len(trajectory):
                bout_row = trajectory.iloc[int(idx)]
                with st.container(border=True):
                    st.markdown(
                        f"#### Bout #{int(bout_row['Fight #'])} — "
                        f"vs **{bout_row['Opponent']}**"
                    )
                    bcols = st.columns(4)
                    bcols[0].metric("Date", str(bout_row["Date"]))
                    bcols[1].metric("Result", str(bout_row["Result"]))
                    bcols[2].metric(
                        "Elo Δ", f"{float(bout_row['Elo delta']):+.1f}"
                    )
                    bcols[3].metric(
                        "Post-fight Elo", f"{float(bout_row['Elo']):.1f}"
                    )
                    if st.button(
                        "Open in Fight Finder →",
                        key=f"open_finder_{selected}_{int(idx)}",
                        type="primary",
                    ):
                        st.session_state["ff_fighter_a"] = selected
                        st.session_state["ff_fighter_b"] = bout_row["Opponent"]
                        st.session_state["nav_target"] = "Fight Finder"
                        st.rerun()


def _format_acc(landed: int, attempted: int) -> str:
    if not attempted:
        return "—"
    return f"{landed / attempted:.0%}"


def render_h2h_stat_comparison(
    fighter_a: str,
    stats_a: dict | None,
    fighter_b: str,
    stats_b: dict | None,
) -> None:
    """Render a per-fight stat comparison table for one bout."""
    if stats_a is None and stats_b is None:
        st.info(
            "No per-fight stat data available for this bout "
            "(common for very early UFC fights)."
        )
        return

    def g(stats: dict | None, key: str, default: int = 0):
        if stats is None:
            return default
        return stats.get(key, default)

    rows = [
        ("Sig. strikes", lambda s: f"{g(s, 'sig_landed')} / {g(s, 'sig_attempted')}"),
        ("Sig. strike acc.", lambda s: _format_acc(g(s, 'sig_landed'), g(s, 'sig_attempted'))),
        ("Total strikes", lambda s: f"{g(s, 'total_landed')} / {g(s, 'total_attempted')}"),
        ("Knockdowns", lambda s: str(g(s, 'kd'))),
        ("Takedowns", lambda s: f"{g(s, 'td_landed')} / {g(s, 'td_attempted')}"),
        ("Takedown acc.", lambda s: _format_acc(g(s, 'td_landed'), g(s, 'td_attempted'))),
        ("Sub. attempts", lambda s: str(g(s, 'sub_att'))),
        ("Control time", lambda s: _format_ctrl(g(s, 'ctrl_sec'))),
        ("Head strikes", lambda s: f"{g(s, 'head_landed')} / {g(s, 'head_attempted')}"),
        ("Body strikes", lambda s: f"{g(s, 'body_landed')} / {g(s, 'body_attempted')}"),
        ("Leg strikes", lambda s: f"{g(s, 'leg_landed')} / {g(s, 'leg_attempted')}"),
    ]

    table_rows = [
        {"Stat": label, fighter_a: fmt(stats_a), fighter_b: fmt(stats_b)}
        for label, fmt in rows
    ]

    # Significant strike differential: how many more sig strikes the fighter
    # landed than they absorbed in this bout. Values mirror across the two
    # fighters (e.g. +12 / -12). Falls back to "—" if either side is missing
    # per-fight stat data.
    if stats_a and stats_b:
        diff = int(stats_a.get("sig_landed", 0)) - int(stats_b.get("sig_landed", 0))
        diff_a_str = f"{diff:+d}"
        diff_b_str = f"{-diff:+d}"
    else:
        diff_a_str = "—"
        diff_b_str = "—"
    # Place the differential row right after Sig. strikes / Sig. strike acc.
    table_rows.insert(
        2,
        {"Stat": "Sig. strike diff.", fighter_a: diff_a_str, fighter_b: diff_b_str},
    )

    table = pd.DataFrame(table_rows)
    st.dataframe(table, hide_index=True, use_container_width=True)


def render_round_by_round_stats(
    fighter_a: str,
    rounds_a: pd.DataFrame,
    fighter_b: str,
    rounds_b: pd.DataFrame,
) -> None:
    """Render a tabbed round-by-round stat comparison for one bout.

    Each round becomes its own tab, mirroring the per-fight comparison
    table but scoped to that round. Falls back to an info banner if the
    bout has no per-round rows in ufc_fight_stats.csv (common for very
    early UFC fights).
    """
    if rounds_a.empty and rounds_b.empty:
        st.info(
            "No per-round stat data available for this bout "
            "(common for very early UFC fights)."
        )
        return

    rounds_set = sorted(set(
        list(rounds_a["round_num"].dropna().astype(int).tolist())
        + list(rounds_b["round_num"].dropna().astype(int).tolist())
    ))
    if not rounds_set:
        st.info("No per-round stat data available for this bout.")
        return

    def g(row: dict | None, key: str, default: int = 0) -> int:
        if row is None:
            return default
        v = row.get(key, default)
        try:
            return int(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    rows = [
        ("Sig. strikes", lambda s: f"{g(s, 'sig_landed')} / {g(s, 'sig_attempted')}"),
        ("Sig. strike acc.", lambda s: _format_acc(g(s, 'sig_landed'), g(s, 'sig_attempted'))),
        ("Total strikes", lambda s: f"{g(s, 'total_landed')} / {g(s, 'total_attempted')}"),
        ("Knockdowns", lambda s: str(g(s, 'kd'))),
        ("Takedowns", lambda s: f"{g(s, 'td_landed')} / {g(s, 'td_attempted')}"),
        ("Takedown acc.", lambda s: _format_acc(g(s, 'td_landed'), g(s, 'td_attempted'))),
        ("Sub. attempts", lambda s: str(g(s, 'sub_att'))),
        ("Control time", lambda s: _format_ctrl(g(s, 'ctrl_sec'))),
        ("Head strikes", lambda s: f"{g(s, 'head_landed')} / {g(s, 'head_attempted')}"),
        ("Body strikes", lambda s: f"{g(s, 'body_landed')} / {g(s, 'body_attempted')}"),
        ("Leg strikes", lambda s: f"{g(s, 'leg_landed')} / {g(s, 'leg_attempted')}"),
        ("Distance strikes", lambda s: f"{g(s, 'distance_landed')} / {g(s, 'distance_attempted')}"),
        ("Clinch strikes", lambda s: f"{g(s, 'clinch_landed')} / {g(s, 'clinch_attempted')}"),
        ("Ground strikes", lambda s: f"{g(s, 'ground_landed')} / {g(s, 'ground_attempted')}"),
    ]

    tabs = st.tabs([f"Round {r}" for r in rounds_set])
    for tab, r in zip(tabs, rounds_set):
        with tab:
            ra = rounds_a[rounds_a["round_num"] == r]
            rb = rounds_b[rounds_b["round_num"] == r]
            row_a = ra.iloc[0].to_dict() if not ra.empty else None
            row_b = rb.iloc[0].to_dict() if not rb.empty else None

            if row_a is None and row_b is None:
                st.caption(f"No data for round {r}.")
                continue

            table_rows = [
                {"Stat": label, fighter_a: fmt(row_a), fighter_b: fmt(row_b)}
                for label, fmt in rows
            ]

            # Sig. strike differential for the round (mirrors across fighters).
            if row_a is not None and row_b is not None:
                diff = g(row_a, "sig_landed") - g(row_b, "sig_landed")
                diff_a_str = f"{diff:+d}"
                diff_b_str = f"{-diff:+d}"
            else:
                diff_a_str = "—"
                diff_b_str = "—"
            table_rows.insert(
                2,
                {"Stat": "Sig. strike diff.", fighter_a: diff_a_str, fighter_b: diff_b_str},
            )

            st.dataframe(
                pd.DataFrame(table_rows),
                hide_index=True,
                use_container_width=True,
            )


def page_fight_finder(peaks: pd.DataFrame, engine: UFCEloEngine) -> None:
    record_nav_step("Fight Finder")
    st.title("Fight Finder")
    st.caption(
        "Pick any two fighters who actually fought each other and see "
        "their head-to-head matchup."
    )

    eligible = peaks.copy().sort_values(
        "Fighter", key=lambda s: s.str.lower()
    ).reset_index(drop=True)
    names = eligible["Fighter"].tolist()
    if not names:
        st.info("No fighters available.")
        return

    # Two text-search fields side by side.
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        fighter_a = st.selectbox(
            f"Fighter 1 ({len(names):,} eligible)",
            options=names,
            index=None,
            placeholder="Start typing a name…",
            key="ff_fighter_a",
        )

    # Field 2 is dynamically filtered to fighter 1's actual opponents.
    opponents = engine.opponents_of(fighter_a) if fighter_a else []
    with col_b:
        if not fighter_a:
            st.selectbox(
                "Fighter 2",
                options=[],
                index=None,
                placeholder="Pick fighter 1 first…",
                disabled=True,
                key="ff_fighter_b_disabled",
            )
            fighter_b = None
        elif not opponents:
            st.selectbox(
                "Fighter 2",
                options=[],
                index=None,
                placeholder="No recorded opponents",
                disabled=True,
                key="ff_fighter_b_empty",
            )
            fighter_b = None
        else:
            fighter_b = st.selectbox(
                f"Fighter 2 ({len(opponents)} opponent"
                f"{'s' if len(opponents) != 1 else ''} of {fighter_a})",
                options=opponents,
                index=None,
                placeholder="Start typing a name…",
                key="ff_fighter_b",
            )

    if not fighter_a or not fighter_b:
        st.info("Select both fighters to see their matchup.")
        return

    # Two-fighter accent banner blending each fighter's palette colour.
    render_two_fighter_banner(
        fighter_color(fighter_a),
        fighter_color(fighter_b),
    )

    # Side-by-side Tale of the Tape.
    bios = load_bios()
    fight_count_map = engine.fight_counts()
    cardio_baselines = build_cardio_baselines()
    tot_left, tot_right = st.columns(2, gap="large")
    for col, name, key_prefix in (
        (tot_left, fighter_a, "ff_a_"),
        (tot_right, fighter_b, "ff_b_"),
    ):
        peak_row = eligible[eligible["Fighter"] == name].iloc[0]
        with col:
            render_tale_of_the_tape(
                name=name,
                bio=bios.get(_normalise_name(name)),
                peak_elo=float(peak_row["Peak Elo"]),
                peak_date=pd.to_datetime(peak_row["Peak Date"]),
                peak_weight_class=str(peak_row["Weight Class"]),
                fights_count=int(fight_count_map.get(name, 0)),
                key_prefix=key_prefix,
                cardio_info=score_fighter(name, cardio_baselines),
            )

    bouts = engine.head_to_head(fighter_a, fighter_b)
    if not bouts:
        st.info(f"No recorded bouts between {fighter_a} and {fighter_b}.")
        return

    # Default to the most recent bout. Show a dropdown only if they fought
    # more than once.
    n = len(bouts)
    if n > 1:
        labels = [
            f"Fight {i + 1} — {pd.to_datetime(b['date']).strftime('%Y-%m-%d')}"
            for i, b in enumerate(bouts)
        ]
        choice = st.selectbox(
            f"Bout ({n} fights between these two)",
            options=list(range(n)),
            format_func=lambda i: labels[i],
            index=n - 1,
            key="ff_bout_choice",
        )
        bout = bouts[choice]
    else:
        bout = bouts[0]

    # Bout summary.
    bout_date = pd.to_datetime(bout["date"]).strftime("%Y-%m-%d")
    if bout["a_outcome"] == "W":
        winner, loser = fighter_a, fighter_b
    elif bout["a_outcome"] == "L":
        winner, loser = fighter_b, fighter_a
    else:
        winner, loser = None, None
    method_tag = UFCEloEngine._format_result(bout["a_outcome"], bout["method"])
    if winner:
        headline = f"**{winner}** def. **{loser}** — {method_tag}"
    else:
        headline = f"**{fighter_a}** vs **{fighter_b}** — {method_tag}"

    st.subheader("Bout summary")
    st.markdown(
        f"**Date:** {bout_date}  \n"
        f"**Event:** {bout.get('event') or '—'}  \n"
        f"**Weight class:** {bout.get('weight_class') or '—'}  \n"
        f"**Result:** {headline}  \n"
        f"**Elo Δ:** {bout['a_delta']:+.1f} ({fighter_a}) / "
        f"{bout['b_delta']:+.1f} ({fighter_b})"
    )

    # Per-fight stat comparison.
    st.subheader("Per-fight stat comparison")
    stats = load_stats()
    stats_a = lookup_fight_stats(
        stats, bout.get("event", ""), bout.get("bout", ""), fighter_a
    )
    stats_b = lookup_fight_stats(
        stats, bout.get("event", ""), bout.get("bout", ""), fighter_b
    )
    render_h2h_stat_comparison(fighter_a, stats_a, fighter_b, stats_b)

    # Round-by-round stat comparison (one tab per round) sourced from the
    # un-aggregated rows of ufc_fight_stats.csv.
    st.subheader("Round-by-round stats")
    stats_by_round = load_stats_by_round()
    rounds_a = lookup_round_stats(
        stats_by_round, bout.get("event", ""), bout.get("bout", ""), fighter_a
    )
    rounds_b = lookup_round_stats(
        stats_by_round, bout.get("event", ""), bout.get("bout", ""), fighter_b
    )
    render_round_by_round_stats(fighter_a, rounds_a, fighter_b, rounds_b)


# ---------------------------------------------------------------------------
# Fight Network (Plotly + NetworkX) + Career Landscape (Plotly)
# ---------------------------------------------------------------------------

def _elo_gradient_color(elo: float, lo: float, hi: float) -> str:
    """Map an Elo value to a hex colour along a cool→hot gradient.

    Cool blue (#1f77b4) at the lowest Elo in the pool, hot red (#E03131)
    at the highest. Used to colour Fight Network nodes.
    """
    if hi <= lo:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (elo - lo) / (hi - lo)))
    r = int(0x1f + t * (0xE0 - 0x1f))
    g = int(0x77 + t * (0x31 - 0x77))
    b = int(0xb4 + t * (0x31 - 0xb4))
    return f"#{r:02x}{g:02x}{b:02x}"


def page_fight_network(
    peaks: pd.DataFrame,
    engine: UFCEloEngine,
    *,
    show_header: bool = True,
) -> None:
    """Fight Network — divisions as Poisson-distributed islands.

    Layout is FULLY pre-computed in Python (no live force simulation):
      1. Variable-radius Poisson-disk place anchors per active division.
         Halo radius scales with sqrt(fighter count); anchor exclusion
         radius allows mild overlap (matches the reference image).
      2. Single-division fighters: mini NetworkX spring_layout per
         division, scaled to ~0.65× the halo radius and translated to
         the anchor.
      3. Multi-division fighters: equal-weighted centroid of their
         division anchors + small deterministic jitter.
      4. Render with vanilla ECharts via st.components.v1.html so the
         iframe size is fully under our control (no streamlit-echarts
         iframe quirks).

    Click-to-inspect was dropped when we left streamlit-echarts (vanilla
    ECharts in an iframe can't push events back to Python); inspection
    now goes through a selectbox in the side panel.
    """
    if show_header:
        st.title("Fight Network")
    st.caption(
        "Each division is its own 'island' — fighters cluster tightly "
        "around their division's halo. Multi-division fighters wander "
        "between the divisions they fought in. Halo size scales with "
        "fighter count."
    )

    import json
    import math
    import random

    DIVISION_COLORS = {
        "Strawweight":       "#FF4D6D",
        "Flyweight":         "#FF9E00",
        "Bantamweight":      "#FFD166",
        "Featherweight":     "#A8E10C",
        "Lightweight":       "#06D6A0",
        "Welterweight":      "#118AB2",
        "Middleweight":      "#5A189A",
        "Light Heavyweight": "#9D4EDD",
        "Heavyweight":       "#E5383B",
        "Open Weight":       "#8D99AE",
        "Catch Weight":      "#6C757D",
    }
    DEFAULT_DIV_COLOR = "#888888"

    def _shared_div(wc):
        if isinstance(wc, str) and wc.startswith("Womens "):
            return wc[len("Womens "):]
        return wc if isinstance(wc, str) else ""

    def _hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    def _gather_div_counts(fighters_set):
        counts = {}
        for f in engine.fight_deltas:
            a, b = f["fighter_a"], f["fighter_b"]
            wc = _shared_div(f["weight_class"])
            if not wc:
                continue
            for who in (a, b):
                if who in fighters_set:
                    counts.setdefault(who, {})
                    counts[who][wc] = counts[who].get(wc, 0) + 1
        return counts

    # ---- Filter UI ----------------------------------------------------
    gender_mode = st.radio(
        "Gender",
        options=["All", "Men's", "Women's"],
        index=0,
        horizontal=True,
        key="fn_gender_mode",
    )

    canonical_shared = list(CANONICAL_WEIGHT_CLASSES)
    if gender_mode == "Men's":
        present_shared = {
            _shared_div(d) for d in peaks["Weight Class"]
            if isinstance(d, str) and not d.startswith("Womens ")
        }
    elif gender_mode == "Women's":
        present_shared = {
            _shared_div(d) for d in peaks["Weight Class"]
            if isinstance(d, str) and d.startswith("Womens ")
        }
    else:
        present_shared = {
            _shared_div(d) for d in peaks["Weight Class"]
            if isinstance(d, str)
        }
    eligible_divs = [d for d in canonical_shared if d in present_shared]

    selected_divs = st.multiselect(
        "Divisions",
        options=eligible_divs,
        default=eligible_divs,
        key=f"fn_div_select_{gender_mode}",
        help=(
            "Each division is shared by men's and women's fighters of "
            "the same weight. Divisions with no qualifying fighters are "
            "hidden from the layout entirely."
        ),
    )
    if not selected_divs:
        selected_divs = eligible_divs

    slider_col, label_col = st.columns([4, 1], gap="medium")
    with slider_col:
        n = st.slider(
            "Top N by peak Elo (within filtered pool)",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
        )
    with label_col:
        label_choice = st.selectbox(
            "Labels",
            options=["10", "15", "25", "All"],
            index=1,
        )

    # ---- Filter the pool ----------------------------------------------
    selected_set = set(selected_divs)
    is_womens = peaks["Weight Class"].astype(str).str.startswith("Womens ")
    if gender_mode == "Men's":
        gender_mask = ~is_womens
    elif gender_mode == "Women's":
        gender_mask = is_womens
    else:
        gender_mask = pd.Series(True, index=peaks.index)
    div_mask = (
        peaks["Weight Class"]
        .astype(str)
        .map(_shared_div)
        .isin(selected_set)
    )
    pool = (
        peaks[gender_mask & div_mask]
        .copy()
        .head(n)
        .reset_index(drop=True)
    )
    fighters_in_pool = set(pool["Fighter"])
    div_counts = _gather_div_counts(fighters_in_pool)
    fighter_divs = {
        f: set(div_counts.get(f, {}).keys()) & selected_set
        for f in fighters_in_pool
    }
    fighter_divs = {f: divs for f, divs in fighter_divs.items() if divs}
    fighters_in_pool = set(fighter_divs.keys())
    pool = pool[pool["Fighter"].isin(fighters_in_pool)].reset_index(drop=True)
    if pool.empty:
        st.info("No fighters match the current filters.")
        return

    name_to_elo = dict(zip(pool["Fighter"], pool["Peak Elo"]))
    name_to_div_raw = dict(zip(pool["Fighter"], pool["Weight Class"]))
    fight_count_map = engine.fight_counts()
    elo_lo = float(pool["Peak Elo"].min())
    elo_hi = float(pool["Peak Elo"].max())

    if label_choice == "All":
        labelled = set(pool["Fighter"])
    else:
        k = int(label_choice)
        labelled = set(
            pool.sort_values("Peak Elo", ascending=False)
            .head(k)["Fighter"]
            .tolist()
        )

    # ---- Active divisions: only those with at least one fighter -------
    active_divs_set = set()
    for divs in fighter_divs.values():
        active_divs_set.update(divs)
    active_divs = [d for d in canonical_shared if d in active_divs_set]
    if not active_divs:
        st.info("No divisions to render.")
        return

    # ---- Halo radius per division (sqrt scaled, clamped) --------------
    div_fighter_counts = {d: 0 for d in active_divs}
    for f, divs in fighter_divs.items():
        for d in divs:
            if d in div_fighter_counts:
                div_fighter_counts[d] += 1
    HALO_R_MIN = 110.0
    HALO_R_MAX = 280.0
    max_count = max(div_fighter_counts.values())
    halo_radius = {}
    for d, c in div_fighter_counts.items():
        if max_count <= 1:
            halo_radius[d] = HALO_R_MIN
        else:
            t = math.sqrt(c) / math.sqrt(max_count)
            halo_radius[d] = HALO_R_MIN + t * (HALO_R_MAX - HALO_R_MIN)

    # ---- Deterministic seed from filter state -------------------------
    seed_str = (
        f"{gender_mode}|"
        f"{','.join(sorted(active_divs))}|"
        f"{n}|{label_choice}"
    )
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:12], 16) & 0xFFFFFFFF

    # ---- Variable-radius Poisson-disk anchor placement ----------------
    # Largest halos placed first (hardest to fit). Min separation between
    # any two anchor centres = (r_i + r_j) * (1 - OVERLAP_FACTOR), so a
    # higher OVERLAP_FACTOR allows more halo bleed-through.
    CANVAS_W = 2400.0
    CANVAS_H = 1500.0
    OVERLAP_FACTOR = 0.18  # mild overlap (matches reference)

    rng = random.Random(seed)
    divs_by_size = sorted(active_divs, key=lambda d: -halo_radius[d])
    placed = []  # list of (div, x, y, r)
    for div in divs_by_size:
        r = halo_radius[div]
        positioned = False
        # Progressively relax overlap if we can't fit at the target.
        for relax in (0.0, 0.1, 0.2, 0.35, 0.5):
            for _ in range(400):
                margin = r * 0.4
                x = rng.uniform(-CANVAS_W / 2 + margin, CANVAS_W / 2 - margin)
                y = rng.uniform(-CANVAS_H / 2 + margin, CANVAS_H / 2 - margin)
                ok = True
                for _, px, py, pr in placed:
                    min_d = (r + pr) * (1 - OVERLAP_FACTOR - relax)
                    if math.hypot(x - px, y - py) < min_d:
                        ok = False
                        break
                if ok:
                    placed.append((div, x, y, r))
                    positioned = True
                    break
            if positioned:
                break
        if not positioned:
            placed.append((div, rng.uniform(-200, 200),
                           rng.uniform(-100, 100), r))

    anchor_pos = {div: (x, y) for div, x, y, _ in placed}
    anchor_radius = {div: r for div, _, _, r in placed}

    # ---- Mini force layout per division -------------------------------
    h2h_pairs = set()
    for f in engine.fight_deltas:
        a, b = f["fighter_a"], f["fighter_b"]
        if a in fighters_in_pool and b in fighters_in_pool:
            h2h_pairs.add(tuple(sorted([a, b])))

    h2h_graph = nx.Graph()
    h2h_graph.add_nodes_from(fighters_in_pool)
    h2h_graph.add_edges_from(h2h_pairs)

    fighter_pos = {}

    for div in active_divs:
        members = [
            f for f, divs in fighter_divs.items()
            if divs == {div}
        ]
        if not members:
            continue
        ax, ay = anchor_pos[div]
        ar = anchor_radius[div]
        if len(members) == 1:
            fighter_pos[members[0]] = (ax, ay)
            continue
        sub = h2h_graph.subgraph(members).copy()
        sub_pos = nx.spring_layout(
            sub,
            seed=(seed ^ (hash(div) & 0xFFFFFFFF)) & 0xFFFFFFFF,
            k=1.0 / math.sqrt(len(members)),
            iterations=80,
        )
        cx = sum(p[0] for p in sub_pos.values()) / len(sub_pos)
        cy = sum(p[1] for p in sub_pos.values()) / len(sub_pos)
        sub_pos = {f: (p[0] - cx, p[1] - cy) for f, p in sub_pos.items()}
        max_r = max(math.hypot(p[0], p[1]) for p in sub_pos.values()) or 1.0
        target_r = ar * 0.65
        scale = target_r / max_r
        for f, (xx, yy) in sub_pos.items():
            fighter_pos[f] = (ax + xx * scale, ay + yy * scale)

    # Multi-division fighters: equal-weighted centroid + jitter
    for f, divs in fighter_divs.items():
        if f in fighter_pos:
            continue
        if len(divs) < 2:
            continue
        xs = [anchor_pos[d][0] for d in divs if d in anchor_pos]
        ys = [anchor_pos[d][1] for d in divs if d in anchor_pos]
        if not xs:
            continue
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        jitter_rng = random.Random((seed ^ (hash(f) & 0xFFFFFFFF)) & 0xFFFFFFFF)
        jx = jitter_rng.uniform(-40, 40)
        jy = jitter_rng.uniform(-40, 40)
        fighter_pos[f] = (cx + jx, cy + jy)

    pool = pool[pool["Fighter"].isin(fighter_pos)].reset_index(drop=True)
    if pool.empty:
        st.info("No fighters could be positioned.")
        return

    # ---- Build ECharts options ----------------------------------------
    # Single graph series. Halo nodes painted first (behind), fighter
    # nodes after (on top). Edges between fighters only.
    nodes = []

    for div in active_divs:
        ax, ay = anchor_pos[div]
        ar = anchor_radius[div]
        base_color = DIVISION_COLORS.get(div, DEFAULT_DIV_COLOR)
        r, g, b = _hex_to_rgb(base_color)
        gradient = {
            "type": "radial",
            "x": 0.5, "y": 0.5, "r": 0.5,
            "colorStops": [
                {"offset": 0.0, "color": f"rgba({r},{g},{b},0.85)"},
                {"offset": 0.5, "color": f"rgba({r},{g},{b},0.30)"},
                {"offset": 1.0, "color": f"rgba({r},{g},{b},0.0)"},
            ],
        }
        nodes.append({
            "id": f"__halo__{div}",
            "name": div,
            "x": ax,
            "y": ay,
            "fixed": True,
            "symbolSize": ar * 2,
            "itemStyle": {"color": gradient, "borderWidth": 0},
            "label": {
                "show": True,
                "position": "inside",
                "color": "rgba(255,255,255,0.55)",
                "fontSize": 12,
                "fontWeight": 700,
                "formatter": div,
            },
            "emphasis": {"disabled": True},
            # Don't dim halos when a fighter is hovered: re-apply the same
            # gradient + opacity in the blur state.
            "blur": {
                "itemStyle": {"color": gradient, "opacity": 1.0},
                "label": {"show": True, "color": "rgba(255,255,255,0.45)"},
            },
            "tooltip": {"show": False},
            "silent": True,
        })

    for name in pool["Fighter"]:
        x, y = fighter_pos[name]
        elo = float(name_to_elo[name])
        n_fights = int(fight_count_map.get(name, 0))
        is_multi = len(fighter_divs[name]) > 1
        nodes.append({
            "id": name,
            "name": name,
            "x": x,
            "y": y,
            "fixed": True,
            "symbolSize": max(10, min(40, 6 + n_fights)),
            "itemStyle": {
                "color": _elo_gradient_color(elo, elo_lo, elo_hi),
                "borderColor": "#FFFFFF" if not is_multi else "#FFD166",
                "borderWidth": 0.6 if not is_multi else 1.6,
            },
            "label": {
                "show": name in labelled,
                "position": "top",
                "color": "#FAFAFA",
                "fontSize": 11,
            },
            "tooltip": {
                "formatter": (
                    f"<b>{name}</b><br/>"
                    f"Peak Elo: {elo:.1f}<br/>"
                    f"UFC fights: {n_fights}<br/>"
                    f"Division: {name_to_div_raw[name]}"
                ),
            },
        })

    edges = []
    for a, b in h2h_pairs:
        if a in fighter_pos and b in fighter_pos:
            edges.append({"source": a, "target": b})

    options = {
        "backgroundColor": "transparent",
        "tooltip": {"trigger": "item"},
        "series": [{
            "type": "graph",
            "layout": "none",
            "roam": True,
            "scaleLimit": {"min": 0.3, "max": 8},
            "data": nodes,
            "edges": edges,
            "lineStyle": {
                "color": "rgba(255,255,255,0.40)",
                "width": 1,
                "opacity": 0.55,
                "curveness": 0.0,
            },
            "label": {"position": "top"},
            "emphasis": {
                "focus": "adjacency",
                "lineStyle": {
                    "color": "rgba(255,255,255,0.95)",
                    "width": 2.5,
                },
                "label": {"show": True, "color": "#FAFAFA"},
                "itemStyle": {
                    "shadowBlur": 18,
                    "shadowColor": "rgba(255,255,255,0.6)",
                },
            },
            "blur": {
                "itemStyle": {"opacity": 0.25},
                "lineStyle": {"opacity": 0.06},
                "label": {"show": False},
            },
        }],
    }

    # ---- Render via vanilla ECharts (st.components.v1.html) -----------
    chart_col, side_col = st.columns([3, 1], gap="medium")
    chart_height = 720

    with chart_col:
        try:
            options_json = json.dumps(options, default=str)
        except Exception as e:
            st.error(f"Failed to serialise chart options: {e}")
            return
        template = """<!DOCTYPE html>
<html>
<head>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<style>
    html, body { margin: 0; padding: 0; background: transparent; overflow: hidden; }
    #chart { width: 100%; height: __HEIGHT__px; }
</style>
</head>
<body>
<div id="chart"></div>
<script>
const options = __OPTIONS_JSON__;
options.toolbox = {
  show: true,
  right: 12,
  top: 8,
  itemSize: 18,
  itemGap: 10,
  iconStyle: { borderColor: '#9aa0a6', borderWidth: 1.4 },
  emphasis: { iconStyle: { borderColor: '#FAFAFA' } },
  feature: {
    myZoomIn: {
      show: true,
      title: 'Zoom in',
      icon: 'path://M19,13H13V19H11V13H5V11H11V5H13V11H19V13Z',
      onclick: function(ecModel, api) {
        api.dispatchAction({
          type: 'graphRoam',
          seriesIndex: 0,
          zoom: 1.5,
          originX: api.getWidth() / 2,
          originY: api.getHeight() / 2
        });
      }
    },
    myZoomOut: {
      show: true,
      title: 'Zoom out',
      icon: 'path://M19,13H5V11H19V13Z',
      onclick: function(ecModel, api) {
        api.dispatchAction({
          type: 'graphRoam',
          seriesIndex: 0,
          zoom: 0.667,
          originX: api.getWidth() / 2,
          originY: api.getHeight() / 2
        });
      }
    },
    restore: { title: 'Reset view' },
    saveAsImage: {
      title: 'Save as image',
      name: 'fight_network',
      backgroundColor: '#0E1117',
      pixelRatio: 2
    }
  }
};
const chart = echarts.init(document.getElementById('chart'), null, { renderer: 'canvas' });
chart.setOption(options);
window.addEventListener('resize', () => chart.resize());
</script>
</body>
</html>"""
        html = (
            template
            .replace("__OPTIONS_JSON__", options_json)
            .replace("__HEIGHT__", str(chart_height))
        )
        st.components.v1.html(html, height=chart_height + 24)

    with side_col:
        st.markdown(
            "<h4 style='letter-spacing:0.06em;font-weight:700;"
            "margin-top:0;'>FIGHTER DETAILS</h4>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Vanilla ECharts can't push click events back to Python — "
            "use the dropdown to inspect a fighter. (Hover the chart "
            "for quick info.)"
        )
        jump_choice = st.selectbox(
            "Inspect a fighter",
            options=sorted(pool["Fighter"].tolist(), key=str.casefold),
            index=None,
            placeholder="Type a name…",
            label_visibility="collapsed",
            key="fn_jump_fighter",
        )
        if jump_choice and jump_choice in fighter_pos:
            with st.container(border=True):
                st.markdown(f"**{jump_choice}**")
                m1, m2 = st.columns(2)
                m1.metric(
                    "Peak Elo",
                    f"{float(name_to_elo[jump_choice]):.1f}",
                )
                m2.metric(
                    "UFC fights",
                    str(int(fight_count_map.get(jump_choice, 0))),
                )
                divs = sorted(fighter_divs[jump_choice])
                st.caption(
                    f"Division{'s' if len(divs) > 1 else ''}: "
                    f"{', '.join(divs)}"
                )
                if st.button(
                    "Open in Fighter Search →",
                    type="primary",
                    use_container_width=True,
                    key=f"fn_open_{jump_choice}",
                ):
                    st.session_state["fs_selected_fighter"] = jump_choice
                    st.session_state["nav_target"] = "Fighter Search"
                    st.rerun()

        st.divider()
        st.caption(
            f"**{len(pool):,} fighters** across **{len(active_divs)} "
            f"division{'s' if len(active_divs) != 1 else ''}**."
        )
        multi_div_count = sum(
            1 for divs in fighter_divs.values() if len(divs) > 1
        )
        if multi_div_count:
            st.caption(
                f"**{multi_div_count} multi-division** fighter"
                f"{'s' if multi_div_count != 1 else ''} bridge "
                "divisions (gold border)."
            )


# DEPRECATED — the original streamlit-echarts force-directed implementation,
# kept temporarily so we can revert quickly if the rebuilt page_fight_network
# above misbehaves. Safe to delete once the new layout is verified.
def _OLD_page_fight_network_DEPRECATED(
    peaks: pd.DataFrame,
    engine: UFCEloEngine,
    *,
    show_header: bool = True,
) -> None:
    # show_header is False when this page is rendered inside a tab on the
    # Stats Visualiser page — the tab label already names the view, so the
    # large H1 would be visually redundant.
    if show_header:
        st.title("Fight Network")
    st.caption(
        "Force-directed ECharts graph of fighters connected by their "
        "head-to-head bouts. 11 invisible anchor points (one per shared "
        "weight class — men's and women's of the same weight share a "
        "single anchor) sit on a loose grid across the canvas: lightest "
        "top-left, heaviest bottom-right. Each fighter is pulled toward "
        "the division(s) they fought in. Hovering a fighter softly glows "
        "the backdrop of every division they competed in."
    )

    import math

    # Per-shared-division colour palette — used by the radial-gradient
    # halo backdrops. Tuned so adjacent classes don't bleed into each
    # other visually when their halos overlap.
    DIVISION_COLORS = {
        "Strawweight":       "#FF4D6D",
        "Flyweight":         "#FF9E00",
        "Bantamweight":      "#FFD166",
        "Featherweight":     "#A8E10C",
        "Lightweight":       "#06D6A0",
        "Welterweight":      "#118AB2",
        "Middleweight":      "#5A189A",
        "Light Heavyweight": "#9D4EDD",
        "Heavyweight":       "#E5383B",
        "Open Weight":       "#8D99AE",
        "Catch Weight":      "#6C757D",
    }
    DEFAULT_DIV_COLOR = "#888888"

    def _shared_div(wc: object) -> str:
        """Strip the 'Womens ' prefix so men's and women's of the same
        weight share a single anchor (e.g. men's Bantamweight + women's
        Bantamweight collapse to 'Bantamweight')."""
        if isinstance(wc, str) and wc.startswith("Womens "):
            return wc[len("Womens "):]
        return wc if isinstance(wc, str) else ""

    def _hex_to_rgb(h: str) -> tuple:
        h = h.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    def _grid_positions(divisions: list) -> dict:
        """Lay anchors out on a roughly square grid that fills the canvas.

        Lightest at top-left, heaviest at bottom-right (row-major). When
        the division list shrinks (filters narrow the pool), the grid
        rebalances so the remaining anchors still spread evenly across
        the canvas instead of clumping in one corner."""
        n = len(divisions)
        if n == 0:
            return {}
        # Bias slightly wider than tall — most screens are landscape.
        cols = max(1, math.ceil(math.sqrt(n * 4 / 3)))
        rows = max(1, math.ceil(n / cols))
        # Tightened from 900x600 — the original canvas was so wide that
        # the chart auto-fit shrank dense pools (e.g. Top 200) into a tiny
        # cluster. Anchors now sit closer together so the fighter graph
        # naturally fills more of the visible chart area.
        CW, CH = 420.0, 300.0
        cell_w = 2 * CW / cols
        cell_h = 2 * CH / rows
        pos: dict = {}
        for i, d in enumerate(divisions):
            r = i // cols
            c = i % cols
            pos[d] = (
                -CW + cell_w * (c + 0.5),
                -CH + cell_h * (r + 0.5),
            )
        return pos

    def _gather_div_counts(fighters_set: set) -> dict:
        """Map fighter -> {shared division: bouts in that division}."""
        counts: dict = {}
        for f in engine.fight_deltas:
            a, b = f["fighter_a"], f["fighter_b"]
            wc = _shared_div(f["weight_class"])
            if not wc:
                continue
            for who in (a, b):
                if who in fighters_set:
                    counts.setdefault(who, {})
                    counts[who][wc] = counts[who].get(wc, 0) + 1
        return counts

    # ---- Filter row 1: gender radio + division multiselect -------------
    gender_mode = st.radio(
        "Gender",
        options=["All", "Men's", "Women's"],
        index=0,
        horizontal=True,
        key="fn_gender_mode",
    )

    canonical_shared = list(CANONICAL_WEIGHT_CLASSES)
    if gender_mode == "Men's":
        present_shared = {
            _shared_div(d) for d in peaks["Weight Class"]
            if isinstance(d, str) and not d.startswith("Womens ")
        }
    elif gender_mode == "Women's":
        present_shared = {
            _shared_div(d) for d in peaks["Weight Class"]
            if isinstance(d, str) and d.startswith("Womens ")
        }
    else:
        present_shared = {
            _shared_div(d) for d in peaks["Weight Class"]
            if isinstance(d, str)
        }
    eligible_divs = [d for d in canonical_shared if d in present_shared]

    selected_divs = st.multiselect(
        "Divisions",
        options=eligible_divs,
        default=eligible_divs,
        key=f"fn_div_select_{gender_mode}",
        help=(
            "Each division is shared by men's and women's fighters of "
            "the same weight. Empty selection falls back to all divisions "
            "in the chosen gender pool. Fighters whose bouts all fall "
            "outside the selection are hidden entirely."
        ),
    )
    if not selected_divs:
        selected_divs = eligible_divs

    # ---- Filter row 2: Top N + label rule ------------------------------
    slider_col, label_col = st.columns([4, 1], gap="medium")
    with slider_col:
        n = st.slider(
            "Top N by peak Elo (within filtered pool)",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help=(
                "Top N is applied AFTER the gender / division filters, so "
                "the slider always pulls the strongest N fighters from the "
                "current pool."
            ),
        )
    with label_col:
        label_choice = st.selectbox(
            "Labels",
            options=["10", "15", "25", "All"],
            index=1,
            help=(
                "How many fighters (by peak Elo) get a permanent name "
                "label. Hovering still reveals the rest."
            ),
        )

    # ---- Filter the fighter pool --------------------------------------
    selected_set = set(selected_divs)
    is_womens = peaks["Weight Class"].astype(str).str.startswith("Womens ")
    if gender_mode == "Men's":
        gender_mask = ~is_womens
    elif gender_mode == "Women's":
        gender_mask = is_womens
    else:
        gender_mask = pd.Series(True, index=peaks.index)
    div_mask = (
        peaks["Weight Class"]
        .astype(str)
        .map(_shared_div)
        .isin(selected_set)
    )
    pool = (
        peaks[gender_mask & div_mask]
        .copy()
        .head(n)
        .reset_index(drop=True)
    )
    fighters = set(pool["Fighter"])
    div_counts = _gather_div_counts(fighters)
    fighters = {
        f for f in fighters
        if any(d in selected_set for d in div_counts.get(f, {}))
    }
    pool = pool[pool["Fighter"].isin(fighters)].reset_index(drop=True)

    if pool.empty:
        st.info("No fighters match the current filters.")
        return

    name_to_elo = dict(zip(pool["Fighter"], pool["Peak Elo"]))
    name_to_div_raw = dict(zip(pool["Fighter"], pool["Weight Class"]))
    fight_count_map = engine.fight_counts()
    elo_lo = float(pool["Peak Elo"].min())
    elo_hi = float(pool["Peak Elo"].max())

    if label_choice == "All":
        labelled = set(pool["Fighter"])
    else:
        k = int(label_choice)
        labelled = set(
            pool.sort_values("Peak Elo", ascending=False)
            .head(k)["Fighter"]
            .tolist()
        )

    # ---- ECharts nodes -------------------------------------------------
    # Anchors come FIRST in the data array so fighter nodes paint on top
    # of them within the same graph series.
    nodes: list = []
    anchor_pos = _grid_positions(selected_divs)

    # Invisible 'frame' corner nodes. These pin the auto-fit bounding
    # box to the full anchor canvas. Without them, ECharts fits only to
    # the (much smaller) fighter cluster's centroid and the division
    # anchors drift offscreen, which is what produced the 'cluster
    # stuck in the corner' default view in earlier iterations.
    CANVAS_W, CANVAS_H = 420.0, 300.0
    for fx, fy in [
        (-CANVAS_W, -CANVAS_H),
        ( CANVAS_W, -CANVAS_H),
        (-CANVAS_W,  CANVAS_H),
        ( CANVAS_W,  CANVAS_H),
    ]:
        nodes.append({
            "id": f"__frame__{fx}_{fy}",
            "name": "",
            "x": fx,
            "y": fy,
            "fixed": True,
            "symbolSize": 1,
            "itemStyle": {"opacity": 0},
            "label": {"show": False},
            "emphasis": {"disabled": True},
            "tooltip": {"show": False},
            "silent": True,
        })
    for div in selected_divs:
        if div not in anchor_pos:
            continue
        ax, ay = anchor_pos[div]
        base_color = DIVISION_COLORS.get(div, DEFAULT_DIV_COLOR)
        r, g, b = _hex_to_rgb(base_color)
        nodes.append({
            "id": f"__anchor__{div}",
            "name": div,
            "x": ax,
            "y": ay,
            "fixed": True,
            "symbolSize": 160,
            "itemStyle": {
                # Radial gradient: solid at the centre, fading to fully
                # transparent at the edge — gives the halo a soft, blurred
                # appearance with no hard circle outline.
                "color": {
                    "type": "radial",
                    "x": 0.5,
                    "y": 0.5,
                    "r": 0.5,
                    "colorStops": [
                        {"offset": 0.0, "color": f"rgba({r},{g},{b},1.0)"},
                        {"offset": 0.55, "color": f"rgba({r},{g},{b},0.45)"},
                        {"offset": 1.0, "color": f"rgba({r},{g},{b},0.0)"},
                    ],
                },
                # Ambient state: ~5% — just barely visible as a tint.
                "opacity": 0.05,
                "borderWidth": 0,
            },
            "label": {
                "show": True,
                "position": "inside",
                "color": "rgba(255,255,255,0.30)",
                "fontSize": 11,
                "fontWeight": 600,
            },
            "emphasis": {
                # Hover state: visible glow but still subtle, so the
                # fighter nodes remain the focus.
                "itemStyle": {"opacity": 0.40},
                "label": {"show": True, "color": "rgba(255,255,255,0.85)"},
            },
            "blur": {
                # When ANOTHER fighter is hovered, halos that aren't
                # adjacent should stay at ambient — not be brightened by
                # the series-level blur fallback (which would actually
                # raise their opacity above 0.05).
                "itemStyle": {"opacity": 0.05},
                "label": {"show": True, "color": "rgba(255,255,255,0.20)"},
            },
            "tooltip": {"show": False},
        })

    for name in pool["Fighter"]:
        elo = float(name_to_elo[name])
        n_fights = int(fight_count_map.get(name, 0))
        nodes.append({
            "id": name,
            "name": name,
            "symbolSize": max(10, min(40, 6 + n_fights)),
            "itemStyle": {
                "color": _elo_gradient_color(elo, elo_lo, elo_hi),
                "borderColor": "#fff",
                "borderWidth": 0.5,
            },
            "label": {
                "show": name in labelled,
                "position": "top",
                "color": "#FAFAFA",
                "fontSize": 11,
            },
            "value": [round(elo, 1), n_fights, name_to_div_raw[name]],
        })

    # ---- ECharts edges -------------------------------------------------
    pair_counts: dict = {}
    for f in engine.fight_deltas:
        a, b = f["fighter_a"], f["fighter_b"]
        if a in fighters and b in fighters:
            key = tuple(sorted([a, b]))
            pair_counts[key] = pair_counts.get(key, 0) + 1

    edges: list = []
    for (a, b), n_bouts in pair_counts.items():
        edges.append({
            "source": a,
            "target": b,
            # Low value → short target length: opponents cluster tightly.
            "value": 1,
            "lineStyle": {
                "color": "#666",
                "width": 1 + (n_bouts - 1) * 2.0,
                "opacity": 0.55,
            },
        })

    # Invisible fighter -> anchor edges. One edge per bout in that
    # division gives the multi-class settling effect (proportional pull)
    # AND wires fighters as 'adjacent' to their division anchors so the
    # adjacency emphasis lights up the right halos on hover.
    for fighter, divs in div_counts.items():
        if fighter not in fighters:
            continue
        for div, count in divs.items():
            if div not in selected_set:
                continue
            for _ in range(count):
                edges.append({
                    "source": fighter,
                    "target": f"__anchor__{div}",
                    # High value → long target length: pulls fighters
                    # out toward their division anchor across the canvas.
                    "value": 10,
                    "lineStyle": {"opacity": 0, "width": 0},
                })

    options = {
        "backgroundColor": "transparent",
        # Floating toolbox in the top-right of the chart. Two custom
        # buttons dispatch ECharts' graphRoam action for incremental
        # zoom in/out; the built-in restore + saveAsImage round it out.
        # `roam: True` on the series still enables wheel-zoom and
        # click-drag panning for power users.
        "toolbox": {
            "show": True,
            "right": 12,
            "top": 8,
            "orient": "horizontal",
            "itemSize": 18,
            "itemGap": 10,
            "iconStyle": {
                "borderColor": "#9aa0a6",
                "borderWidth": 1.4,
            },
            "emphasis": {
                "iconStyle": {"borderColor": "#FAFAFA"},
            },
            "feature": {
                "myZoomIn": {
                    "show": True,
                    "title": "Zoom in",
                    "icon": (
                        "path://M19,13H13V19H11V13H5V11H11V5H13V11H19V13Z"
                    ),
                    # ECharts calls toolbox onclick as
                    # onclick(ecModel, api) — `this` is the feature model,
                    # NOT the chart instance, so dispatchAction must be
                    # called on the `api` argument instead. The api also
                    # exposes getWidth / getHeight for the zoom origin.
                    "onclick": JsCode(
                        "function(ecModel, api){"
                        " api.dispatchAction({type:'graphRoam',"
                        " seriesIndex: 0, zoom: 1.5,"
                        " originX: api.getWidth()/2,"
                        " originY: api.getHeight()/2});"
                        "}"
                    ),
                },
                "myZoomOut": {
                    "show": True,
                    "title": "Zoom out",
                    "icon": "path://M19,13H5V11H19V13Z",
                    "onclick": JsCode(
                        "function(ecModel, api){"
                        " api.dispatchAction({type:'graphRoam',"
                        " seriesIndex: 0, zoom: 0.667,"
                        " originX: api.getWidth()/2,"
                        " originY: api.getHeight()/2});"
                        "}"
                    ),
                },
                "restore": {"title": "Reset view"},
                "saveAsImage": {
                    "title": "Save as image",
                    "name": "fight_network",
                    "backgroundColor": "#0E1117",
                    "pixelRatio": 2,
                },
            },
        },
        "tooltip": {"trigger": "item"},
        "series": [{
            "type": "graph",
            "layout": "force",
            # No explicit center/zoom — we let ECharts auto-fit to the
            # bounding box of all nodes. The 4 invisible corner 'frame'
            # nodes (added in the data array below) guarantee the bbox
            # always covers the full anchor canvas, so the data origin
            # ends up at the chart's geometric centre and the division
            # halos stay onscreen no matter how the fighters cluster.
            "force": {
                # Tuning targets:
                #   repulsion 500     — strong push between non-adjacent
                #                       fighters so the graph spreads out.
                #   edgeLength 100/500 — short for head-to-head ties
                #                        (opponents cluster), long for
                #                        fighter→anchor virtual edges
                #                        (fighters pulled far toward their
                #                        division zone).
                #   gravity 0.2        — firm pull toward chart centre so
                #                        the cluster stays framed even
                #                        when divisions are asymmetric.
                #   friction 1.0       — maximal damping. NOTE: in ECharts
                #                        higher friction = slower motion;
                #                        at 1.0 the layout may settle so
                #                        slowly it looks frozen. If that
                #                        happens, drop back toward 0.6.
                "repulsion": 500,
                "edgeLength": [100, 500],
                "gravity": 0.2,
                "friction": 1.0,
                "layoutAnimation": True,
            },
            "roam": True,
            "draggable": True,
            # Clamp roam zoom so users can't shrink the graph past the
            # point of no return, or magnify a single node off-screen.
            "scaleLimit": {"min": 0.5, "max": 8},
            "data": nodes,
            "edges": edges,
            "label": {"position": "top"},
            "lineStyle": {"opacity": 0.55},
            "emphasis": {
                "focus": "adjacency",
                "lineStyle": {"width": 3, "opacity": 0.9},
                "label": {"show": True},
                "itemStyle": {
                    "shadowBlur": 20,
                    "shadowColor": "rgba(255, 255, 255, 0.4)",
                },
            },
            "blur": {
                "itemStyle": {"opacity": 0.15},
                "lineStyle": {"opacity": 0.05},
                "label": {"show": False},
            },
        }],
    }

    # ---- Render: chart on the left, fighter side-panel on the right ---
    chart_col, side_col = st.columns([3, 1], gap="medium")
    with chart_col:
        clicked = st_echarts(
            options=options,
            events={
                "click": (
                    "function(p){"
                    " return {dataType: p.dataType,"
                    " name: p.data && p.data.name,"
                    " id: p.data && p.data.id};"
                    "}"
                ),
            },
            height="600px",
            key=f"fn_echart_{gender_mode}_{n}_{len(selected_divs)}",
        )

    with side_col:
        st.markdown(
            "<h4 style='letter-spacing:0.06em;font-weight:700;"
            "margin-top:0;'>FIGHTER DETAILS</h4>",
            unsafe_allow_html=True,
        )
        picked = None
        if isinstance(clicked, dict) and clicked.get("dataType") == "node":
            cid = clicked.get("id")
            cand = clicked.get("name")
            if (
                isinstance(cid, str)
                and not cid.startswith("__anchor__")
                and isinstance(cand, str)
            ):
                picked = cand
        if picked and picked in fighters:
            with st.container(border=True):
                st.markdown(f"**{picked}**")
                m1, m2 = st.columns(2)
                m1.metric("Peak Elo", f"{float(name_to_elo[picked]):.1f}")
                m2.metric(
                    "UFC fights",
                    str(int(fight_count_map.get(picked, 0))),
                )
                st.caption(f"Division: {name_to_div_raw[picked]}")
                if st.button(
                    "Open in Fighter Search →",
                    type="primary",
                    use_container_width=True,
                    key=f"fn_open_{picked}",
                ):
                    st.session_state["fs_selected_fighter"] = picked
                    st.session_state["nav_target"] = "Fighter Search"
                    st.rerun()
        else:
            st.caption("Click any node to see fighter details here.")

        st.divider()
        st.caption("Or jump directly:")
        jump_choice = st.selectbox(
            "Fighter",
            options=sorted(pool["Fighter"].tolist(), key=str.casefold),
            index=None,
            placeholder="Type a name…",
            label_visibility="collapsed",
            key="fn_jump_fighter",
        )
        if st.button(
            "Open →",
            disabled=jump_choice is None,
            use_container_width=True,
            key="fn_open_fighter_search",
        ):
            st.session_state["fs_selected_fighter"] = jump_choice
            st.session_state["nav_target"] = "Fighter Search"
            st.rerun()

    # ---- Legend strip --------------------------------------------------
    legend_cols = st.columns(2)
    with legend_cols[0]:
        st.markdown(
            "<div style='font-size:0.78rem;opacity:0.7;'>"
            "<strong>Node size</strong> — number of UFC fights"
            "</div>",
            unsafe_allow_html=True,
        )
    with legend_cols[1]:
        st.markdown(
            f"<div style='font-size:0.78rem;opacity:0.7;'>"
            f"<strong>Node colour</strong> — peak Elo: "
            f"<span style='color:#1f77b4'>{elo_lo:.0f}</span> → "
            f"<span style='color:#E03131'>{elo_hi:.0f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def page_career_landscape(
    peaks: pd.DataFrame,
    engine: UFCEloEngine,
    *,
    show_header: bool = True,
) -> None:
    # See note on page_fight_network: title is suppressed when tabbed.
    if show_header:
        st.title("Career Landscape")
    st.caption(
        "Each dot is one fighter — x-axis is UFC fights, y-axis is peak Elo. "
        "**Click any dot** to open that fighter in Fighter Search."
    )

    fight_count_map = engine.fight_counts()
    df = peaks.copy()
    df["Fights"] = (
        df["Fighter"].map(fight_count_map).fillna(0).astype(int)
    )
    df = df[df["Fights"] > 0].copy().reset_index(drop=True)
    df["Color"] = df["Fighter"].apply(fighter_color)

    scatter_fig = go.Figure()
    scatter_fig.add_trace(
        go.Scatter(
            x=df["Fights"],
            y=[round(e, 1) for e in df["Peak Elo"]],
            mode="markers",
            marker=dict(
                size=10,
                color=df["Color"].tolist(),
                line=dict(color="white", width=0.5),
                opacity=0.85,
            ),
            customdata=list(zip(
                df["Fighter"].astype(str),
                df["Weight Class"].astype(str),
            )),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Weight class: %{customdata[1]}<br>"
                "Fights: %{x}<br>"
                "Peak Elo: %{y:.1f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    scatter_fig.update_layout(
        height=600,
        xaxis_title="UFC fights",
        yaxis_title="Peak Elo",
        margin=dict(t=30, b=40, l=50, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        hovermode="closest",
    )
    scatter_fig.update_xaxes(gridcolor="#1A1D24", zerolinecolor="#1A1D24")
    scatter_fig.update_yaxes(gridcolor="#1A1D24", zerolinecolor="#1A1D24")

    # Native Streamlit plotly click events (Streamlit >= 1.39).
    landscape_event = st.plotly_chart(
        scatter_fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="career_landscape_chart",
    )

    clicked_points = []
    if landscape_event and getattr(landscape_event, "selection", None):
        clicked_points = landscape_event.selection.get("points", []) or []

    if clicked_points:
        point = clicked_points[0]
        idx = point.get("point_index")
        if idx is None:
            idx = point.get("pointIndex")
        if idx is not None and 0 <= int(idx) < len(df):
            picked = str(df.iloc[int(idx)]["Fighter"])
            st.session_state["fs_selected_fighter"] = picked
            st.session_state["nav_target"] = "Fighter Search"
            st.rerun()

    # Manual fallback dropdown — handy when dots are clustered too tightly
    # to click cleanly.
    st.markdown("**Or jump to a fighter manually:**")
    jump_col, btn_col = st.columns([4, 1], gap="medium")
    with jump_col:
        jump_choice = st.selectbox(
            "Fighter",
            options=sorted(df["Fighter"].tolist(), key=str.casefold),
            index=None,
            placeholder="Start typing a name…",
            label_visibility="collapsed",
            key="cl_jump_fighter",
        )
    with btn_col:
        if st.button(
            "Open in Fighter Search →",
            disabled=jump_choice is None,
            type="primary",
            use_container_width=True,
        ):
            st.session_state["fs_selected_fighter"] = jump_choice
            st.session_state["nav_target"] = "Fighter Search"
            st.rerun()


def page_top_10_rankings(peaks: pd.DataFrame, engine: UFCEloEngine) -> None:
    """Tabbed home for both Top 10 ranking views.

    Mirrors the Stats Visualiser / round-by-round pattern: a single page
    title + caption, then `st.tabs(...)` with one tab per ranking view.
    Each tab calls into the existing page function with `show_header=False`
    so the inner H1 doesn't duplicate the tab label.
    """
    record_nav_step("Top 10 Rankings")
    st.title("Top 10 Rankings")
    st.caption(
        "Peak Elo leaderboards. Overall lists the top 10 across the entire "
        "dataset; Per Weight Class breaks the same metric down by the "
        "weight class in which each fighter hit their peak."
    )
    tabs = st.tabs(["Overall", "Per Weight Class"])
    with tabs[0]:
        page_overall(peaks, engine, show_header=False)
    with tabs[1]:
        page_by_weight_class(peaks, show_header=False)


def page_stats_visualiser(peaks: pd.DataFrame, engine: UFCEloEngine) -> None:
    """Tabbed home for both visualiser pages.

    Mirrors the round-by-round tab pattern in Fight Finder: a single page
    title + caption, then `st.tabs(...)` with one tab per visualisation.
    Each tab calls into the existing page function with `show_header=False`
    so the inner H1 doesn't duplicate the tab label.
    """
    record_nav_step("Stats Visualiser")
    st.title("Stats Visualiser")
    st.caption(
        "Two ways to explore the dataset visually. Career Landscape plots "
        "every fighter as a dot (UFC fights vs. peak Elo); Fight Network "
        "draws a force-directed graph of head-to-head bouts among the top "
        "fighters."
    )
    tabs = st.tabs(["Career Landscape", "Fight Network"])
    with tabs[0]:
        page_career_landscape(peaks, engine, show_header=False)
    with tabs[1]:
        page_fight_network(peaks, engine, show_header=False)


def page_settings() -> None:
    record_nav_step("Settings")
    st.title("Settings")
    st.caption(
        "Tune the Elo engine's parameters and watch the rankings update. "
        "All values reset to the code defaults every time the app launches."
    )

    with st.container(border=True):
        st.markdown("### What is Elo?")
        st.markdown(
            "Elo is a rating system originally invented for chess that "
            "estimates each competitor's skill from match outcomes. Every "
            "fighter starts at the same **base rating**. After each fight "
            "the winner takes points from the loser; how many points change "
            "hands is governed by the **K-factor** and by how surprising the "
            "result was — beating a much higher-rated opponent swings the "
            "ratings more than beating an equal one. Run across 30 years of "
            "UFC data, this produces a *peak Elo* per fighter, which the "
            "rest of the app ranks and visualises."
        )

    st.markdown("### Engine parameters")
    st.caption(
        "Edit any of the four values below. The leaderboard recomputes "
        "immediately — a brief spinner means Streamlit is rebuilding the "
        "engine for this combination of parameters."
    )

    # value=... explicitly seeds each input with the code default on first
    # render. Streamlit then writes the chosen value back into
    # session_state[key] so the leaderboard recomputes against the user's
    # tweaks. Reset button below clears the keys to fall back to value=.
    col1, col2 = st.columns(2)
    with col1:
        st.number_input(
            "K-factor",
            min_value=1.0,
            max_value=200.0,
            value=float(K_FACTOR),
            step=1.0,
            key="elo_k",
            help=(
                "Controls how reactive the rating system is. Higher K = "
                "each fight moves the rating more, so fighters rise and "
                "fall faster. Lower K = ratings are more stable but slower "
                "to recognise new talent. Default: 32 (chess-standard)."
            ),
        )
        st.number_input(
            "Base rating",
            min_value=0.0,
            max_value=5000.0,
            value=float(BASE_RATING),
            step=50.0,
            key="elo_base",
            help=(
                "The starting rating every fighter is assigned before their "
                "UFC debut. Changing it shifts the absolute numbers up or "
                "down but does not change anyone's relative ranking. "
                "Default: 1500."
            ),
        )
    with col2:
        # Toggle above each multiplier so users can disable the bonus
        # entirely. When off, the input is greyed out and main() passes
        # 1.0 to the engine for that fight category.
        finish_enabled = st.toggle(
            "Apply finish multiplier",
            value=True,
            key="elo_finish_enabled",
            help=(
                "Off = KO/TKO/SUB wins use the base K-factor with no bonus. "
                "On = the multiplier below scales K for those finishes."
            ),
        )
        st.number_input(
            "Finish multiplier (KO/TKO/SUB)",
            min_value=0.5,
            max_value=3.0,
            value=float(FINISH_MULTIPLIER),
            step=0.05,
            format="%.2f",
            key="elo_finish",
            disabled=not finish_enabled,
            help=(
                "Amplifies the K-factor when a fight ends by knockout, TKO, "
                "or submission. Higher values reward finishers more heavily, "
                "so decision-heavy careers will rank lower relative to "
                "finishers. Default: 1.20. Toggle off to disable the bonus."
            ),
        )
        decision_enabled = st.toggle(
            "Apply decision multiplier",
            value=True,
            key="elo_decision_enabled",
            help=(
                "Off = decision wins use the base K-factor with no boost. "
                "On = the multiplier below scales K for decision finishes."
            ),
        )
        st.number_input(
            "Decision multiplier",
            min_value=0.5,
            max_value=3.0,
            value=float(DECISION_MULTIPLIER),
            step=0.05,
            format="%.2f",
            key="elo_decision",
            disabled=not decision_enabled,
            help=(
                "Multiplier applied when a fight goes to the judges. Set "
                "below 1.0 to discount decision wins relative to finishes; "
                "set above 1.0 to reward grinding it out. Default: 1.00. "
                "Toggle off to disable the multiplier entirely."
            ),
        )

    st.divider()
    btn_left, btn_right = st.columns([1, 4])
    with btn_left:
        if st.button("Reset to defaults", use_container_width=True):
            # Delete the keys so the widgets fall back to their value=
            # defaults on the next render. This is the recommended
            # Streamlit reset pattern: assigning to a widget's key after
            # it has been instantiated raises an error, but popping the
            # key entirely is allowed.
            for k in (
                "elo_k",
                "elo_base",
                "elo_finish",
                "elo_decision",
                "elo_finish_enabled",
                "elo_decision_enabled",
            ):
                st.session_state.pop(k, None)
            st.rerun()
    with btn_right:
        st.caption(
            f"Code defaults: K={K_FACTOR}, base={BASE_RATING}, "
            f"finish={FINISH_MULTIPLIER}, decision={DECISION_MULTIPLIER}."
        )

    # Current configuration rendered as four shadcn metric cards so the
    # Settings page matches the Tale of the Tape / Stats & Records visual
    # language used everywhere else in the app.
    st.markdown(
        "<h3 style='letter-spacing:0.08em;font-weight:800;"
        "margin-top:1.5rem;margin-bottom:0.6rem;'>CURRENT CONFIGURATION</h3>",
        unsafe_allow_html=True,
    )
    # 2x2 grid biased to the left: outer split keeps the cards compact
    # (~40% of the page width) instead of stretching across the full row.
    cfg_left, _cfg_spacer = st.columns([2, 3])
    with cfg_left:
        row1_a, row1_b = st.columns(2)
        with row1_a:
            ui.metric_card(
                title="K-factor",
                content=f"{st.session_state['elo_k']:.0f}",
                description="Reactivity per fight",
                key="settings_metric_k",
            )
        with row1_b:
            ui.metric_card(
                title="Base rating",
                content=f"{st.session_state['elo_base']:.0f}",
                description="Starting Elo for every fighter",
                key="settings_metric_base",
            )
        # Multiplier cards reflect the toggle state: show "Off" instead
        # of a number when the user has disabled that bonus, with a
        # description that confirms the engine is using base K for that
        # fight category.
        finish_on = st.session_state.get("elo_finish_enabled", True)
        decision_on = st.session_state.get("elo_decision_enabled", True)
        row2_a, row2_b = st.columns(2)
        with row2_a:
            ui.metric_card(
                title="Finish multiplier",
                content=(
                    f"{st.session_state['elo_finish']:.2f}×"
                    if finish_on else "Off"
                ),
                description=(
                    "KO / TKO / Submission boost"
                    if finish_on
                    else "KO/TKO/SUB use base K (no bonus)"
                ),
                key="settings_metric_finish",
            )
        with row2_b:
            ui.metric_card(
                title="Decision multiplier",
                content=(
                    f"{st.session_state['elo_decision']:.2f}×"
                    if decision_on else "Off"
                ),
                description=(
                    "Judges' decision boost"
                    if decision_on
                    else "Decisions use base K (no bonus)"
                ),
                key="settings_metric_decision",
            )


# ---------------------------------------------------------------------------
# Browser-style navigation history (Back / Forward)
# ---------------------------------------------------------------------------
# Tracks only top-nav page changes, so the history doesn't churn on in-page
# interactions (fighter selection, chart clicks, filter tweaks). Each top-
# level page function calls record_nav_step() once, and render_back_forward()
# draws the controls above the nav bar. The Back/Forward buttons set
# nav_target (which option_menu already reads via default_index) plus a
# _nav_restoring flag so the resulting rerun's record_nav_step is suppressed
# — i.e. restoring a snapshot doesn't re-record it as a fresh navigation.

def record_nav_step(page_name: str) -> None:
    """Append a page to the nav-history stack (idempotent on repeats).

    Called at the top of each top-level page function. No-op when we're in
    the middle of restoring a snapshot via the Back/Forward buttons.
    """
    history = st.session_state.setdefault("_nav_history", [])
    index = st.session_state.get("_nav_index", -1)

    if st.session_state.pop("_nav_restoring", False):
        # Just restored from history — don't re-record.
        return

    # Skip if the current page already matches the current entry.
    if 0 <= index < len(history) and history[index] == page_name:
        return

    # Truncate any forward entries (classic browser-history semantics) and push.
    history = history[: index + 1] + [page_name]
    st.session_state["_nav_history"] = history
    st.session_state["_nav_index"] = len(history) - 1


def render_back_forward() -> None:
    """Render ← Back / Forward → buttons above the top nav bar.

    Both buttons set st.session_state["nav_target"] (which option_menu
    already reads via default_index), set _nav_restoring so the next
    record_nav_step() call is suppressed, and rerun.
    """
    history = st.session_state.get("_nav_history", [])
    index = st.session_state.get("_nav_index", -1)

    can_back = index > 0
    can_forward = 0 <= index < len(history) - 1

    back_col, fwd_col, _spacer = st.columns([2, 2, 16])
    with back_col:
        if st.button(
            "← Back",
            key="_nav_back",
            disabled=not can_back,
            use_container_width=True,
            help=(
                f"Back to {history[index - 1]}" if can_back
                else "No previous page"
            ),
        ):
            st.session_state["_nav_index"] = index - 1
            st.session_state["nav_target"] = history[index - 1]
            st.session_state["_nav_restoring"] = True
            st.rerun()
    with fwd_col:
        if st.button(
            "Forward →",
            key="_nav_forward",
            disabled=not can_forward,
            use_container_width=True,
            help=(
                f"Forward to {history[index + 1]}" if can_forward
                else "No next page"
            ),
        ):
            st.session_state["_nav_index"] = index + 1
            st.session_state["nav_target"] = history[index + 1]
            st.session_state["_nav_restoring"] = True
            st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="UFC Peak Elo",
        page_icon="🥊",
        layout="wide",
    )
    inject_global_css()

    # Warm the cardio-score baselines on app launch so the first fighter
    # the user opens renders instantly. build_cardio_baselines() is
    # @st.cache_resource-cached, so this runs once per process and every
    # later call (from page_fighter_search, page_fight_finder, the cardio
    # sort modes, etc.) is an instant cache lookup. Pays the spinner cost
    # up front instead of on first fighter load.
    build_cardio_baselines()

    # Browser-style ← / → buttons above the top nav. Rendered before
    # option_menu so they sit at the top of the page; only top-nav page
    # changes are tracked (see record_nav_step() in each page_* function).
    render_back_forward()

    # Read tunable parameters from session_state, falling back to the
    # code defaults from elo_engine.py on first launch. Because
    # session_state is per-session, the parameters naturally reset to
    # these defaults every time the app is opened.
    #
    # When a multiplier toggle is off, pass 1.0 to the engine so those
    # fights use the base K-factor (no bonus, no penalty).
    finish_mult = (
        float(st.session_state.get("elo_finish", FINISH_MULTIPLIER))
        if st.session_state.get("elo_finish_enabled", True)
        else 1.0
    )
    decision_mult = (
        float(st.session_state.get("elo_decision", DECISION_MULTIPLIER))
        if st.session_state.get("elo_decision_enabled", True)
        else 1.0
    )
    engine = build_engine(
        k=st.session_state.get("elo_k", float(K_FACTOR)),
        base=st.session_state.get("elo_base", float(BASE_RATING)),
        finish_mult=finish_mult,
        decision_mult=decision_mult,
    )
    peaks = engine.peaks_dataframe()

    if peaks.empty:
        st.warning(
            "No fights were processed. Check that the CSV files are "
            "present and non-empty."
        )
        return

    # Top-of-page horizontal nav (streamlit-option-menu) replaces st.tabs.
    options = [
        "Top 10 Rankings",
        "Fighter Search",
        "Fight Finder",
        "Stats Visualiser",
        "Settings",
    ]
    icons = [
        "trophy",
        "search",
        "people-fill",
        "graph-up",
        "sliders",
    ]

    # Programmatic navigation hook: any page can set
    # st.session_state['nav_target'] to a tab name and call st.rerun() to
    # jump there. We consume it here as `manual_select`.
    nav_target = st.session_state.pop("nav_target", None)
    manual_select = (
        options.index(nav_target) if nav_target in options else None
    )

    selected_page = option_menu(
        menu_title=None,
        options=options,
        icons=icons,
        orientation="horizontal",
        default_index=0,
        manual_select=manual_select,
        key="main_nav",
        styles={
            "container": {
                "padding": "4px 0",
                "background-color": "#0E1117",
            },
            "icon": {"color": "#FAFAFA", "font-size": "16px"},
            "nav-link": {
                "font-size": "14px",
                "color": "#FAFAFA",
                "text-align": "center",
                "margin": "0 4px",
                "padding": "10px 14px",
                "border-radius": "8px",
                "--hover-color": "#1A1D24",
            },
            "nav-link-selected": {
                "background-color": "#E03131",
                "color": "#FFFFFF",
                "font-weight": "700",
            },
        },
    )

    if selected_page == "Top 10 Rankings":
        page_top_10_rankings(peaks, engine)
    elif selected_page == "Fighter Search":
        page_fighter_search(peaks, engine)
    elif selected_page == "Fight Finder":
        page_fight_finder(peaks, engine)
    elif selected_page == "Stats Visualiser":
        page_stats_visualiser(peaks, engine)
    elif selected_page == "Settings":
        page_settings()


if __name__ == "__main__":
    main()
