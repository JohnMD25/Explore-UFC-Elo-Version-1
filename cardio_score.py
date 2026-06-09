"""cardio_score.py

Cardio score for UFC fighters. Quantifies how a fighter's round-by-round
work-rate (sig strikes attempted + takedown attempts + control time) trends
across a fight, then maps the across-career trend to a 0–100 index.

Methodology (locked spec):
  1. Per round: W_r = z(SSA/min) + z(TDA/min) + z(ctrl_fraction).
     z() uses dataset-wide per-minute mu/sigma per component.
  2. Drop rounds < 60s. Drop fights ending in round 1 entirely.
  3. Fit log(W_r + c) = log(W_0) - lambda * r via OLS per fight, where c
     is chosen so min(W_r) + c = 1 (keeps log finite, additive on raw scale).
  4. Fighter lambda_bar = sum(lambda_f * R_f) / sum(R_f) across qualifying
     fights, weighted by rounds completed.
  5. CardioScore = 100 / (1 + exp(-z)), z = (-lambda_bar - mu_lam) / sd_lam.
  6. Display rule: requires >= 3 qualifying fights, else 'insufficient sample'.

Higher score = output rises (or holds) late in fights.
Lower score = output decays late.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def _normalise_name(name: object) -> str:
    """Match the normalisation used throughout app.py so cross-CSV joins line up."""
    if not isinstance(name, str):
        return ""
    n = unicodedata.normalize("NFKC", name).strip()
    n = re.sub(r"\s+", " ", n)
    return n.casefold()


def _final_round_minutes(round_value: object, time_value: object):
    """Parse (final_round_int, M:SS_in_final_round) -> (round, minutes_float)."""
    try:
        fr = int(float(str(round_value).strip()))
    except (ValueError, TypeError):
        return None
    m = re.match(r"\s*(\d+):(\d+)", str(time_value))
    if not m:
        return None
    return fr, int(m.group(1)) + int(m.group(2)) / 60.0


def _round_duration_seconds(final_round: int, final_round_min: float, this_round: int) -> float:
    """Duration of a specific round in seconds.

    Rounds before the final round are full 5:00 (300s). The final round is
    `final_round_min` minutes long. Rounds after the final round didn't happen.
    """
    if this_round < final_round:
        return 300.0
    if this_round == final_round:
        return float(final_round_min) * 60.0
    return 0.0


@dataclass
class CardioBaselines:
    """Frozen dataset-wide statistics used to score every fighter."""
    mu_ssa: float
    sd_ssa: float
    mu_tda: float
    sd_tda: float
    mu_ctrl: float
    sd_ctrl: float
    mu_lambda: float
    sd_lambda: float
    fighter_lambdas: dict  # normalised_name -> {'lambda_bar', 'n_fights', 'n_rounds'}


@dataclass
class CardioScore:
    score: Optional[float]   # 0–100, or None if insufficient sample
    lambda_bar: Optional[float]
    n_fights: int
    n_rounds: int
    qualified: bool


def compute_baselines(
    stats_by_round: pd.DataFrame,
    fight_results: pd.DataFrame,
) -> Optional[CardioBaselines]:
    """Compute frozen dataset-wide baselines from per-round stats + fight results.

    Args:
        stats_by_round: from app.load_stats_by_round(). Needs columns EVENT,
            BOUT, fighter_norm, round_num, sig_attempted, td_attempted,
            ctrl_sec.
        fight_results: raw ufc_fight_results.csv. Needs (after column-norm)
            EVENT, BOUT, ROUND, TIME.

    Returns None when there isn't enough data to derive the baselines.
    """
    if stats_by_round.empty or fight_results.empty:
        return None

    fr = fight_results.copy()
    fr.columns = [str(c).strip().upper().replace(" ", "") for c in fr.columns]
    if not {"EVENT", "BOUT", "ROUND", "TIME"} <= set(fr.columns):
        return None

    finals: dict = {}
    for _, row in fr.iterrows():
        parsed = _final_round_minutes(row.get("ROUND"), row.get("TIME"))
        if parsed is None:
            continue
        finals[(str(row["EVENT"]).strip(), str(row["BOUT"]).strip())] = parsed

    qualifying: list = []
    for _, row in stats_by_round.iterrows():
        rn = row.get("round_num")
        if pd.isna(rn):
            continue
        rn = int(rn)
        key = (str(row["EVENT"]), str(row["BOUT"]))
        final = finals.get(key)
        if final is None:
            continue
        final_round, final_time = final
        if final_round < 2:                           # exclude round-1 finishes
            continue
        dur_s = _round_duration_seconds(final_round, final_time, rn)
        if dur_s < 60:                                # drop tiny truncated rounds
            continue
        dur_min = dur_s / 60.0
        ssa = float(row.get("sig_attempted", 0) or 0)
        tda = float(row.get("td_attempted", 0) or 0)
        ctrl = float(row.get("ctrl_sec", 0) or 0)
        qualifying.append({
            "event": key[0], "bout": key[1], "round": rn,
            "fighter": row["fighter_norm"],
            "ssa_pm": ssa / dur_min,
            "tda_pm": tda / dur_min,
            "ctrl_frac": min(ctrl / dur_s, 1.0),
        })

    if not qualifying:
        return None

    qdf = pd.DataFrame(qualifying)

    # Component-wise per-minute mu/sigma drive the W_r z-score sum.
    mu_ssa = float(qdf["ssa_pm"].mean())
    sd_ssa = float(qdf["ssa_pm"].std(ddof=0)) or 1.0
    mu_tda = float(qdf["tda_pm"].mean())
    sd_tda = float(qdf["tda_pm"].std(ddof=0)) or 1.0
    mu_ctrl = float(qdf["ctrl_frac"].mean())
    sd_ctrl = float(qdf["ctrl_frac"].std(ddof=0)) or 1.0

    qdf["W"] = (
        (qdf["ssa_pm"] - mu_ssa) / sd_ssa
        + (qdf["tda_pm"] - mu_tda) / sd_tda
        + (qdf["ctrl_frac"] - mu_ctrl) / sd_ctrl
    )

    # Per-fight OLS slope on log(W + c) ~ a - lambda * r.
    fighter_records: dict = {}
    for (event, bout, fighter), grp in qdf.groupby(["event", "bout", "fighter"]):
        if len(grp) < 2:                              # need >= 2 rounds for a slope
            continue
        c_offset = 1.0 - float(grp["W"].min())
        log_w = np.log(grp["W"].to_numpy() + c_offset)
        rounds = grp["round"].to_numpy(dtype=float)
        slope, _intercept = np.polyfit(rounds, log_w, 1)
        lam = -float(slope)
        fighter_records.setdefault(fighter, []).append((lam, len(grp)))

    fighter_lambdas: dict = {}
    for fighter, recs in fighter_records.items():
        total_r = sum(r for _, r in recs)
        if total_r == 0:
            continue
        lam_bar = sum(l * r for l, r in recs) / total_r
        fighter_lambdas[fighter] = {
            "lambda_bar": lam_bar,
            "n_fights": len(recs),
            "n_rounds": total_r,
        }

    if not fighter_lambdas:
        return None

    lams = [v["lambda_bar"] for v in fighter_lambdas.values()]
    mu_lam = float(np.mean(lams))
    sd_lam = float(np.std(lams, ddof=0)) or 1.0

    return CardioBaselines(
        mu_ssa=mu_ssa, sd_ssa=sd_ssa,
        mu_tda=mu_tda, sd_tda=sd_tda,
        mu_ctrl=mu_ctrl, sd_ctrl=sd_ctrl,
        mu_lambda=mu_lam, sd_lambda=sd_lam,
        fighter_lambdas=fighter_lambdas,
    )


def score_fighter(
    fighter: str,
    baselines: Optional[CardioBaselines],
    *,
    min_fights: int = 3,
) -> CardioScore:
    """Look up `fighter` in the precomputed baselines and return a CardioScore.

    Returns CardioScore(score=None, qualified=False) if the fighter has fewer
    than `min_fights` qualifying fights — the caller should display
    'Insufficient sample' in that case.
    """
    if baselines is None:
        return CardioScore(None, None, 0, 0, False)
    rec = baselines.fighter_lambdas.get(_normalise_name(fighter))
    if rec is None:
        return CardioScore(None, None, 0, 0, False)
    n = rec["n_fights"]
    lam = rec["lambda_bar"]
    if n < min_fights:
        return CardioScore(None, lam, n, rec["n_rounds"], False)
    z = (-lam - baselines.mu_lambda) / baselines.sd_lambda
    return CardioScore(
        score=100.0 / (1.0 + math.exp(-z)),
        lambda_bar=lam,
        n_fights=n,
        n_rounds=rec["n_rounds"],
        qualified=True,
    )


def leaderboard(baselines: Optional[CardioBaselines], *, min_fights: int = 3) -> pd.DataFrame:
    """Convenience: return a sorted DataFrame of every qualifying fighter.

    Columns: fighter_norm, score, lambda_bar, n_fights, n_rounds.
    Useful for sanity-checking the score against known cardio reputations.
    """
    if baselines is None:
        return pd.DataFrame(columns=["fighter_norm", "score", "lambda_bar", "n_fights", "n_rounds"])
    rows = []
    for fighter, rec in baselines.fighter_lambdas.items():
        if rec["n_fights"] < min_fights:
            continue
        z = (-rec["lambda_bar"] - baselines.mu_lambda) / baselines.sd_lambda
        rows.append({
            "fighter_norm": fighter,
            "score": 100.0 / (1.0 + math.exp(-z)),
            "lambda_bar": rec["lambda_bar"],
            "n_fights": rec["n_fights"],
            "n_rounds": rec["n_rounds"],
        })
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)