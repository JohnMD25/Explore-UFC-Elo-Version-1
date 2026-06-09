"""UFC Elo engine.

Processes fights in chronological order and tracks each fighter's peak rating
(plus the date and weight class at which the peak was reached).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# 🔧 ELO CONFIGURATION — EDIT THESE VALUES TO TUNE THE ENGINE
# ---------------------------------------------------------------------------
# Starting rating assigned to every fighter before their first UFC bout.
BASE_RATING: float = 1500.0

# Base K-factor: controls how much a single result moves the rating.
# Higher K = bigger swings per fight; lower K = more stable ratings.
K_FACTOR: float = 32.0

# Method-of-finish multipliers applied to K for wins/losses.
# (Draws always use the base K, regardless of method.)
FINISH_MULTIPLIER: float = 1.20    # KO / TKO / Submission
DECISION_MULTIPLIER: float = 1.00  # Decision

# Weight classes to exclude entirely. Matched as a lower-case substring,
# so "catch weight" will also catch "Catch Weight Bout", etc.
EXCLUDED_WEIGHT_CLASSES: tuple[str, ...] = (
    "catch weight",
    "catch weight bout",
)
# ---------------------------------------------------------------------------


@dataclass
class Peak:
    rating: float
    date: pd.Timestamp
    weight_class: str


class UFCEloEngine:
    """Classic Elo with a small method-of-finish multiplier."""

    def __init__(
        self,
        k: float = K_FACTOR,
        base: float = BASE_RATING,
        finish_multiplier: float = FINISH_MULTIPLIER,
        decision_multiplier: float = DECISION_MULTIPLIER,
        excluded_weight_classes: Optional[Iterable[str]] = None,
    ) -> None:
        # Defaults are pulled from the ELO CONFIGURATION block at the top of
        # this file — edit those constants to change engine behaviour.
        self.k = k
        self.base = base
        self.finish_multiplier = finish_multiplier
        self.decision_multiplier = decision_multiplier
        self.excluded_weight_classes = {
            s.lower()
            for s in (excluded_weight_classes or EXCLUDED_WEIGHT_CLASSES)
        }
        self.ratings: Dict[str, float] = defaultdict(lambda: self.base)
        self.peaks: Dict[str, Peak] = {}
        # Per-fighter post-fight Elo history (one entry per processed fight).
        self.history: Dict[str, list] = defaultdict(list)
        # Per-fight Elo swing records, used by `top_fight_deltas`.
        self.fight_deltas: list = []

    # --- core math ---------------------------------------------------------
    def expected_score(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def method_multiplier(self, method: object) -> float:
        if not isinstance(method, str):
            return self.decision_multiplier
        m = method.lower()
        if "ko" in m or "tko" in m:
            return self.finish_multiplier
        if "sub" in m:
            return self.finish_multiplier
        return self.decision_multiplier

    # --- bookkeeping -------------------------------------------------------
    def _update_peak(self, fighter: str, rating: float, date: pd.Timestamp, weight_class: str) -> None:
        prev = self.peaks.get(fighter)
        if prev is None or rating > prev.rating:
            self.peaks[fighter] = Peak(rating=rating, date=date, weight_class=weight_class)

    def _is_excluded_class(self, weight_class: object) -> bool:
        if not isinstance(weight_class, str):
            return True
        wc = weight_class.lower()
        return any(ex in wc for ex in self.excluded_weight_classes)

    # --- public API --------------------------------------------------------
    def process_fight(
        self,
        fighter_a: str,
        fighter_b: str,
        outcome: str,
        method: str,
        weight_class: str,
        date: pd.Timestamp,
        event: str = "",
        bout: str = "",
        title_type: str = "None",
    ) -> None:
        """Outcome is from fighter_a's perspective: 'W', 'L', 'D', or 'NC'.

        title_type tags the bout's championship status: 'None', 'Title',
        or 'Interim'. It is stored on each fight_deltas record so display
        layers can append a 'Title Fight' suffix to the weight class.
        """
        if self._is_excluded_class(weight_class):
            return
        if outcome == "NC":
            return
        if outcome not in {"W", "L", "D"}:
            return

        ra = self.ratings[fighter_a]
        rb = self.ratings[fighter_b]
        ea = self.expected_score(ra, rb)
        eb = 1.0 - ea

        if outcome == "W":
            sa, sb = 1.0, 0.0
        elif outcome == "L":
            sa, sb = 0.0, 1.0
        else:  # draw
            sa, sb = 0.5, 0.5

        # Draws use the base K. Wins/losses can be amplified by finish.
        if outcome == "D":
            k_eff = self.k
        else:
            k_eff = self.k * self.method_multiplier(method)

        new_ra = ra + k_eff * (sa - ea)
        new_rb = rb + k_eff * (sb - eb)

        self.ratings[fighter_a] = new_ra
        self.ratings[fighter_b] = new_rb
        self.history[fighter_a].append(new_ra)
        self.history[fighter_b].append(new_rb)
        self._update_peak(fighter_a, new_ra, date, weight_class)
        self._update_peak(fighter_b, new_rb, date, weight_class)

        # Record the per-fight Elo swing for later "biggest fight" /
        # per-fighter history lookups.
        self.fight_deltas.append(
            {
                "date": date,
                "event": event,
                "bout": bout,
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "delta_a": new_ra - ra,
                "delta_b": new_rb - rb,
                "post_a": new_ra,
                "post_b": new_rb,
                "outcome": outcome,
                "method": method,
                "weight_class": weight_class,
                "title_type": title_type,
            }
        )

    def run(self, fights: pd.DataFrame) -> None:
        """Expects columns: DATE, FIGHTER_A, FIGHTER_B, OUTCOME, METHOD, WEIGHTCLASS.

        Optional columns: EVENT, BOUT, TITLE_TYPE.
        DataFrame must already be sorted by DATE ascending.
        """
        has_event = "EVENT" in fights.columns
        has_bout = "BOUT" in fights.columns
        has_title = "TITLE_TYPE" in fights.columns
        for row in fights.itertuples(index=False):
            self.process_fight(
                fighter_a=row.FIGHTER_A,
                fighter_b=row.FIGHTER_B,
                outcome=row.OUTCOME,
                method=row.METHOD,
                weight_class=row.WEIGHTCLASS,
                date=row.DATE,
                event=getattr(row, "EVENT", "") if has_event else "",
                bout=getattr(row, "BOUT", "") if has_bout else "",
                title_type=getattr(row, "TITLE_TYPE", "None") if has_title else "None",
            )

    def peaks_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "Fighter": fighter,
                "Peak Elo": peak.rating,
                "Peak Date": peak.date,
                "Weight Class": peak.weight_class,
            }
            for fighter, peak in self.peaks.items()
        ]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("Peak Elo", ascending=False).reset_index(drop=True)

    def history_dataframe(self, fighters: Iterable[str]) -> pd.DataFrame:
        """Return long-format history (Fighter, Fight #, Elo) for the given fighters."""
        rows = []
        for fighter in fighters:
            for i, rating in enumerate(self.history.get(fighter, []), start=1):
                rows.append({"Fighter": fighter, "Fight #": i, "Elo": rating})
        return pd.DataFrame(rows)

    def top_fight_deltas(self, n: int = 10) -> pd.DataFrame:
        """Return the n fights with the biggest absolute Elo swing.

        Columns: Date, Matchup, Elo delta.
        - Matchup is formatted 'Winner def. Loser' (or 'A drew B' for draws).
        - 'Elo delta' shows '+winner / +loser' (loser is negative for
          decisive outcomes).
        """
        if not self.fight_deltas:
            return pd.DataFrame(columns=["Date", "Matchup", "Elo delta"])

        rows = []
        for f in self.fight_deltas:
            outcome = f["outcome"]
            if outcome == "W":
                matchup = f"{f['fighter_a']} def. {f['fighter_b']}"
                winner_delta, loser_delta = f["delta_a"], f["delta_b"]
            elif outcome == "L":
                matchup = f"{f['fighter_b']} def. {f['fighter_a']}"
                winner_delta, loser_delta = f["delta_b"], f["delta_a"]
            else:  # draw — order by which fighter gained
                if f["delta_a"] >= f["delta_b"]:
                    matchup = f"{f['fighter_a']} drew {f['fighter_b']}"
                    winner_delta, loser_delta = f["delta_a"], f["delta_b"]
                else:
                    matchup = f"{f['fighter_b']} drew {f['fighter_a']}"
                    winner_delta, loser_delta = f["delta_b"], f["delta_a"]

            date = f["date"]
            date_str = date.strftime("%Y-%m-%d") if pd.notna(date) else ""

            rows.append(
                {
                    "Date": date_str,
                    "Matchup": matchup,
                    "Elo delta": f"{winner_delta:+.1f} / {loser_delta:+.1f}",
                    "_magnitude": abs(f["delta_a"]),
                }
            )

        df = (
            pd.DataFrame(rows)
            .sort_values("_magnitude", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )
        return df.drop(columns=["_magnitude"])

    @staticmethod
    def _format_result(outcome: str, method: object) -> str:
        """Combine outcome (W/L/D) with a short method tag (KO/SUB/DEC/DQ)."""
        label = {"W": "Win", "L": "Loss", "D": "Draw"}.get(outcome, outcome)
        tag = ""
        if isinstance(method, str) and method.strip():
            m = method.lower()
            if "ko" in m or "tko" in m:
                tag = "KO/TKO"
            elif "sub" in m:
                tag = "SUB"
            elif "dec" in m or "decision" in m:
                tag = "DEC"
            elif "dq" in m or "disqualif" in m:
                tag = "DQ"
            else:
                tag = method.strip()
        return f"{label} ({tag})" if tag else label

    def fighter_fights(self, fighter: str) -> pd.DataFrame:
        """Return all fights for a given fighter, most recent first.

        Columns: Date, Opponent, Result, Weight Class, ΔElo.
        Outcome is from the perspective of the named fighter.
        """
        invert = {"W": "L", "L": "W", "D": "D"}
        rows = []
        for f in self.fight_deltas:
            if f["fighter_a"] == fighter:
                opponent = f["fighter_b"]
                outcome = f["outcome"]
                delta = f["delta_a"]
            elif f["fighter_b"] == fighter:
                opponent = f["fighter_a"]
                outcome = invert.get(f["outcome"], f["outcome"])
                delta = f["delta_b"]
            else:
                continue

            wc = f["weight_class"]
            tt = f.get("title_type", "None")
            if tt == "Title":
                wc_display = f"{wc} (Title Fight)"
            elif tt == "Interim":
                wc_display = f"{wc} (Interim Title Fight)"
            else:
                wc_display = wc
            rows.append(
                {
                    "Date": f["date"],
                    "Opponent": opponent,
                    "Result": self._format_result(outcome, f["method"]),
                    "Weight Class": wc_display,
                    "ΔElo": round(delta, 1),
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("Date", ascending=False).reset_index(drop=True)

    def fight_counts(self) -> Dict[str, int]:
        """Return a mapping of fighter name → number of UFC fights processed."""
        counts: Dict[str, int] = defaultdict(int)
        for f in self.fight_deltas:
            counts[f["fighter_a"]] += 1
            counts[f["fighter_b"]] += 1
        return dict(counts)

    def opponents_of(self, fighter: str) -> list:
        """Return all distinct opponents this fighter has faced, alphabetical."""
        opponents: set = set()
        for f in self.fight_deltas:
            if f["fighter_a"] == fighter:
                opponents.add(f["fighter_b"])
            elif f["fighter_b"] == fighter:
                opponents.add(f["fighter_a"])
        return sorted(opponents, key=str.casefold)

    def head_to_head(self, fighter_a: str, fighter_b: str) -> list:
        """Return every bout between these two fighters, oldest first.

        Each entry is the original fight_deltas record plus a normalised
        view from fighter_a's perspective: a_outcome / a_delta / b_delta.
        """
        invert = {"W": "L", "L": "W", "D": "D"}
        bouts = []
        for f in self.fight_deltas:
            if {f["fighter_a"], f["fighter_b"]} != {fighter_a, fighter_b}:
                continue
            if f["fighter_a"] == fighter_a:
                a_outcome = f["outcome"]
                a_delta = f["delta_a"]
                b_delta = f["delta_b"]
            else:
                a_outcome = invert.get(f["outcome"], f["outcome"])
                a_delta = f["delta_b"]
                b_delta = f["delta_a"]
            bouts.append(
                {**f, "a_outcome": a_outcome, "a_delta": a_delta, "b_delta": b_delta}
            )
        bouts.sort(key=lambda x: x["date"])
        return bouts