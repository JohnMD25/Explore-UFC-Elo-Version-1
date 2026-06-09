"""Pure-function tests for parsing helpers.

Dual-import pattern: tries the post-refactor module first, falls back to
the current monolithic app.py. Run with `pytest tests/test_parsing.py`.
"""

from __future__ import annotations

import math

import pytest

try:
    from data.parsing import (
        _clean_weight_class,
        _format_wc_with_title,
        _method_tag,
        _normalise_name,
        _parse_ctrl_seconds,
        _parse_height,
        _parse_inches,
        _parse_lbs,
        _parse_round_time_to_minutes,
        _parse_x_of_y,
    )
except ImportError:
    from app import (
        _clean_weight_class,
        _format_wc_with_title,
        _method_tag,
        _normalise_name,
        _parse_ctrl_seconds,
        _parse_height,
        _parse_inches,
        _parse_lbs,
        _parse_round_time_to_minutes,
        _parse_x_of_y,
    )


# ---------------------------------------------------------------------------
# _clean_weight_class
# ---------------------------------------------------------------------------

class TestCleanWeightClass:
    def test_basic_division(self):
        assert _clean_weight_class("Welterweight Bout") == ("Welterweight", "None", False)

    def test_ufc_prefix_stripped(self):
        assert _clean_weight_class("UFC Welterweight Title Bout") == ("Welterweight", "Title", False)

    def test_interim_title(self):
        assert _clean_weight_class("UFC Interim Lightweight Title Bout") == ("Lightweight", "Interim", False)

    def test_womens_division(self):
        assert _clean_weight_class("Women's Bantamweight Bout") == ("Womens Bantamweight", "None", True)

    def test_womens_title(self):
        assert _clean_weight_class("UFC Women's Strawweight Title Bout") == ("Womens Strawweight", "Title", True)

    def test_catch_weight(self):
        assert _clean_weight_class("Catch Weight Bout") == ("Catch Weight", "None", False)

    def test_open_weight(self):
        assert _clean_weight_class("Open Weight Bout") == ("Open Weight", "None", False)

    def test_heavyweight(self):
        assert _clean_weight_class("Heavyweight Bout") == ("Heavyweight", "None", False)

    def test_light_heavyweight(self):
        assert _clean_weight_class("Light Heavyweight Bout") == ("Light Heavyweight", "None", False)

    def test_tournament_keyword_stripped(self):
        clean, title_type, is_womens = _clean_weight_class("Heavyweight Tournament")
        assert clean == "Heavyweight"
        assert title_type == "None"
        assert is_womens is False

    def test_empty_string(self):
        assert _clean_weight_class("") == ("", "None", False)

    def test_none_value(self):
        assert _clean_weight_class(None) == ("", "None", False)

    def test_whitespace_only(self):
        assert _clean_weight_class("   ") == ("", "None", False)

    def test_unknown_class_returns_three_tuple(self):
        # Unknown classes are kept verbatim and warned to stdout. We just
        # assert the contract (3-tuple, correct flag types) holds.
        result = _clean_weight_class("Mythical Cruiserweight Bout")
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result[1] == "None"
        assert result[2] is False


# ---------------------------------------------------------------------------
# _format_wc_with_title
# ---------------------------------------------------------------------------

class TestFormatWcWithTitle:
    def test_no_title(self):
        assert _format_wc_with_title("Welterweight", "None") == "Welterweight"

    def test_title_appended(self):
        assert _format_wc_with_title("Welterweight", "Title") == "Welterweight (Title Fight)"

    def test_interim_appended(self):
        assert _format_wc_with_title("Lightweight", "Interim") == "Lightweight (Interim Title Fight)"

    def test_empty_class_with_title(self):
        assert _format_wc_with_title("", "Title") == "Title Fight"

    def test_empty_class_with_interim(self):
        assert _format_wc_with_title("", "Interim") == "Interim Title Fight"

    def test_none_inputs(self):
        assert _format_wc_with_title(None, None) == ""


# ---------------------------------------------------------------------------
# _parse_round_time_to_minutes
# ---------------------------------------------------------------------------

class TestParseRoundTime:
    def test_round_3_at_4_32(self):
        # Round 3 at 4:32 = 2 full rounds + 4.5333 = ~14.5333 minutes.
        result = _parse_round_time_to_minutes(3, "4:32")
        assert result is not None
        assert math.isclose(result, 14.5333, rel_tol=1e-3)

    def test_round_1_at_0_00(self):
        # Edge case for first-round-finish detection: zero-second "finish".
        assert _parse_round_time_to_minutes(1, "0:00") == 0.0

    def test_round_1_full_round(self):
        assert _parse_round_time_to_minutes(1, "5:00") == 5.0

    def test_round_5_full_fight(self):
        assert _parse_round_time_to_minutes(5, "5:00") == 25.0

    def test_string_round_number(self):
        result = _parse_round_time_to_minutes("2", "2:30")
        assert result is not None
        assert math.isclose(result, 7.5, rel_tol=1e-6)

    def test_invalid_round(self):
        assert _parse_round_time_to_minutes("not a round", "1:00") is None

    def test_invalid_time(self):
        assert _parse_round_time_to_minutes(2, "garbage") is None

    def test_zero_round_returns_none(self):
        assert _parse_round_time_to_minutes(0, "1:00") is None


# ---------------------------------------------------------------------------
# _parse_x_of_y (sig strikes, takedowns, etc.)
# ---------------------------------------------------------------------------

class TestParseXOfY:
    def test_basic(self):
        assert _parse_x_of_y("12 of 30") == (12, 30)

    def test_zero_zero(self):
        assert _parse_x_of_y("0 of 0") == (0, 0)

    def test_dashes(self):
        assert _parse_x_of_y("--") == (0, 0)

    def test_empty_string(self):
        assert _parse_x_of_y("") == (0, 0)

    def test_none(self):
        assert _parse_x_of_y(None) == (0, 0)

    def test_extra_whitespace(self):
        assert _parse_x_of_y("  5 of 8  ") == (5, 8)


# ---------------------------------------------------------------------------
# _parse_ctrl_seconds (M:SS → seconds)
# ---------------------------------------------------------------------------

class TestParseCtrlSeconds:
    def test_basic(self):
        assert _parse_ctrl_seconds("2:30") == 150

    def test_zero(self):
        assert _parse_ctrl_seconds("0:00") == 0

    def test_double_digit_seconds(self):
        assert _parse_ctrl_seconds("4:45") == 285

    def test_dashes(self):
        assert _parse_ctrl_seconds("--") == 0

    def test_empty_string(self):
        assert _parse_ctrl_seconds("") == 0

    def test_none(self):
        assert _parse_ctrl_seconds(None) == 0


# ---------------------------------------------------------------------------
# _parse_height / _parse_lbs / _parse_inches
# ---------------------------------------------------------------------------

class TestParseHeight:
    def test_5_11(self):
        assert _parse_height("5' 11\"") == 71.0

    def test_6_0(self):
        assert _parse_height("6' 0\"") == 72.0

    def test_5_4(self):
        assert _parse_height("5' 4\"") == 64.0

    def test_no_space(self):
        # The regex tolerates 0+ whitespace between feet and inches.
        assert _parse_height("5'11\"") == 71.0

    def test_dashes(self):
        assert _parse_height("--") is None

    def test_empty_string(self):
        assert _parse_height("") is None

    def test_none(self):
        assert _parse_height(None) is None


class TestParseLbs:
    def test_with_unit(self):
        assert _parse_lbs("170 lbs.") == 170.0

    def test_no_unit(self):
        assert _parse_lbs("155") == 155.0

    def test_dashes(self):
        assert _parse_lbs("--") is None

    def test_none(self):
        assert _parse_lbs(None) is None


class TestParseInches:
    def test_with_unit(self):
        assert _parse_inches("76\"") == 76.0

    def test_no_unit(self):
        assert _parse_inches("72") == 72.0

    def test_dashes(self):
        assert _parse_inches("--") is None

    def test_none(self):
        assert _parse_inches(None) is None


# ---------------------------------------------------------------------------
# _normalise_name
# ---------------------------------------------------------------------------

class TestNormaliseName:
    def test_collapses_whitespace(self):
        assert _normalise_name("  Conor   McGregor  ") == _normalise_name("conor mcgregor")

    def test_case_insensitive(self):
        assert _normalise_name("JON JONES") == _normalise_name("jon jones")

    def test_unaccented_basic_match(self):
        assert _normalise_name("Jose Aldo ") == _normalise_name("jose aldo")

    def test_accents_currently_preserved(self):
        # NFKC normalisation does NOT strip accents — only compatibility forms.
        # If accent-insensitive matching is wanted, switch to NFD + filter
        # combining chars in a behaviour-changing pass. This test documents
        # the current contract.
        assert _normalise_name("José Aldo") != _normalise_name("Jose Aldo")

    def test_empty(self):
        assert _normalise_name("") == ""

    def test_none(self):
        assert _normalise_name(None) == ""


# ---------------------------------------------------------------------------
# _method_tag
# ---------------------------------------------------------------------------

class TestMethodTag:
    def test_ko_tko(self):
        assert _method_tag("KO/TKO") == "KO/TKO"

    def test_ko_lowercase(self):
        assert _method_tag("ko") == "KO/TKO"

    def test_tko_alone(self):
        assert _method_tag("TKO") == "KO/TKO"

    def test_submission(self):
        assert _method_tag("Submission") == "SUB"

    def test_decision_unanimous(self):
        assert _method_tag("Decision - Unanimous") == "DEC"

    def test_decision_word_only(self):
        assert _method_tag("decision") == "DEC"

    def test_unknown_method_falls_through(self):
        assert _method_tag("Doctor Stoppage") == "Other"

    def test_none_value(self):
        assert _method_tag(None) == "Other"

    def test_empty_string(self):
        assert _method_tag("") == "Other"