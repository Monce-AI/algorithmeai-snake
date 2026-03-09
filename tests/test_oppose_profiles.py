"""Tests for Snake v5.2.0 — oppose profiles and new literal types."""
import json
import os
import tempfile

import pytest
from algorithmeai import Snake
from algorithmeai.snake import (
    _levenshtein, _jaccard_bigrams, _common_prefix_len, _common_suffix_len,
    _entropy, _hex_ratio, _repeat_period_score, _count_upper, _count_digits,
    _count_special, _VALID_PROFILES,
)

SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "fixtures", "sample.csv")

# ---- Helper functions ----

TEXT_DATA = [
    {"label": "A", "name": "Hello World", "code": "ABC123"},
    {"label": "A", "name": "Hi There", "code": "DEF456"},
    {"label": "B", "name": "Goodbye Moon", "code": "GHI789"},
    {"label": "B", "name": "See Ya", "code": "JKL012"},
    {"label": "C", "name": "Farewell Sun", "code": "MNO345"},
    {"label": "C", "name": "Later Star", "code": "PQR678"},
]

NUMERIC_DATA = [
    {"label": "high", "temp": 98.6, "pressure": 1013.25, "humidity": 45.0},
    {"label": "high", "temp": 99.1, "pressure": 1010.00, "humidity": 50.0},
    {"label": "low", "temp": 36.5, "pressure": 980.50, "humidity": 80.0},
    {"label": "low", "temp": 35.0, "pressure": 975.00, "humidity": 85.0},
    {"label": "mid", "temp": 72.0, "pressure": 1000.00, "humidity": 60.0},
    {"label": "mid", "temp": 68.5, "pressure": 995.00, "humidity": 65.0},
]

CRYPTO_DATA = [
    {"label": "hash", "val": "a3f8c2e1b907d456"},
    {"label": "hash", "val": "ff12ab9900cd3344"},
    {"label": "id", "val": "MXKL-4492-B-ZZQP"},
    {"label": "id", "val": "NRST-7781-C-WWXR"},
    {"label": "base64", "val": "SGVsbG8gV29ybGQ="},
    {"label": "base64", "val": "R29vZGJ5ZSBNb29u"},
]

LONG_TEXT_DATA = [
    {"label": "formal", "text": "Dear Sir, I am writing to express my deepest concerns regarding the recent policy changes that were announced last week in the board meeting."},
    {"label": "formal", "text": "To Whom It May Concern, please accept this letter as my formal objection."},
    {"label": "casual", "text": "hey man whats up just wanted to check in"},
    {"label": "casual", "text": "yo dude heard about the party tonight? lemme know if you need a ride or something lol, also bring chips and drinks for everyone please"},
    {"label": "technical", "text": "The TCP/IP handshake protocol implements a three-way mechanism for establishing reliable connections between distributed systems across the network."},
    {"label": "technical", "text": "O(log n) complexity."},
]

CATEGORICAL_DATA = [
    {"label": "action", "genres": "Action,Adventure,Sci-Fi"},
    {"label": "action", "genres": "Action,Thriller,Crime"},
    {"label": "drama", "genres": "Drama,Romance"},
    {"label": "drama", "genres": "Drama,Biography,History"},
    {"label": "comedy", "genres": "Comedy,Family,Animation"},
    {"label": "comedy", "genres": "Comedy,Romance"},
]


# ===========================================================================
# Module-level helper tests
# ===========================================================================

class TestHelperFunctions:
    def test_levenshtein_identical(self):
        assert _levenshtein("abc", "abc") == 0

    def test_levenshtein_empty(self):
        assert _levenshtein("", "abc") == 3
        assert _levenshtein("abc", "") == 3

    def test_levenshtein_different(self):
        assert _levenshtein("kitten", "sitting") == 3

    def test_jaccard_identical(self):
        assert _jaccard_bigrams("hello", "hello") == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard_bigrams("ab", "cd") == 0.0

    def test_jaccard_partial(self):
        j = _jaccard_bigrams("abc", "abd")
        assert 0 < j < 1

    def test_common_prefix(self):
        assert _common_prefix_len("abcdef", "abcxyz") == 3
        assert _common_prefix_len("xyz", "abc") == 0

    def test_common_suffix(self):
        assert _common_suffix_len("testing", "running") == 3
        assert _common_suffix_len("abc", "xyz") == 0

    def test_entropy_uniform(self):
        e = _entropy("abcd")
        assert e == pytest.approx(2.0, abs=0.01)

    def test_entropy_single_char(self):
        assert _entropy("aaaa") == 0.0

    def test_entropy_empty(self):
        assert _entropy("") == 0.0

    def test_hex_ratio(self):
        assert _hex_ratio("0123456789abcdef") == 1.0
        assert _hex_ratio("") == 0.0
        assert _hex_ratio("zzzz") == 0.0

    def test_repeat_period_score(self):
        assert _repeat_period_score("abababab") > 0.8
        assert _repeat_period_score("ab") == 0.0  # too short

    def test_count_upper(self):
        assert _count_upper("Hello World") == 2
        assert _count_upper("hello") == 0

    def test_count_digits(self):
        assert _count_digits("abc123") == 3
        assert _count_digits("abc") == 0

    def test_count_special(self):
        assert _count_special("a!b@c#") == 3
        assert _count_special("abc") == 0


# ===========================================================================
# Profile auto-detection
# ===========================================================================

class TestProfileDetection:
    def test_auto_numeric_only(self):
        s = Snake(NUMERIC_DATA, target_index="label", n_layers=1, bucket=5, oppose_profile="auto")
        assert s.oppose_profile == "scientific"

    def test_auto_text_only_long(self):
        s = Snake(LONG_TEXT_DATA, target_index="label", n_layers=1, bucket=5, oppose_profile="auto")
        assert s.oppose_profile == "linguistic"

    def test_explicit_override(self):
        s = Snake(TEXT_DATA, target_index="label", n_layers=1, bucket=5, oppose_profile="industrial")
        assert s.oppose_profile == "industrial"

    def test_invalid_profile_falls_to_auto(self):
        s = Snake(TEXT_DATA, target_index="label", n_layers=1, bucket=5, oppose_profile="nonexistent")
        assert s.oppose_profile in _VALID_PROFILES
        assert s.oppose_profile != "auto"  # auto resolves to a concrete profile

    def test_default_is_auto(self):
        s = Snake(TEXT_DATA, target_index="label", n_layers=1, bucket=5)
        assert s.oppose_profile in _VALID_PROFILES
        assert s.oppose_profile != "auto"


# ===========================================================================
# Each profile trains and predicts without error
# ===========================================================================

class TestProfileTraining:
    @pytest.mark.parametrize("profile", ["balanced", "linguistic", "industrial",
                                          "cryptographic", "scientific", "categorical"])
    def test_profile_trains_text_data(self, profile):
        s = Snake(TEXT_DATA, target_index="label", n_layers=2, bucket=5, oppose_profile=profile)
        pred = s.get_prediction({"name": "Hello World", "code": "ABC123"})
        assert pred in ["A", "B", "C"]

    @pytest.mark.parametrize("profile", ["balanced", "scientific"])
    def test_profile_trains_numeric_data(self, profile):
        s = Snake(NUMERIC_DATA, target_index="label", n_layers=2, bucket=5, oppose_profile=profile)
        pred = s.get_prediction({"temp": 98.6, "pressure": 1013.25, "humidity": 45.0})
        assert pred in ["high", "low", "mid"]

    def test_crypto_profile_on_crypto_data(self):
        s = Snake(CRYPTO_DATA, target_index="label", n_layers=2, bucket=5, oppose_profile="cryptographic")
        pred = s.get_prediction({"val": "a3f8c2e1b907d456"})
        assert pred in ["hash", "id", "base64"]

    def test_categorical_profile(self):
        s = Snake(CATEGORICAL_DATA, target_index="label", n_layers=2, bucket=5, oppose_profile="categorical")
        pred = s.get_prediction({"genres": "Action,Comedy"})
        assert pred in ["action", "drama", "comedy"]


# ===========================================================================
# New literal types in apply_literal
# ===========================================================================

class TestNewLiteralTypes:
    def setup_method(self):
        self.s = Snake(TEXT_DATA, target_index="label", n_layers=1, bucket=5)

    def test_tuc_literal(self):
        lit = [1, 3.0, True, "TUC"]  # uppercase count >= 3
        assert self.s.apply_literal({"name": "HELLO"}, lit) is True
        assert self.s.apply_literal({"name": "hello"}, lit) is False

    def test_tdc_literal(self):
        lit = [1, 2.0, True, "TDC"]  # digit count >= 2
        assert self.s.apply_literal({"name": "ab12"}, lit) is True
        assert self.s.apply_literal({"name": "abc"}, lit) is False

    def test_tsc_literal(self):
        lit = [1, 1.0, True, "TSC"]  # special count >= 1
        assert self.s.apply_literal({"name": "a!b"}, lit) is True
        assert self.s.apply_literal({"name": "abc"}, lit) is False

    def test_lev_literal(self):
        lit = [1, ["hello", 2.0], False, "LEV"]  # levenshtein > 2
        assert self.s.apply_literal({"name": "world"}, lit) is True
        assert self.s.apply_literal({"name": "hallo"}, lit) is False

    def test_jac_literal(self):
        lit = [1, ["hello", 0.5], True, "JAC"]  # jaccard >= 0.5
        assert self.s.apply_literal({"name": "hello"}, lit) is True
        assert self.s.apply_literal({"name": "xyz"}, lit) is False

    def test_pfx_literal(self):
        lit = [1, ["hello", 3.0], True, "PFX"]  # prefix >= 3
        assert self.s.apply_literal({"name": "helium"}, lit) is True
        assert self.s.apply_literal({"name": "world"}, lit) is False

    def test_sfx_literal(self):
        lit = [1, ["hello", 3.0], True, "SFX"]  # suffix >= 3
        assert self.s.apply_literal({"name": "jello"}, lit) is True
        assert self.s.apply_literal({"name": "world"}, lit) is False

    def test_ent_literal(self):
        lit = [1, 2.0, True, "ENT"]  # entropy >= 2.0
        assert self.s.apply_literal({"name": "abcdefgh"}, lit) is True
        assert self.s.apply_literal({"name": "aaaa"}, lit) is False

    def test_hex_literal(self):
        lit = [1, 0.8, True, "HEX"]  # hex ratio >= 0.8
        assert self.s.apply_literal({"name": "a1b2c3d4"}, lit) is True
        assert self.s.apply_literal({"name": "xyz!@#"}, lit) is False

    def test_rep_literal(self):
        lit = [1, 0.7, True, "REP"]  # repeat score >= 0.7
        assert self.s.apply_literal({"name": "abababab"}, lit) is True
        assert self.s.apply_literal({"name": "hello"}, lit) is False

    def test_cfc_literal(self):
        ref_freq = {"a": 0.5, "b": 0.5}
        lit = [1, [ref_freq, 1.0], False, "CFC"]  # chi_sq >= 1.0
        assert self.s.apply_literal({"name": "xxxxxx"}, lit) is True
        assert self.s.apply_literal({"name": "ababab"}, lit) is False

    def test_nd_literal(self):
        s = Snake(NUMERIC_DATA, target_index="label", n_layers=1, bucket=5)
        lit = [1, 3.0, True, "ND"]  # digit count >= 3
        assert s.apply_literal({"temp": 98.6}, lit) is True   # "98.6" has 3 digits
        assert s.apply_literal({"temp": 1.0}, lit) is False   # "1.0" has 2 digits

    def test_nz_literal(self):
        s = Snake(NUMERIC_DATA, target_index="label", n_layers=1, bucket=5)
        lit = [1, [50.0, 10.0, 1.0], True, "NZ"]  # z-score >= 1.0
        assert s.apply_literal({"temp": 70.0}, lit) is True
        assert s.apply_literal({"temp": 55.0}, lit) is False

    def test_nl_literal(self):
        s = Snake(NUMERIC_DATA, target_index="label", n_layers=1, bucket=5)
        lit = [1, 3.0, True, "NL"]  # log >= 3.0
        assert s.apply_literal({"temp": 100.0}, lit) is True
        assert s.apply_literal({"temp": 1.0}, lit) is False

    def test_nmg_literal(self):
        s = Snake(NUMERIC_DATA, target_index="label", n_layers=1, bucket=5)
        lit = [1, 1.5, True, "NMG"]  # magnitude >= 1.5
        assert s.apply_literal({"temp": 100.0}, lit) is True
        assert s.apply_literal({"temp": 5.0}, lit) is False

    def test_unknown_tag_returns_false(self):
        lit = [1, "x", False, "ZZZZZ"]
        assert self.s.apply_literal({"name": "hello"}, lit) is False


# ===========================================================================
# JSON round-trip with oppose_profile
# ===========================================================================

class TestJSONRoundTrip:
    def test_profile_persists_in_json(self):
        s = Snake(TEXT_DATA, target_index="label", n_layers=1, bucket=5, oppose_profile="linguistic")
        try:
            path = tempfile.mktemp(suffix=".json")
            s.to_json(path)
            with open(path) as f:
                raw = json.load(f)
            assert raw["config"]["oppose_profile"] == "linguistic"
            s2 = Snake(path)
            assert s2.oppose_profile == "linguistic"
        finally:
            os.unlink(path)

    def test_old_model_defaults_to_oppose(self):
        """Models without oppose_profile should use original oppose."""
        s = Snake(TEXT_DATA, target_index="label", n_layers=1, bucket=5, oppose_profile="balanced")
        try:
            path = tempfile.mktemp(suffix=".json")
            s.to_json(path)
            # Manually remove oppose_profile from config
            with open(path) as f:
                raw = json.load(f)
            del raw["config"]["oppose_profile"]
            with open(path, "w") as f:
                json.dump(raw, f)
            s2 = Snake(path)
            # Should fallback to original oppose
            assert s2._active_oppose == s2.oppose
        finally:
            os.unlink(path)

    def test_new_literal_types_survive_json_roundtrip(self):
        """Train with a profile that produces new literal types, save/load, predict."""
        s = Snake(CRYPTO_DATA, target_index="label", n_layers=3, bucket=5, oppose_profile="cryptographic")
        try:
            path = tempfile.mktemp(suffix=".json")
            s.to_json(path)
            s2 = Snake(path)
            pred = s2.get_prediction({"val": "a3f8c2e1b907d456"})
            assert pred in ["hash", "id", "base64"]
        finally:
            os.unlink(path)

    def test_scientific_profile_roundtrip(self):
        s = Snake(NUMERIC_DATA, target_index="label", n_layers=2, bucket=5, oppose_profile="scientific")
        try:
            path = tempfile.mktemp(suffix=".json")
            s.to_json(path)
            s2 = Snake(path)
            pred = s2.get_prediction({"temp": 98.6, "pressure": 1013.25, "humidity": 45.0})
            assert pred in ["high", "low", "mid"]
        finally:
            os.unlink(path)


# ===========================================================================
# Literal type distribution (statistical test over N oppose calls)
# ===========================================================================

class TestLiteralDistribution:
    def _collect_literal_tags(self, profile, data, n_calls=200):
        s = Snake(data, target_index="label", n_layers=1, bucket=5, oppose_profile=profile)
        tags = []
        for _ in range(n_calls):
            T = data[0]
            F = data[2]  # different class
            lit = s._active_oppose(T, F)
            if lit is not None:
                tags.append(lit[3])
        return tags

    def test_balanced_produces_variety(self):
        tags = self._collect_literal_tags("balanced", TEXT_DATA)
        unique_tags = set(tags)
        assert len(unique_tags) >= 3, f"balanced should produce variety, got {unique_tags}"

    def test_linguistic_heavy_on_distance(self):
        tags = self._collect_literal_tags("linguistic", TEXT_DATA)
        distance_tags = [t for t in tags if t in ("LEV", "JAC")]
        assert len(distance_tags) > 0, "linguistic should produce distance literals"

    def test_scientific_has_numeric_variety(self):
        tags = self._collect_literal_tags("scientific", NUMERIC_DATA)
        unique_tags = set(tags)
        assert "N" in unique_tags, "scientific should produce N literals"

    def test_cryptographic_produces_crypto_types(self):
        tags = self._collect_literal_tags("cryptographic", CRYPTO_DATA)
        crypto_tags = [t for t in tags if t in ("ENT", "HEX", "REP", "CFC", "TUC", "TDC", "TSC")]
        assert len(crypto_tags) > 0, f"cryptographic should produce crypto literals, got {set(tags)}"


# ===========================================================================
# Format literal text for new types
# ===========================================================================

class TestFormatLiteralText:
    def setup_method(self):
        self.s = Snake(TEXT_DATA, target_index="label", n_layers=1, bucket=5)

    def test_tuc_format(self):
        lit = (1, 3.0, True, "TUC")
        txt = self.s._format_literal_text(lit)
        assert "uppercase" in txt

    def test_lev_format(self):
        lit = (1, ["hello", 2.0], False, "LEV")
        txt = self.s._format_literal_text(lit)
        assert "levenshtein" in txt

    def test_ent_format(self):
        lit = (1, 2.5, True, "ENT")
        txt = self.s._format_literal_text(lit)
        assert "entropy" in txt

    def test_nz_format(self):
        lit = (1, [50.0, 10.0, 1.5], True, "NZ")
        txt = self.s._format_literal_text(lit)
        assert "zscore" in txt

    def test_nmg_format(self):
        lit = (1, 2.0, False, "NMG")
        txt = self.s._format_literal_text(lit)
        assert "magnitude" in txt


# ===========================================================================
# Backwards compatibility
# ===========================================================================

class TestBackwardsCompat:
    def test_sample_csv_default_profile(self):
        """Training from CSV with default params should work."""
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=1, bucket=5)
        assert s.oppose_profile in _VALID_PROFILES
        pred = s.get_prediction({"color": "red", "size": 10.0, "shape": "round"})
        assert pred is not None

    def test_oppose_still_works_directly(self):
        """Original oppose() is untouched and still callable."""
        s = Snake(TEXT_DATA, target_index="label", n_layers=1, bucket=5)
        T = TEXT_DATA[0]
        F = TEXT_DATA[2]
        lit = s.oppose(T, F)
        assert lit is not None
        assert len(lit) == 4
        assert lit[3] in ("T", "TN", "TLN", "TWS", "TPS", "TSS", "N")
