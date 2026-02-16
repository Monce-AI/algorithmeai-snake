"""Tests for logging behavior: log accumulation, JSON persistence, banner."""
import os
import json
import tempfile
import pytest

from algorithmeai import Snake

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_CSV = os.path.join(FIXTURES, "sample.csv")


class TestLogging:
    def test_log_always_accumulates(self):
        """Even with vocal=False, log should have content."""
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=1, bucket=5, vocal=False)
        assert len(s.log) > 0
        assert "TRAINING" in s.log

    def test_log_preserved_in_json(self):
        """Saved model JSON should contain the log key."""
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=1, bucket=5, vocal=False)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            s.to_json(path)
            with open(path) as f:
                data = json.load(f)
            assert "log" in data
            assert len(data["log"]) > 0
        finally:
            os.unlink(path)

    def test_log_restored_from_json(self):
        """Loading a model should restore its log (including banner)."""
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=1, bucket=5, vocal=False)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            s.to_json(path)
            s2 = Snake(path)
            assert "v4.3.2" in s2.log
        finally:
            os.unlink(path)

    def test_banner_in_log_always(self):
        """Banner with version and author should always be in log."""
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=1, bucket=5, vocal=False)
        assert "v4.3.2" in s.log
        assert "Charles Dana" in s.log
