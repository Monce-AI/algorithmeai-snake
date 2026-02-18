"""Tests for CLI: snake train / predict / info via subprocess."""
import os
import sys
import json
import subprocess
import tempfile
import pytest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_CSV = os.path.join(FIXTURES, "sample.csv")
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def _run(*args, check=True):
    """Run `python -m algorithmeai <args>` and return CompletedProcess."""
    cmd = [sys.executable, "-m", "algorithmeai"] + list(args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        check=check,
    )


class TestCLITrain:
    def test_train_produces_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        try:
            result = _run("train", SAMPLE_CSV, "--target", "3", "--layers", "2",
                          "--bucket", "5", "-o", out_path)
            assert result.returncode == 0
            with open(out_path) as f:
                data = json.load(f)
            assert "version" in data
            assert "population" in data
            assert "layers" in data
        finally:
            os.unlink(out_path)
            # Also clean up the default snakeclassifier.json if created
            default_path = os.path.join(PROJECT_ROOT, "snakeclassifier.json")
            if os.path.exists(default_path):
                os.unlink(default_path)

    def test_train_vocal_flag(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        try:
            result = _run("train", SAMPLE_CSV, "--target", "3", "--layers", "1",
                          "--bucket", "5", "-o", out_path, "--vocal")
            assert result.returncode == 0
            # vocal should produce training output on stdout
            assert "TRAINING" in result.stdout or "Snake" in result.stdout or "layer" in result.stdout.lower()
        finally:
            os.unlink(out_path)
            default_path = os.path.join(PROJECT_ROOT, "snakeclassifier.json")
            if os.path.exists(default_path):
                os.unlink(default_path)


class TestCLIPredict:
    @pytest.fixture(autouse=True)
    def _model(self, tmp_path):
        self.model_path = str(tmp_path / "model.json")
        _run("train", SAMPLE_CSV, "--target", "3", "--layers", "2",
             "--bucket", "5", "-o", self.model_path)
        # Clean up default
        default_path = os.path.join(PROJECT_ROOT, "snakeclassifier.json")
        if os.path.exists(default_path):
            os.unlink(default_path)

    def test_predict_basic(self):
        result = _run("predict", self.model_path,
                       "-q", '{"color": "red", "size": 10.0, "shape": "circle"}')
        assert result.returncode == 0
        assert "Prediction:" in result.stdout
        assert "Confidence:" in result.stdout

    def test_predict_with_audit(self):
        result = _run("predict", self.model_path,
                       "-q", '{"color": "red", "size": 10.0, "shape": "circle"}',
                       "--audit")
        assert result.returncode == 0
        assert "LAYER" in result.stdout
        assert "PREDICTION" in result.stdout


class TestCLIInfo:
    @pytest.fixture(autouse=True)
    def _model(self, tmp_path):
        self.model_path = str(tmp_path / "model.json")
        _run("train", SAMPLE_CSV, "--target", "3", "--layers", "2",
             "--bucket", "5", "-o", self.model_path)
        default_path = os.path.join(PROJECT_ROOT, "snakeclassifier.json")
        if os.path.exists(default_path):
            os.unlink(default_path)

    def test_info_output(self):
        result = _run("info", self.model_path)
        assert result.returncode == 0
        assert "v4.4.3" in result.stdout
        assert "Target:" in result.stdout
        assert "Population:" in result.stdout
        assert "Layers:" in result.stdout


class TestCLINoCommand:
    def test_no_command_shows_help(self):
        result = _run(check=False)
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert "snake" in output.lower() or "usage" in output.lower() or "help" in output.lower()
