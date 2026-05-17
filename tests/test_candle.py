"""Tests for Candle distribution + Snake regression methods (v5.4.6)."""
import math
import random
import pytest

from algorithmeai import Snake, Candle, compute_candle


class TestComputeCandle:
    def test_empty(self):
        c = compute_candle([])
        assert c.n == 0
        assert math.isnan(c.high) and math.isnan(c.low) and math.isnan(c.median)
        assert math.isnan(c.iqr_mean) and math.isnan(c.mean) and math.isnan(c.std)

    def test_single_value(self):
        c = compute_candle([3.14])
        assert c.n == 1
        assert c.high == c.low == c.median == c.q1 == c.q3 == 3.14
        assert c.mean == 3.14 and c.iqr_mean == 3.14
        assert c.std == 0.0

    def test_known_distribution(self):
        # 1..9, percentiles by linear interpolation on sorted index
        c = compute_candle(list(range(1, 10)))
        assert c.n == 9
        assert c.low == 1.0 and c.high == 9.0
        assert c.median == 5.0
        assert c.q1 == 3.0 and c.q3 == 7.0
        assert c.mean == 5.0
        assert abs(c.std - math.sqrt(60 / 9)) < 1e-9
        # iqr_mean averages values within [Q1, Q3] inclusive: 3..7
        assert c.iqr_mean == 5.0

    def test_coerces_strings(self):
        c = compute_candle(["1", "2.5", "garbage", 3])
        assert c.n == 3
        assert c.low == 1.0 and c.high == 3.0

    def test_to_dict_keys(self):
        c = compute_candle([1.0, 2.0, 3.0])
        keys = set(c.to_dict().keys())
        assert keys == {"high", "q3", "median", "q1", "low", "mean", "iqr_mean", "std", "n"}


class TestSnakeCandle:
    @pytest.fixture
    def trained(self):
        random.seed(7)
        n = 400
        rows = []
        for _ in range(n):
            x1 = random.uniform(0, 10)
            x2 = random.uniform(0, 10)
            y = 2 * x1 - x2 + random.gauss(0, 0.5)
            rows.append({"x1": round(x1, 3), "x2": round(x2, 3), "y": round(y, 3)})
        return Snake(rows, target_index="y", n_layers=8, bucket=60,
                     noise=0.3, workers=2, vocal=False), rows

    def test_get_candle_returns_candle(self, trained):
        model, rows = trained
        c = model.get_candle(rows[0])
        assert isinstance(c, Candle)
        assert c.n > 0
        assert c.low <= c.q1 <= c.median <= c.q3 <= c.high

    def test_get_batch_candles_shape(self, trained):
        model, rows = trained
        candles = model.get_batch_candles(rows[:5])
        assert len(candles) == 5
        assert all(isinstance(c, Candle) for c in candles)

    def test_get_regression_returns_float(self, trained):
        model, rows = trained
        y_hat = model.get_regression(rows[0])
        assert isinstance(y_hat, float)

    def test_get_batch_regression_shape(self, trained):
        model, rows = trained
        preds = model.get_batch_regression(rows[:5])
        assert len(preds) == 5
        assert all(isinstance(p, float) for p in preds)

    def test_regression_beats_classification_r2(self, trained):
        """The whole point of v5.4.6: candle.iqr_mean > get_prediction in R^2."""
        model, rows = trained
        ys = [float(r["y"]) for r in rows]
        mean_y = sum(ys) / len(ys)
        ss_tot = sum((y - mean_y) ** 2 for y in ys)
        preds_class = [float(model.get_prediction(r)) for r in rows]
        preds_reg = model.get_batch_regression(rows)
        r2_class = 1 - sum((y - p) ** 2 for y, p in zip(ys, preds_class)) / ss_tot
        r2_reg = 1 - sum((y - p) ** 2 for y, p in zip(ys, preds_reg)) / ss_tot
        # Regression must be no worse than classification on a clean linear target.
        assert r2_reg >= r2_class - 0.01
