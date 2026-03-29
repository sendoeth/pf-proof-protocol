#!/usr/bin/env python3
"""
Tests for the Continuous Proof Maintenance Protocol.

Covers: resolution windows, rolling metrics, CUSUM drift edge cases,
atomic update integrity, staleness thresholds, empty inputs, price gaps,
schema validation, cross-schema compat, and cron idempotency.
"""

import json
import math
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maintain_proof import (
    wilson_ci,
    brier_score,
    brier_decomposition,
    calibration_slope,
    CUSUMDetector,
    PriceResolver,
    ProofMaintainer,
    atomic_write,
    validate_report,
    load_signals_from_array,
    load_signals_from_log,
    PROTOCOL_VERSION,
    DEFAULT_CUSUM_ALLOWANCE,
    DEFAULT_CUSUM_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(signal_id, symbol, direction, confidence, timestamp,
                 regime="SYSTEMIC", method="calibrated"):
    return {
        "signal_id": signal_id,
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence,
        "horizon_hours": 24,
        "timestamp": timestamp,
        "regime": regime,
        "method": method,
    }


def _make_signals(n, symbol="BTC", start="2026-03-20T12:00:00+00:00",
                  direction="bullish", confidence=0.55):
    """Generate n signals spaced 4 hours apart."""
    signals = []
    base = datetime.fromisoformat(start)
    for i in range(n):
        ts = (base + timedelta(hours=i * 4)).isoformat()
        signals.append(_make_signal(f"sig-{symbol}-{i:03d}", symbol,
                                    direction, confidence, ts))
    return signals


def _make_csv_prices(symbols, timestamps, base_prices, changes):
    """Create a temp CSV price file. changes is a list of pct changes."""
    lines = ["timestamp,symbol,close_price"]
    for sym, base_price in zip(symbols, base_prices):
        price = base_price
        for ts, chg in zip(timestamps, changes):
            price = price * (1 + chg)
            lines.append(f"{ts},{sym},{price:.2f}")
    path = tempfile.mktemp(suffix=".csv")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def _quick_resolver_and_signals(n=10, accuracy=0.5):
    """Create a CSV-backed resolver and signals with controllable accuracy."""
    base_ts = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
    signals = []
    timestamps = []
    # Generate signal timestamps and end timestamps
    for i in range(n):
        sig_ts = base_ts + timedelta(hours=i * 4)
        end_ts = sig_ts + timedelta(hours=24)
        signals.append(_make_signal(
            f"sig-{i:03d}", "BTC",
            "bullish", 0.55, sig_ts.isoformat()
        ))
        timestamps.append(sig_ts.isoformat())
        timestamps.append(end_ts.isoformat())

    # Build CSV: for 'accuracy' fraction, price goes UP (bullish correct)
    lines = ["timestamp,symbol,close_price"]
    n_correct = int(n * accuracy)
    for i in range(n):
        sig_ts = base_ts + timedelta(hours=i * 4)
        end_ts = sig_ts + timedelta(hours=24)
        start_price = 84000.0
        if i < n_correct:
            end_price = start_price + 100  # bullish correct
        else:
            end_price = start_price - 100  # bullish wrong
        lines.append(f"{sig_ts.isoformat()},BTC,{start_price:.2f}")
        lines.append(f"{end_ts.isoformat()},BTC,{end_price:.2f}")

    csv_path = tempfile.mktemp(suffix=".csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(lines))
    return signals, PriceResolver(csv_path=csv_path), csv_path


# ---------------------------------------------------------------------------
# Test: Wilson CI
# ---------------------------------------------------------------------------

class TestWilsonCI(unittest.TestCase):
    def test_zero_trials(self):
        ci = wilson_ci(0, 0)
        self.assertEqual(ci["n"], 0)
        self.assertEqual(ci["point"], 0.0)

    def test_all_hits(self):
        ci = wilson_ci(100, 100)
        self.assertAlmostEqual(ci["point"], 1.0, places=4)
        self.assertGreater(ci["lower"], 0.95)

    def test_no_hits(self):
        ci = wilson_ci(0, 100)
        self.assertAlmostEqual(ci["point"], 0.0, places=4)
        self.assertLess(ci["upper"], 0.05)

    def test_ci_contains_point(self):
        ci = wilson_ci(55, 100)
        self.assertLessEqual(ci["lower"], ci["point"])
        self.assertGreaterEqual(ci["upper"], ci["point"])

    def test_ci_narrows_with_n(self):
        ci_small = wilson_ci(5, 10)
        ci_large = wilson_ci(50, 100)
        width_small = ci_small["upper"] - ci_small["lower"]
        width_large = ci_large["upper"] - ci_large["lower"]
        self.assertGreater(width_small, width_large)


# ---------------------------------------------------------------------------
# Test: Brier Score
# ---------------------------------------------------------------------------

class TestBrierScore(unittest.TestCase):
    def test_perfect_forecast(self):
        bs = brier_score([1.0, 0.0], [1.0, 0.0])
        self.assertAlmostEqual(bs, 0.0, places=6)

    def test_worst_forecast(self):
        bs = brier_score([1.0, 0.0], [0.0, 1.0])
        self.assertAlmostEqual(bs, 1.0, places=6)

    def test_random_forecast(self):
        bs = brier_score([0.5, 0.5], [1.0, 0.0])
        self.assertAlmostEqual(bs, 0.25, places=6)

    def test_empty(self):
        bs = brier_score([], [])
        self.assertAlmostEqual(bs, 0.25)  # default

    def test_decomposition_sums(self):
        f = [0.55] * 10 + [0.52] * 10
        o = [1.0] * 6 + [0.0] * 4 + [1.0] * 5 + [0.0] * 5
        d = brier_decomposition(f, o)
        # Brier = reliability - resolution + uncertainty (approximately)
        self.assertEqual(d["n"], 20)
        self.assertGreater(d["brier_score"], 0)
        self.assertLess(d["brier_score"], 1)


# ---------------------------------------------------------------------------
# Test: Calibration Slope
# ---------------------------------------------------------------------------

class TestCalibrationSlope(unittest.TestCase):
    def test_perfect_calibration(self):
        # Confidence exactly matches outcome
        f = [0.0, 0.25, 0.5, 0.75, 1.0]
        o = [0.0, 0.25, 0.5, 0.75, 1.0]
        slope = calibration_slope(f, o)
        self.assertAlmostEqual(slope, 1.0, places=2)

    def test_constant_forecast(self):
        f = [0.5, 0.5, 0.5]
        o = [0.0, 1.0, 0.0]
        slope = calibration_slope(f, o)
        self.assertAlmostEqual(slope, 0.0, places=2)

    def test_too_few_points(self):
        slope = calibration_slope([0.5, 0.6], [1.0, 0.0])
        self.assertAlmostEqual(slope, 0.0)


# ---------------------------------------------------------------------------
# Test: CUSUM Drift Detection
# ---------------------------------------------------------------------------

class TestCUSUMNoDrift(unittest.TestCase):
    """CUSUM stays STABLE when accuracy matches target."""

    def test_stable_at_target(self):
        d = CUSUMDetector(target=0.5, allowance=0.02, threshold=1.5)
        for _ in range(100):
            d.update(0.5)
        self.assertEqual(d.to_dict()["drift_status"], "STABLE")

    def test_alternating_correct_incorrect(self):
        d = CUSUMDetector(target=0.5, allowance=0.02, threshold=1.5)
        for i in range(50):
            d.update(1.0 if i % 2 == 0 else 0.0)
        # Should remain STABLE or WATCH (random walk around target)
        self.assertIn(d.to_dict()["drift_status"], ["STABLE", "WATCH"])

    def test_observations_counted(self):
        d = CUSUMDetector(target=0.5)
        for _ in range(25):
            d.update(0.5)
        self.assertEqual(d.to_dict()["observations_since_reset"], 25)


class TestCUSUMGradualDrift(unittest.TestCase):
    """CUSUM detects gradual accuracy degradation."""

    def test_gradual_degradation(self):
        d = CUSUMDetector(target=0.55, allowance=0.02, threshold=1.5)
        # Feed mostly incorrect outcomes
        for _ in range(80):
            d.update(0.0)
        status = d.to_dict()
        self.assertIn(status["drift_status"], ["DRIFTING", "DEGRADED"])
        self.assertEqual(status["drift_direction"], "degrading")

    def test_gradual_improvement(self):
        d = CUSUMDetector(target=0.5, allowance=0.02, threshold=1.5)
        # Feed mostly correct outcomes
        for _ in range(80):
            d.update(1.0)
        status = d.to_dict()
        self.assertIn(status["drift_status"], ["DRIFTING", "DEGRADED"])
        self.assertEqual(status["drift_direction"], "improving")


class TestCUSUMSuddenDrift(unittest.TestCase):
    """CUSUM detects sudden accuracy collapse."""

    def test_sudden_collapse(self):
        d = CUSUMDetector(target=0.6, allowance=0.02, threshold=1.5)
        # 20 correct then 20 wrong
        for _ in range(20):
            d.update(1.0)
        for _ in range(20):
            d.update(0.0)
        # Should detect degrading
        status = d.to_dict()
        self.assertEqual(status["drift_direction"], "degrading")

    def test_sudden_improvement(self):
        d = CUSUMDetector(target=0.4, allowance=0.02, threshold=1.5)
        for _ in range(20):
            d.update(0.0)
        for _ in range(20):
            d.update(1.0)
        status = d.to_dict()
        self.assertEqual(status["drift_direction"], "improving")


class TestCUSUMReset(unittest.TestCase):
    def test_reset_clears_state(self):
        d = CUSUMDetector(target=0.5)
        for _ in range(50):
            d.update(0.0)
        d.reset()
        self.assertEqual(d.cusum_pos, 0.0)
        self.assertEqual(d.cusum_neg, 0.0)
        self.assertEqual(d.observations, 0)
        self.assertIsNone(d.drift_detected_at)

    def test_degraded_after_long_drift(self):
        d = CUSUMDetector(target=0.5, allowance=0.02, threshold=1.5)
        # 100 wrong outcomes
        for _ in range(100):
            d.update(0.0)
        self.assertEqual(d.to_dict()["drift_status"], "DEGRADED")

    def test_watch_near_threshold(self):
        d = CUSUMDetector(target=0.5, allowance=0.02, threshold=5.0)
        # Feed enough to approach but not exceed
        for _ in range(10):
            d.update(0.0)
        status = d.to_dict()["drift_status"]
        self.assertIn(status, ["STABLE", "WATCH"])


# ---------------------------------------------------------------------------
# Test: Rolling Windows
# ---------------------------------------------------------------------------

class TestRollingWindows(unittest.TestCase):
    def test_all_time_window_includes_all(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(20, accuracy=0.6)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            all_time = [w for w in report["rolling_windows"] if w["window_id"] == "all_time"]
            self.assertEqual(len(all_time), 1)
            self.assertEqual(all_time[0]["n_resolved"], 20)
        finally:
            os.unlink(csv_path)

    def test_7d_window_filters_old(self):
        # Create signals spanning 14 days — only last 7 days should be in 7d window
        base = datetime(2026, 3, 10, 12, 0, 0, tzinfo=timezone.utc)
        signals = []
        lines = ["timestamp,symbol,close_price"]
        for i in range(14):  # one signal per day
            sig_ts = base + timedelta(days=i)
            end_ts = sig_ts + timedelta(hours=24)
            signals.append(_make_signal(f"sig-{i}", "BTC", "bullish", 0.55,
                                       sig_ts.isoformat()))
            lines.append(f"{sig_ts.isoformat()},BTC,84000")
            lines.append(f"{end_ts.isoformat()},BTC,84100")  # bullish correct

        csv_path = tempfile.mktemp(suffix=".csv")
        with open(csv_path, "w") as f:
            f.write("\n".join(lines))
        try:
            resolver = PriceResolver(csv_path=csv_path)
            now = base + timedelta(days=14)
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver, now=now)

            w7d = [w for w in report["rolling_windows"] if w["window_id"] == "7d"]
            wall = [w for w in report["rolling_windows"] if w["window_id"] == "all_time"]
            self.assertEqual(len(w7d), 1)
            self.assertLessEqual(w7d[0]["n_resolved"], 7)
            self.assertEqual(wall[0]["n_resolved"], 14)
        finally:
            os.unlink(csv_path)

    def test_empty_window(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(5, accuracy=1.0)
        try:
            now = datetime(2026, 4, 15, 0, 0, 0, tzinfo=timezone.utc)
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver, now=now)
            w7d = [w for w in report["rolling_windows"] if w["window_id"] == "7d"]
            self.assertEqual(w7d[0]["n_resolved"], 0)
        finally:
            os.unlink(csv_path)

    def test_window_accuracy_computation(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(10, accuracy=0.7)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            all_time = [w for w in report["rolling_windows"] if w["window_id"] == "all_time"][0]
            self.assertAlmostEqual(all_time["accuracy"]["point"], 0.7, places=2)
        finally:
            os.unlink(csv_path)

    def test_per_symbol_breakdown(self):
        base = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        signals = []
        lines = ["timestamp,symbol,close_price"]
        for i, sym in enumerate(["BTC", "ETH"]):
            for j in range(5):
                sig_ts = base + timedelta(hours=(i * 5 + j) * 4)
                end_ts = sig_ts + timedelta(hours=24)
                signals.append(_make_signal(f"sig-{sym}-{j}", sym, "bullish", 0.55,
                                           sig_ts.isoformat()))
                lines.append(f"{sig_ts.isoformat()},{sym},{84000 if sym == 'BTC' else 1920}")
                end_p = 84100 if sym == "BTC" else 1910  # BTC up, ETH down
                lines.append(f"{end_ts.isoformat()},{sym},{end_p}")

        csv_path = tempfile.mktemp(suffix=".csv")
        with open(csv_path, "w") as f:
            f.write("\n".join(lines))
        try:
            resolver = PriceResolver(csv_path=csv_path)
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            all_time = [w for w in report["rolling_windows"] if w["window_id"] == "all_time"][0]
            self.assertIn("BTC", all_time["per_symbol"])
            self.assertIn("ETH", all_time["per_symbol"])
            self.assertEqual(all_time["per_symbol"]["BTC"]["n"], 5)
        finally:
            os.unlink(csv_path)


# ---------------------------------------------------------------------------
# Test: Rolling Metrics
# ---------------------------------------------------------------------------

class TestRollingMetrics(unittest.TestCase):
    def test_brier_score_range(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(20, accuracy=0.6)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            all_time = [w for w in report["rolling_windows"] if w["window_id"] == "all_time"][0]
            self.assertGreaterEqual(all_time["brier_score"], 0.0)
            self.assertLessEqual(all_time["brier_score"], 1.0)
        finally:
            os.unlink(csv_path)

    def test_reliability_lower_is_better(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(20, accuracy=0.6)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            all_time = [w for w in report["rolling_windows"] if w["window_id"] == "all_time"][0]
            self.assertGreaterEqual(all_time["reliability"], 0.0)
        finally:
            os.unlink(csv_path)

    def test_resolution_rate_computed(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(10, accuracy=0.5)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            all_time = [w for w in report["rolling_windows"] if w["window_id"] == "all_time"][0]
            self.assertGreater(all_time["resolution_rate"], 0.0)
        finally:
            os.unlink(csv_path)


# ---------------------------------------------------------------------------
# Test: Atomic Update
# ---------------------------------------------------------------------------

class TestAtomicUpdate(unittest.TestCase):
    def test_atomic_write_creates_file(self):
        path = tempfile.mktemp(suffix=".json")
        try:
            atomic_write(path, {"test": True})
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                data = json.load(f)
            self.assertTrue(data["test"])
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_atomic_write_includes_signals(self):
        path = tempfile.mktemp(suffix=".json")
        try:
            atomic_write(path, {"meta": "test"},
                        include_signals=True,
                        resolved_signals=[{"id": 1}])
            with open(path) as f:
                data = json.load(f)
            self.assertIn("resolved_signals", data)
            self.assertEqual(len(data["resolved_signals"]), 1)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_atomic_write_no_signals(self):
        path = tempfile.mktemp(suffix=".json")
        try:
            atomic_write(path, {"meta": "test"}, include_signals=False)
            with open(path) as f:
                data = json.load(f)
            self.assertNotIn("resolved_signals", data)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_atomic_write_overwrites(self):
        path = tempfile.mktemp(suffix=".json")
        try:
            atomic_write(path, {"version": 1})
            atomic_write(path, {"version": 2})
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["version"], 2)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_atomic_valid_json(self):
        path = tempfile.mktemp(suffix=".json")
        signals, resolver, csv_path = _quick_resolver_and_signals(5, accuracy=0.6)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            atomic_write(path, report, resolved_signals=m.get_resolved_signals())
            with open(path) as f:
                data = json.load(f)
            self.assertIn("meta", data)
            self.assertIn("resolved_signals", data)
        finally:
            if os.path.exists(path):
                os.unlink(path)
            os.unlink(csv_path)


# ---------------------------------------------------------------------------
# Test: Staleness Thresholds
# ---------------------------------------------------------------------------

class TestStaleness(unittest.TestCase):
    def test_live_when_fresh(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(5, accuracy=0.5)
        try:
            # Set now close to signal timestamps
            now = datetime(2026, 3, 20, 20, 0, 0, tzinfo=timezone.utc)
            m = ProofMaintainer(producer_id="test", staleness_hours=24)
            report = m.maintain(signals, resolver, now=now)
            self.assertEqual(report["freshness"]["freshness_grade"], "LIVE")
        finally:
            os.unlink(csv_path)

    def test_stale_when_old(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(5, accuracy=0.5)
        try:
            now = datetime(2026, 3, 25, 0, 0, 0, tzinfo=timezone.utc)
            m = ProofMaintainer(producer_id="test", staleness_hours=24)
            report = m.maintain(signals, resolver, now=now)
            self.assertIn(report["freshness"]["freshness_grade"], ["STALE", "EXPIRED"])
        finally:
            os.unlink(csv_path)

    def test_expired_when_very_old(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(5, accuracy=0.5)
        try:
            now = datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc)
            m = ProofMaintainer(producer_id="test", staleness_hours=24)
            report = m.maintain(signals, resolver, now=now)
            self.assertEqual(report["freshness"]["freshness_grade"], "EXPIRED")
        finally:
            os.unlink(csv_path)

    def test_custom_staleness_threshold(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(5, accuracy=0.5)
        try:
            now = datetime(2026, 3, 21, 18, 0, 0, tzinfo=timezone.utc)
            m = ProofMaintainer(producer_id="test", staleness_hours=6)
            report = m.maintain(signals, resolver, now=now)
            # With 6h threshold, signals from March 20 at +12h should be STALE
            grade = report["freshness"]["freshness_grade"]
            self.assertIn(grade, ["RECENT", "STALE", "EXPIRED"])
        finally:
            os.unlink(csv_path)

    def test_age_hours_computed(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(5, accuracy=0.5)
        try:
            now = datetime(2026, 3, 22, 12, 0, 0, tzinfo=timezone.utc)
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver, now=now)
            self.assertGreater(report["freshness"]["age_hours"], 0)
        finally:
            os.unlink(csv_path)


# ---------------------------------------------------------------------------
# Test: Empty Inputs
# ---------------------------------------------------------------------------

class TestEmptyInputs(unittest.TestCase):
    def test_no_signals(self):
        resolver = PriceResolver(csv_path=None)
        m = ProofMaintainer(producer_id="test")
        report = m.maintain([], resolver)
        self.assertEqual(report["snapshot"]["n_resolved_total"], 0)
        self.assertEqual(report["snapshot"]["n_new_resolved"], 0)
        self.assertEqual(len(report["rolling_windows"]), 4)

    def test_no_price_data(self):
        signals = _make_signals(5)
        resolver = PriceResolver(csv_path=None)  # No CSV, no yfinance
        m = ProofMaintainer(producer_id="test")
        report = m.maintain(signals, resolver)
        self.assertEqual(report["snapshot"]["n_resolved_total"], 0)
        self.assertEqual(report["resolution_delta"]["n_failed"], 5)

    def test_empty_csv(self):
        csv_path = tempfile.mktemp(suffix=".csv")
        with open(csv_path, "w") as f:
            f.write("timestamp,symbol,close_price\n")
        try:
            signals = _make_signals(3)
            resolver = PriceResolver(csv_path=csv_path)
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            self.assertEqual(report["snapshot"]["n_resolved_total"], 0)
        finally:
            os.unlink(csv_path)


# ---------------------------------------------------------------------------
# Test: Price Gaps
# ---------------------------------------------------------------------------

class TestPriceGaps(unittest.TestCase):
    def test_partial_price_data(self):
        """Only some signals have matching prices."""
        base = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        signals = _make_signals(5, start=base.isoformat())
        # Only provide prices for first 2 signals
        lines = ["timestamp,symbol,close_price"]
        for i in range(2):
            sig_ts = base + timedelta(hours=i * 4)
            end_ts = sig_ts + timedelta(hours=24)
            lines.append(f"{sig_ts.isoformat()},BTC,84000")
            lines.append(f"{end_ts.isoformat()},BTC,84100")

        csv_path = tempfile.mktemp(suffix=".csv")
        with open(csv_path, "w") as f:
            f.write("\n".join(lines))
        try:
            resolver = PriceResolver(csv_path=csv_path)
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            self.assertEqual(report["resolution_delta"]["n_resolved"], 2)
            self.assertEqual(report["resolution_delta"]["n_failed"], 3)
        finally:
            os.unlink(csv_path)

    def test_price_gap_limitation_detected(self):
        """High failure rate triggers PRICE_DATA_GAPS limitation."""
        signals = _make_signals(10)
        resolver = PriceResolver(csv_path=None)
        m = ProofMaintainer(producer_id="test")
        report = m.maintain(signals, resolver)
        limitation_ids = [l["id"] for l in report["limitations"]]
        self.assertIn("PRICE_DATA_GAPS", limitation_ids)

    def test_zero_start_price_handled(self):
        base = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        signals = [_make_signal("sig-0", "BTC", "bullish", 0.55, base.isoformat())]
        lines = ["timestamp,symbol,close_price"]
        lines.append(f"{base.isoformat()},BTC,0")
        end_ts = base + timedelta(hours=24)
        lines.append(f"{end_ts.isoformat()},BTC,100")

        csv_path = tempfile.mktemp(suffix=".csv")
        with open(csv_path, "w") as f:
            f.write("\n".join(lines))
        try:
            resolver = PriceResolver(csv_path=csv_path)
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            self.assertEqual(report["resolution_delta"]["n_resolved"], 0)
        finally:
            os.unlink(csv_path)


# ---------------------------------------------------------------------------
# Test: Schema Validation
# ---------------------------------------------------------------------------

class TestSchemaValidation(unittest.TestCase):
    def test_valid_report_passes(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(10, accuracy=0.5)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            errors = validate_report(report)
            self.assertEqual(len(errors), 0, f"Validation errors: {errors}")
        finally:
            os.unlink(csv_path)

    def test_empty_report_passes(self):
        resolver = PriceResolver(csv_path=None)
        m = ProofMaintainer(producer_id="test")
        report = m.maintain([], resolver)
        errors = validate_report(report)
        self.assertEqual(len(errors), 0, f"Validation errors: {errors}")

    def test_example_report_passes(self):
        example_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "example_proof", "proof_maintenance_example.json"
        )
        if os.path.exists(example_path):
            with open(example_path) as f:
                report = json.load(f)
            errors = validate_report(report)
            self.assertEqual(len(errors), 0, f"Validation errors: {errors}")

    def test_missing_field_detected(self):
        report = {"meta": {}, "protocol_version": "1.0.0"}
        errors = validate_report(report)
        self.assertGreater(len(errors), 0)


# ---------------------------------------------------------------------------
# Test: Cross-Schema Compatibility
# ---------------------------------------------------------------------------

class TestCrossSchemaCompat(unittest.TestCase):
    def test_protocol_version_semver(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(5, accuracy=0.5)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            import re
            semver = re.compile(r"^\d+\.\d+\.\d+$")
            self.assertRegex(report["protocol_version"], semver)
            self.assertRegex(report["meta"]["signal_schema_version"], semver)
            self.assertRegex(report["meta"]["resolution_protocol_version"], semver)
        finally:
            os.unlink(csv_path)

    def test_signal_fields_match_schema(self):
        """Resolved signals should contain fields compatible with pf-signal-schema."""
        signals, resolver, csv_path = _quick_resolver_and_signals(5, accuracy=0.5)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            resolved = m.get_resolved_signals()
            for sig in resolved:
                if sig.get("resolved"):
                    self.assertIn("symbol", sig)
                    self.assertIn("direction", sig)
                    self.assertIn("confidence", sig)
                    self.assertIn("timestamp", sig)
                    self.assertIn(sig["direction"], ["bullish", "bearish"])
                    self.assertGreaterEqual(sig["confidence"], 0)
                    self.assertLessEqual(sig["confidence"], 1)
        finally:
            os.unlink(csv_path)

    def test_reputation_grade_matches_resolution(self):
        """Grades should match pf-resolution-protocol: A/B/C/D/F."""
        signals, resolver, csv_path = _quick_resolver_and_signals(10, accuracy=0.5)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            self.assertIn(report["reputation"]["grade"], ["A", "B", "C", "D", "F"])
        finally:
            os.unlink(csv_path)

    def test_freshness_grade_enum(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(5, accuracy=0.5)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            self.assertIn(report["freshness"]["freshness_grade"],
                         ["LIVE", "RECENT", "STALE", "EXPIRED"])
        finally:
            os.unlink(csv_path)

    def test_drift_status_enum(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(5, accuracy=0.5)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            self.assertIn(report["drift"]["drift_status"],
                         ["STABLE", "WATCH", "DRIFTING", "DEGRADED"])
        finally:
            os.unlink(csv_path)


# ---------------------------------------------------------------------------
# Test: Cron Idempotency
# ---------------------------------------------------------------------------

class TestCronIdempotency(unittest.TestCase):
    def test_rerun_same_signals_no_change(self):
        """Running twice with the same signals should not duplicate resolutions."""
        signals, resolver, csv_path = _quick_resolver_and_signals(10, accuracy=0.6)
        try:
            m = ProofMaintainer(producer_id="test")
            now = datetime(2026, 3, 22, 0, 0, 0, tzinfo=timezone.utc)
            report1 = m.maintain(signals, resolver, now=now)
            report2 = m.maintain(signals, resolver, now=now + timedelta(minutes=15))

            self.assertEqual(report1["snapshot"]["n_resolved_total"],
                           report2["snapshot"]["n_resolved_total"])
            self.assertEqual(report2["snapshot"]["n_new_resolved"], 0)
            self.assertEqual(report2["resolution_delta"]["n_skipped"], 10)
        finally:
            os.unlink(csv_path)

    def test_incremental_add_new_signals(self):
        """Adding new signals to existing set resolves only the new ones."""
        signals_batch1, resolver, csv_path = _quick_resolver_and_signals(5, accuracy=0.6)
        try:
            m = ProofMaintainer(producer_id="test")
            now = datetime(2026, 3, 22, 0, 0, 0, tzinfo=timezone.utc)
            report1 = m.maintain(signals_batch1, resolver, now=now)

            # Add 5 more signals
            base = datetime(2026, 3, 21, 12, 0, 0, tzinfo=timezone.utc)
            batch2 = []
            lines = ["timestamp,symbol,close_price"]
            with open(csv_path) as f:
                existing = f.read()
            for i in range(5):
                sig_ts = base + timedelta(hours=i * 4)
                end_ts = sig_ts + timedelta(hours=24)
                batch2.append(_make_signal(f"sig-new-{i}", "BTC", "bullish", 0.55,
                                          sig_ts.isoformat()))
                lines.append(f"{sig_ts.isoformat()},BTC,84000")
                lines.append(f"{end_ts.isoformat()},BTC,84100")

            with open(csv_path, "a") as f:
                f.write("\n" + "\n".join(lines[1:]))

            resolver2 = PriceResolver(csv_path=csv_path)
            all_signals = signals_batch1 + batch2
            report2 = m.maintain(all_signals, resolver2, now=now + timedelta(hours=1))

            self.assertEqual(report2["snapshot"]["n_resolved_total"], 10)
            self.assertEqual(report2["snapshot"]["n_new_resolved"], 5)
        finally:
            os.unlink(csv_path)

    def test_snapshot_version_increments(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(5, accuracy=0.5)
        try:
            m = ProofMaintainer(producer_id="test")
            now = datetime(2026, 3, 22, 0, 0, 0, tzinfo=timezone.utc)
            r1 = m.maintain(signals, resolver, now=now)
            self.assertEqual(r1["snapshot"]["snapshot_version"], 1)
            # Simulate snapshot persistence
            m._previous_snapshot = r1["snapshot"]
            r2 = m.maintain(signals, resolver, now=now + timedelta(hours=1))
            self.assertEqual(r2["snapshot"]["snapshot_version"], 2)
        finally:
            os.unlink(csv_path)


# ---------------------------------------------------------------------------
# Test: Report Structure
# ---------------------------------------------------------------------------

class TestReportStructure(unittest.TestCase):
    def test_required_top_level_keys(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(5, accuracy=0.5)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            for key in ["meta", "freshness", "snapshot", "resolution_delta",
                        "rolling_windows", "drift", "calibration_trajectory",
                        "reputation", "protocol_version", "limitations"]:
                self.assertIn(key, report, f"Missing key: {key}")
        finally:
            os.unlink(csv_path)

    def test_meta_fields(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(5, accuracy=0.5)
        try:
            m = ProofMaintainer(producer_id="my-producer", wallet="rWallet123")
            report = m.maintain(signals, resolver)
            self.assertEqual(report["meta"]["producer_id"], "my-producer")
            self.assertEqual(report["meta"]["wallet"], "rWallet123")
            self.assertEqual(report["meta"]["protocol_version"], PROTOCOL_VERSION)
        finally:
            os.unlink(csv_path)

    def test_four_rolling_windows(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(5, accuracy=0.5)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            window_ids = [w["window_id"] for w in report["rolling_windows"]]
            self.assertIn("7d", window_ids)
            self.assertIn("14d", window_ids)
            self.assertIn("30d", window_ids)
            self.assertIn("all_time", window_ids)
        finally:
            os.unlink(csv_path)


# ---------------------------------------------------------------------------
# Test: Limitations Detection
# ---------------------------------------------------------------------------

class TestLimitations(unittest.TestCase):
    def test_small_sample_detected(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(10, accuracy=0.5)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            lim_ids = [l["id"] for l in report["limitations"]]
            self.assertIn("SMALL_SAMPLE", lim_ids)
        finally:
            os.unlink(csv_path)

    def test_no_small_sample_when_enough(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(150, accuracy=0.5)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            lim_ids = [l["id"] for l in report["limitations"]]
            self.assertNotIn("SMALL_SAMPLE", lim_ids)
        finally:
            os.unlink(csv_path)

    def test_limitation_has_bias_fields(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(10, accuracy=0.5)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            for lim in report["limitations"]:
                self.assertIn("id", lim)
                self.assertIn("description", lim)
                self.assertIn("bias_direction", lim)
                self.assertIn("bias_magnitude", lim)
        finally:
            os.unlink(csv_path)

    def test_stale_proof_limitation(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(5, accuracy=0.5)
        try:
            now = datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc)
            m = ProofMaintainer(producer_id="test", staleness_hours=24)
            report = m.maintain(signals, resolver, now=now)
            lim_ids = [l["id"] for l in report["limitations"]]
            self.assertIn("STALE_PROOF", lim_ids)
        finally:
            os.unlink(csv_path)


# ---------------------------------------------------------------------------
# Test: Calibration Trajectory
# ---------------------------------------------------------------------------

class TestCalibrationTrajectory(unittest.TestCase):
    def test_checkpoints_generated(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(200, accuracy=0.55)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            traj = report["calibration_trajectory"]
            self.assertGreater(len(traj["checkpoints"]), 0)
            for cp in traj["checkpoints"]:
                self.assertIn("n_cumulative", cp)
                self.assertIn("accuracy", cp)
                self.assertIn("brier_score", cp)
        finally:
            os.unlink(csv_path)

    def test_trend_computed_with_enough_data(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(400, accuracy=0.55)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            traj = report["calibration_trajectory"]
            if len(traj["checkpoints"]) >= 3:
                self.assertIsNotNone(traj["trend"])
                self.assertIn("slope_per_day", traj["trend"])
        finally:
            os.unlink(csv_path)

    def test_no_trend_with_few_checkpoints(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(10, accuracy=0.5)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            traj = report["calibration_trajectory"]
            self.assertIsNone(traj["trend"])
        finally:
            os.unlink(csv_path)


# ---------------------------------------------------------------------------
# Test: Signal Loading
# ---------------------------------------------------------------------------

class TestSignalLoading(unittest.TestCase):
    def test_load_array_format(self):
        signals = [
            {"signal_id": "s1", "symbol": "BTC", "direction": "bullish",
             "confidence": 0.55, "timestamp": "2026-03-20T12:00:00+00:00"}
        ]
        loaded = load_signals_from_array(signals)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["symbol"], "BTC")

    def test_load_log_format(self):
        log_data = {
            "runs": [{
                "timestamp": "2026-03-20T12:00:00+00:00",
                "regime": "SYSTEMIC",
                "signals_sent": [{
                    "signal": {
                        "symbol": "BTC",
                        "direction": "bullish",
                        "confidence": 0.55,
                        "horizon_hours": 24,
                    },
                    "response": json.dumps({"signal_id": "s1"}),
                    "http_status": 201,
                }]
            }]
        }
        path = tempfile.mktemp(suffix=".json")
        with open(path, "w") as f:
            json.dump(log_data, f)
        try:
            loaded = load_signals_from_log(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["symbol"], "BTC")
        finally:
            os.unlink(path)

    def test_equity_signals_filtered(self):
        log_data = {
            "runs": [{
                "timestamp": "2026-03-20T12:00:00+00:00",
                "signals_sent": [
                    {"signal": {"symbol": "BTC", "direction": "bullish", "confidence": 0.5},
                     "response": "{}", "http_status": 201},
                    {"signal": {"symbol": "NVDA", "direction": "bullish", "confidence": 0.5},
                     "response": "{}", "http_status": 201},
                ]
            }]
        }
        path = tempfile.mktemp(suffix=".json")
        with open(path, "w") as f:
            json.dump(log_data, f)
        try:
            loaded = load_signals_from_log(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["symbol"], "BTC")
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test: File I/O End-to-End
# ---------------------------------------------------------------------------

class TestFileIO(unittest.TestCase):
    def test_output_and_reload(self):
        """Write proof surface, reload it, run again — idempotent."""
        signals, resolver, csv_path = _quick_resolver_and_signals(10, accuracy=0.6)
        proof_path = tempfile.mktemp(suffix=".json")
        try:
            m1 = ProofMaintainer(producer_id="test")
            report1 = m1.maintain(signals, resolver)
            atomic_write(proof_path, report1, resolved_signals=m1.get_resolved_signals())

            # Reload and re-run
            m2 = ProofMaintainer(producer_id="test")
            m2.load_existing_proof(proof_path)
            report2 = m2.maintain(signals, resolver)

            self.assertEqual(report2["snapshot"]["n_new_resolved"], 0)
            self.assertEqual(report2["snapshot"]["n_resolved_total"],
                           report1["snapshot"]["n_resolved_total"])
        finally:
            os.unlink(csv_path)
            if os.path.exists(proof_path):
                os.unlink(proof_path)

    def test_example_output_exists(self):
        example_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "example_proof", "proof_maintenance_example.json"
        )
        self.assertTrue(os.path.exists(example_path))


# ---------------------------------------------------------------------------
# Test: End-to-End Pipeline
# ---------------------------------------------------------------------------

class TestEndToEnd(unittest.TestCase):
    def test_full_pipeline_with_csv(self):
        """Complete pipeline: load signals, resolve, compute metrics, validate."""
        signals, resolver, csv_path = _quick_resolver_and_signals(20, accuracy=0.55)
        try:
            m = ProofMaintainer(
                producer_id="e2e-test",
                wallet="rTestWallet",
                staleness_hours=48,
            )
            now = datetime(2026, 3, 22, 0, 0, 0, tzinfo=timezone.utc)
            report = m.maintain(signals, resolver, now=now)

            # Verify structure
            self.assertEqual(report["meta"]["producer_id"], "e2e-test")
            self.assertEqual(report["snapshot"]["n_resolved_total"], 20)
            self.assertEqual(len(report["rolling_windows"]), 4)

            # Validate against schema
            errors = validate_report(report)
            self.assertEqual(len(errors), 0, f"Validation errors: {errors}")
        finally:
            os.unlink(csv_path)

    def test_pipeline_with_no_resolvable_signals(self):
        signals = _make_signals(5)
        resolver = PriceResolver(csv_path=None)
        m = ProofMaintainer(producer_id="test")
        report = m.maintain(signals, resolver)

        self.assertEqual(report["snapshot"]["n_resolved_total"], 0)
        errors = validate_report(report)
        self.assertEqual(len(errors), 0, f"Validation errors: {errors}")

    def test_pipeline_json_serializable(self):
        signals, resolver, csv_path = _quick_resolver_and_signals(10, accuracy=0.5)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            serialized = json.dumps(report, default=str)
            self.assertIsInstance(serialized, str)
            reparsed = json.loads(serialized)
            self.assertEqual(reparsed["meta"]["producer_id"], "test")
        finally:
            os.unlink(csv_path)

    def test_window_divergence_limitation(self):
        """If 7d accuracy differs significantly from all-time, detect it."""
        # Create signals: first batch low accuracy, recent batch high accuracy
        base = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        signals = []
        lines = ["timestamp,symbol,close_price"]

        # 100 old signals (low accuracy)
        for i in range(100):
            sig_ts = base + timedelta(hours=i * 4)
            end_ts = sig_ts + timedelta(hours=24)
            signals.append(_make_signal(f"old-{i}", "BTC", "bullish", 0.55,
                                       sig_ts.isoformat()))
            lines.append(f"{sig_ts.isoformat()},BTC,84000")
            if i < 40:  # 40% accuracy
                lines.append(f"{end_ts.isoformat()},BTC,84100")
            else:
                lines.append(f"{end_ts.isoformat()},BTC,83900")

        # 30 recent signals (high accuracy)
        recent_base = datetime(2026, 3, 25, 12, 0, 0, tzinfo=timezone.utc)
        for i in range(30):
            sig_ts = recent_base + timedelta(hours=i * 4)
            end_ts = sig_ts + timedelta(hours=24)
            signals.append(_make_signal(f"new-{i}", "BTC", "bullish", 0.55,
                                       sig_ts.isoformat()))
            lines.append(f"{sig_ts.isoformat()},BTC,84000")
            if i < 24:  # 80% accuracy
                lines.append(f"{end_ts.isoformat()},BTC,84100")
            else:
                lines.append(f"{end_ts.isoformat()},BTC,83900")

        csv_path = tempfile.mktemp(suffix=".csv")
        with open(csv_path, "w") as f:
            f.write("\n".join(lines))

        try:
            resolver = PriceResolver(csv_path=csv_path)
            now = datetime(2026, 3, 31, 0, 0, 0, tzinfo=timezone.utc)
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver, now=now)
            lim_ids = [l["id"] for l in report["limitations"]]
            self.assertIn("WINDOW_DIVERGENCE", lim_ids)
        finally:
            os.unlink(csv_path)


class TestReputationScoring(unittest.TestCase):
    """Tests for Producer Reputation Score computation."""

    def test_perfect_accuracy_high_score(self):
        """Perfect accuracy should yield higher reputation than random."""
        sigs_perf, res_perf, csv_perf = _quick_resolver_and_signals(20, accuracy=1.0)
        sigs_rand, res_rand, csv_rand = _quick_resolver_and_signals(20, accuracy=0.5)
        try:
            m1 = ProofMaintainer(producer_id="test")
            r1 = m1.maintain(sigs_perf, res_perf)
            m2 = ProofMaintainer(producer_id="test")
            r2 = m2.maintain(sigs_rand, res_rand)
            self.assertGreater(r1["reputation"]["score"], r2["reputation"]["score"])
        finally:
            os.unlink(csv_perf)
            os.unlink(csv_rand)

    def test_random_accuracy_low_score(self):
        """50% accuracy should yield low reputation."""
        signals, resolver, csv_path = _quick_resolver_and_signals(20, accuracy=0.5)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            self.assertLess(report["reputation"]["score"], 0.5)
        finally:
            os.unlink(csv_path)

    def test_grade_assignment(self):
        """Grade should match score thresholds."""
        signals, resolver, csv_path = _quick_resolver_and_signals(20, accuracy=1.0)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            grade = report["reputation"]["grade"]
            self.assertIn(grade, ["A", "B", "C", "D", "F"])
        finally:
            os.unlink(csv_path)

    def test_no_signals_f_grade(self):
        """No resolved signals should give F grade."""
        m = ProofMaintainer(producer_id="test")
        resolver = PriceResolver(csv_path=None)
        report = m.maintain([], resolver)
        self.assertEqual(report["reputation"]["grade"], "F")
        self.assertEqual(report["reputation"]["score"], 0.0)

    def test_reputation_delta_on_reload(self):
        """After loading existing proof, second run shows score delta."""
        signals, resolver, csv_path = _quick_resolver_and_signals(10, accuracy=0.7)
        proof_path = tempfile.mktemp(suffix=".json")
        try:
            m1 = ProofMaintainer(producer_id="test")
            now = datetime(2026, 3, 22, 0, 0, 0, tzinfo=timezone.utc)
            r1 = m1.maintain(signals, resolver, now=now)
            self.assertIsNone(r1["reputation"]["previous_score"])

            # Write proof, reload, re-run — delta should appear
            atomic_write(proof_path, r1, resolved_signals=m1.get_resolved_signals())
            m2 = ProofMaintainer(producer_id="test")
            m2.load_existing_proof(proof_path)
            r2 = m2.maintain(signals, resolver, now=now + timedelta(hours=1))
            self.assertIsNotNone(r2["reputation"]["previous_score"])
            self.assertIsNotNone(r2["reputation"]["score_delta"])
        finally:
            os.unlink(csv_path)
            if os.path.exists(proof_path):
                os.unlink(proof_path)


class TestMultiSymbol(unittest.TestCase):
    """Tests for multi-symbol signal handling."""

    def test_per_symbol_accuracy_varies(self):
        """Different symbols can have different accuracy."""
        base = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        signals = []
        lines = ["timestamp,symbol,close_price"]

        # BTC: 100% accuracy (5 signals)
        for i in range(5):
            sig_ts = base + timedelta(hours=i * 4)
            end_ts = sig_ts + timedelta(hours=24)
            signals.append(_make_signal(f"btc-{i}", "BTC", "bullish", 0.6,
                                        sig_ts.isoformat()))
            lines.append(f"{sig_ts.isoformat()},BTC,84000")
            lines.append(f"{end_ts.isoformat()},BTC,84200")

        # SOL: 0% accuracy (5 signals)
        for i in range(5):
            sig_ts = base + timedelta(hours=i * 4)
            end_ts = sig_ts + timedelta(hours=24)
            signals.append(_make_signal(f"sol-{i}", "SOL", "bullish", 0.6,
                                        sig_ts.isoformat()))
            lines.append(f"{sig_ts.isoformat()},SOL,135")
            lines.append(f"{end_ts.isoformat()},SOL,134")

        csv_path = tempfile.mktemp(suffix=".csv")
        with open(csv_path, "w") as f:
            f.write("\n".join(lines))

        try:
            resolver = PriceResolver(csv_path=csv_path)
            now = datetime(2026, 3, 25, 0, 0, 0, tzinfo=timezone.utc)
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver, now=now)

            all_time = [w for w in report["rolling_windows"]
                       if w["window_id"] == "all_time"][0]
            self.assertEqual(all_time["per_symbol"]["BTC"]["accuracy"], 1.0)
            self.assertEqual(all_time["per_symbol"]["SOL"]["accuracy"], 0.0)
        finally:
            os.unlink(csv_path)

    def test_overall_accuracy_is_aggregate(self):
        """Overall accuracy should reflect all symbols combined."""
        signals, resolver, csv_path = _quick_resolver_and_signals(20, accuracy=0.6)
        try:
            m = ProofMaintainer(producer_id="test")
            report = m.maintain(signals, resolver)
            all_time = [w for w in report["rolling_windows"]
                       if w["window_id"] == "all_time"][0]
            self.assertAlmostEqual(all_time["accuracy"]["point"], 0.6, delta=0.15)
        finally:
            os.unlink(csv_path)


class TestCUSUMParameters(unittest.TestCase):
    """Tests for CUSUM parameter customization."""

    def test_custom_allowance(self):
        """Custom allowance K changes CUSUM sensitivity."""
        d1 = CUSUMDetector(target=0.5, allowance=0.01, threshold=1.5)
        d2 = CUSUMDetector(target=0.5, allowance=0.10, threshold=1.5)

        # Feed same degrading data
        for _ in range(50):
            d1.update(0.0)
            d2.update(0.0)

        # Tighter allowance accumulates faster
        self.assertGreater(d1.cusum_neg, d2.cusum_neg)

    def test_custom_threshold(self):
        """Higher threshold H requires more evidence for drift detection."""
        d_low = CUSUMDetector(target=0.5, allowance=0.02, threshold=0.5)
        d_high = CUSUMDetector(target=0.5, allowance=0.02, threshold=5.0)

        # 5 wrong observations: cusum_neg = 5*(0.5-0.02) = 2.4
        # Low threshold (0.5) exceeded, high threshold (5.0) not
        for _ in range(5):
            d_low.update(0.0)
            d_high.update(0.0)

        d_low_state = d_low.to_dict()
        d_high_state = d_high.to_dict()
        self.assertIn(d_low_state["drift_status"], ["DRIFTING", "DEGRADED"])
        self.assertIn(d_high_state["drift_status"], ["STABLE", "WATCH"])


if __name__ == "__main__":
    unittest.main()
