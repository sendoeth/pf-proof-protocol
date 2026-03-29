#!/usr/bin/env python3
"""
Continuous Proof Maintenance Protocol
======================================
Standalone CLI + cron-deployable script that maintains a live, rolling proof
surface for a Post Fiat signal producer. Resolves unresolved signals against
market prices, computes rolling accuracy/Brier per window (7d/14d/30d/all-time),
runs CUSUM calibration drift detection, and atomically updates the proof surface
with append-only versioned snapshots and staleness guarantees.

Part of the pf-proof-protocol — fifth companion to pf-signal-schema,
pf-routing-protocol, pf-resolution-protocol, and pf-aggregation-protocol.

Dependencies: jsonschema (required), yfinance + pandas + numpy + scipy (for
live price resolution; CSV fallback available without these).

Usage:
    # Resolve new signals and update proof surface
    python maintain_proof.py signal_log.json --proof-surface proof_surface.json

    # With CSV price fallback (no yfinance needed)
    python maintain_proof.py signal_log.json --price-csv prices.csv

    # JSON output
    python maintain_proof.py signal_log.json --json

    # Cron mode (silent unless drift detected)
    python maintain_proof.py signal_log.json --cron

    # Validate existing report against schema
    python maintain_proof.py signal_log.json --validate
"""

import json
import os
import sys
import math
import copy
import shutil
import tempfile
import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

try:
    import jsonschema
    from jsonschema import Draft202012Validator
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from scipy import stats as sp_stats
    HAS_SCIENCE = True
except ImportError:
    HAS_SCIENCE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = "1.0.0"
SIGNAL_SCHEMA_VERSION = "1.0.0"
RESOLUTION_PROTOCOL_VERSION = "1.0.0"

CRYPTO_SYMBOLS = ["BTC", "ETH", "SOL", "LINK"]
YF_TICKERS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "LINK": "LINK-USD",
}

DEFAULT_STALENESS_HOURS = 24
DEFAULT_CUSUM_ALLOWANCE = 0.02  # K: half the shift we want to detect (4%)
DEFAULT_CUSUM_THRESHOLD = 1.5   # H: decision threshold
DEFAULT_WINDOW_DAYS = [7, 14, 30, 0]  # 0 = all_time
WINDOW_IDS = {7: "7d", 14: "14d", 30: "30d", 0: "all_time"}
CHECKPOINT_INTERVAL = 100  # signals between calibration checkpoints

EPS = 1e-12
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proof_protocol.json")


# ---------------------------------------------------------------------------
# Math Utilities
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, trials: int, z: float = 1.645) -> Dict[str, Any]:
    """Wilson score CI for binomial proportion. Default z=1.645 for 90% CI."""
    if trials == 0:
        return {"point": 0.0, "lower": 0.0, "upper": 0.0, "n": 0}
    p_hat = successes / trials
    denom = 1 + z ** 2 / trials
    center = (p_hat + z ** 2 / (2 * trials)) / denom
    margin = (z / denom) * math.sqrt(
        p_hat * (1 - p_hat) / trials + z ** 2 / (4 * trials ** 2)
    )
    return {
        "point": round(p_hat, 6),
        "lower": round(max(0, center - margin), 6),
        "upper": round(min(1, center + margin), 6),
        "n": trials,
    }


def brier_score(forecasts: List[float], outcomes: List[float]) -> float:
    """Mean Brier score. 0 = perfect, 0.25 = random."""
    if not forecasts:
        return 0.25
    return sum((f - o) ** 2 for f, o in zip(forecasts, outcomes)) / len(forecasts)


def brier_decomposition(forecasts: List[float], outcomes: List[float],
                         n_bins: int = 10) -> Dict[str, Any]:
    """Murphy-Winkler Brier decomposition into reliability, resolution, uncertainty."""
    if not forecasts or len(forecasts) != len(outcomes):
        return {"brier_score": 0.25, "reliability": 0.0, "resolution": 0.0,
                "uncertainty": 0.25, "n": 0}

    n = len(forecasts)
    bs = sum((f - o) ** 2 for f, o in zip(forecasts, outcomes)) / n
    o_bar = sum(outcomes) / n
    uncertainty = o_bar * (1 - o_bar)

    # Bin forecasts
    reliability = 0.0
    resolution = 0.0
    for k in range(n_bins):
        lo = k / n_bins
        hi = (k + 1) / n_bins
        bin_f = [f for f in forecasts if lo <= f < hi or (k == n_bins - 1 and f == 1.0)]
        bin_o = [o for f, o in zip(forecasts, outcomes)
                 if lo <= f < hi or (k == n_bins - 1 and f == 1.0)]
        n_k = len(bin_f)
        if n_k == 0:
            continue
        f_k = sum(bin_f) / n_k
        o_k = sum(bin_o) / n_k
        w = n_k / n
        reliability += w * (f_k - o_k) ** 2
        resolution += w * (o_k - o_bar) ** 2

    return {
        "brier_score": round(bs, 6),
        "reliability": round(reliability, 6),
        "resolution": round(resolution, 6),
        "uncertainty": round(uncertainty, 6),
        "n": n,
    }


def calibration_slope(forecasts: List[float], outcomes: List[float]) -> float:
    """Linear regression slope of outcomes on forecasts. 1.0 = perfect."""
    if len(forecasts) < 3:
        return 0.0
    if HAS_SCIENCE:
        f = np.array(forecasts)
        o = np.array(outcomes)
        if np.std(f) < EPS:
            return 0.0
        slope, _, _, _, _ = sp_stats.linregress(f, o)
        return round(float(slope), 4)
    else:
        # Fallback: manual OLS
        n = len(forecasts)
        mean_f = sum(forecasts) / n
        mean_o = sum(outcomes) / n
        cov = sum((f - mean_f) * (o - mean_o) for f, o in zip(forecasts, outcomes)) / n
        var_f = sum((f - mean_f) ** 2 for f in forecasts) / n
        if var_f < EPS:
            return 0.0
        return round(cov / var_f, 4)


# ---------------------------------------------------------------------------
# CUSUM Drift Detector
# ---------------------------------------------------------------------------

class CUSUMDetector:
    """
    Cumulative Sum (CUSUM) chart for detecting calibration drift.

    Monitors accuracy observations against a target rate. When systematic
    deviations accumulate beyond the threshold H, drift is flagged.

    Two-sided: tracks both positive (improving) and negative (degrading) shifts.

    Parameters:
        target: Target accuracy (baseline). Typically historical all-time accuracy.
        allowance (K): Slack parameter. Shifts must exceed K per observation
            to accumulate. Default: 0.02 (detects 4% shift).
        threshold (H): Decision threshold. CUSUM exceeding H triggers drift.
            Default: 1.5 (ARL ~100 observations at no-shift).
    """

    def __init__(self, target: float = 0.5,
                 allowance: float = DEFAULT_CUSUM_ALLOWANCE,
                 threshold: float = DEFAULT_CUSUM_THRESHOLD):
        self.target = target
        self.allowance = allowance
        self.threshold = threshold
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.drift_detected_at: Optional[str] = None
        self.observations = 0
        self.run_length = 0

    def update(self, observation: float, timestamp: Optional[str] = None) -> str:
        """
        Process one accuracy observation (0 or 1 for single signal,
        or fractional for batch accuracy).

        Returns drift_status: STABLE, WATCH, DRIFTING, DEGRADED.
        """
        deviation = observation - self.target

        # Two-sided CUSUM
        self.cusum_pos = max(0.0, self.cusum_pos + deviation - self.allowance)
        self.cusum_neg = max(0.0, self.cusum_neg - deviation - self.allowance)
        self.observations += 1

        # Determine status
        max_cusum = max(self.cusum_pos, self.cusum_neg)

        if max_cusum >= self.threshold:
            if self.drift_detected_at is None:
                self.drift_detected_at = timestamp
            self.run_length += 1
            if self.run_length >= 50:
                return "DEGRADED"
            return "DRIFTING"
        elif max_cusum >= self.threshold * 0.6:
            self.run_length += 1
            return "WATCH"
        else:
            self.run_length = 0
            self.drift_detected_at = None
            return "STABLE"

    def get_drift_direction(self) -> str:
        """Which direction is the drift, if any?"""
        if self.cusum_pos > self.cusum_neg and self.cusum_pos >= self.threshold * 0.6:
            return "improving"
        elif self.cusum_neg > self.cusum_pos and self.cusum_neg >= self.threshold * 0.6:
            return "degrading"
        return "none"

    def reset(self):
        """Reset CUSUM state."""
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.drift_detected_at = None
        self.observations = 0
        self.run_length = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize CUSUM state for proof report."""
        max_cusum = max(self.cusum_pos, self.cusum_neg)
        if max_cusum >= self.threshold:
            if self.run_length >= 50:
                status = "DEGRADED"
            else:
                status = "DRIFTING"
        elif max_cusum >= self.threshold * 0.6:
            status = "WATCH"
        else:
            status = "STABLE"

        return {
            "cusum_pos": round(self.cusum_pos, 6),
            "cusum_neg": round(self.cusum_neg, 6),
            "drift_status": status,
            "drift_detected_at": self.drift_detected_at,
            "drift_direction": self.get_drift_direction(),
            "target_accuracy": self.target,
            "allowance": self.allowance,
            "threshold": self.threshold,
            "observations_since_reset": self.observations,
            "run_length": self.run_length,
        }


# ---------------------------------------------------------------------------
# Price Resolution Layer
# ---------------------------------------------------------------------------

class PriceResolver:
    """Fetches prices for signal resolution. Supports Yahoo Finance + CSV fallback."""

    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = csv_path
        self._csv_data: Optional[Dict] = None
        self._yf_data: Dict[str, Any] = {}
        self._source = "csv_fallback" if csv_path else "yahoo_finance"

    def _load_csv(self) -> Dict[str, Dict[str, float]]:
        """Load CSV with columns: timestamp, symbol, close_price."""
        if self._csv_data is not None:
            return self._csv_data
        if not self.csv_path or not os.path.exists(self.csv_path):
            return {}
        prices: Dict[str, Dict[str, float]] = defaultdict(dict)
        with open(self.csv_path) as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    ts, symbol, price = parts[0], parts[1], float(parts[2])
                    prices[symbol][ts] = price
        self._csv_data = dict(prices)
        return self._csv_data

    def _fetch_yf(self, symbol: str, start: str, end: str) -> bool:
        """Fetch hourly data from Yahoo Finance for a symbol."""
        if not HAS_SCIENCE:
            return False
        if symbol in self._yf_data:
            return True
        if symbol not in YF_TICKERS:
            return False
        try:
            ticker = YF_TICKERS[symbol]
            tk = yf.Ticker(ticker)
            df = tk.history(start=start, end=end, interval="1h")
            if df.empty:
                return False
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            elif str(df.index.tz) != "UTC":
                df.index = df.index.tz_convert("UTC")
            full_range = pd.date_range(
                start=df.index.min(), end=df.index.max(), freq="h", tz="UTC"
            )
            df = df.reindex(full_range)
            df = df.ffill(limit=4)
            self._yf_data[symbol] = df
            return True
        except Exception:
            return False

    def get_price_at(self, symbol: str, timestamp: datetime) -> Optional[float]:
        """Get price at a specific timestamp."""
        if self.csv_path:
            csv = self._load_csv()
            sym_prices = csv.get(symbol, {})
            ts_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
            for key in [ts_str, ts_str + "+00:00", ts_str + "Z"]:
                if key in sym_prices:
                    return sym_prices[key]
            # Find nearest within 2 hours
            ts_epoch = timestamp.timestamp()
            best = None
            best_gap = float("inf")
            for k, v in sym_prices.items():
                try:
                    kts = datetime.fromisoformat(k.replace("Z", "+00:00")).timestamp()
                    gap = abs(kts - ts_epoch)
                    if gap < best_gap and gap <= 7200:
                        best_gap = gap
                        best = v
                except (ValueError, TypeError):
                    continue
            return best

        if not HAS_SCIENCE or symbol not in self._yf_data:
            return None
        df = self._yf_data[symbol]
        ts = pd.Timestamp(timestamp)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        idx = df.index.get_indexer([ts], method="nearest")[0]
        if idx < 0 or idx >= len(df):
            return None
        nearest = df.index[idx]
        if abs((nearest - ts).total_seconds()) > 7200:
            return None
        close = df.iloc[idx]["Close"]
        if HAS_SCIENCE and pd.isna(close):
            return None
        return float(close)

    def resolve_signal(self, signal: Dict) -> Dict:
        """Resolve a single signal against price data."""
        ts_str = signal.get("timestamp", "")
        try:
            signal_time = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return {**signal, "resolved": False, "resolution_reason": "invalid_timestamp"}

        horizon = signal.get("horizon_hours", 24)
        window_end = signal_time + timedelta(hours=horizon)
        symbol = signal.get("symbol", "")

        start_price = self.get_price_at(symbol, signal_time)
        end_price = self.get_price_at(symbol, window_end)

        if start_price is None or end_price is None:
            return {**signal, "resolved": False, "resolution_reason": "missing_price_data"}
        if start_price == 0:
            return {**signal, "resolved": False, "resolution_reason": "zero_start_price"}

        pct_change = (end_price - start_price) / start_price
        actual_direction = "bullish" if pct_change >= 0 else "bearish"
        direction_correct = (actual_direction == signal.get("direction", ""))
        outcome = 1.0 if direction_correct else 0.0
        bs = (signal.get("confidence", 0.5) - outcome) ** 2

        return {
            **signal,
            "resolved": True,
            "resolution_reason": "price_data_available",
            "start_price": round(start_price, 6),
            "end_price": round(end_price, 6),
            "pct_change": round(pct_change, 6),
            "actual_direction": actual_direction,
            "direction_correct": direction_correct,
            "outcome": outcome,
            "brier_score": round(bs, 6),
        }

    @property
    def source(self) -> str:
        return self._source


# ---------------------------------------------------------------------------
# Signal Loader
# ---------------------------------------------------------------------------

def load_signals_from_log(log_path: str) -> List[Dict]:
    """Load crypto signals from b1e55ed-format signal log."""
    with open(log_path) as f:
        data = json.load(f)

    signals = []
    for run in data.get("runs", []):
        run_ts = run.get("timestamp", "")
        regime = run.get("regime", "UNKNOWN")

        for entry in run.get("signals_sent", []):
            sig = entry.get("signal", {})
            symbol = sig.get("symbol", "")

            if symbol not in CRYPTO_SYMBOLS:
                continue

            response = {}
            resp_str = entry.get("response", "{}")
            if isinstance(resp_str, str):
                try:
                    response = json.loads(resp_str)
                except (json.JSONDecodeError, TypeError):
                    pass
            elif isinstance(resp_str, dict):
                response = resp_str

            signals.append({
                "signal_id": response.get("signal_id", sig.get("signal_client_id", "")),
                "symbol": symbol,
                "direction": sig.get("direction", ""),
                "confidence": sig.get("confidence", 0.5),
                "horizon_hours": sig.get("horizon_hours", 24),
                "timestamp": run_ts,
                "regime": regime,
                "method": sig.get("_method", "arbitrary_beta") if "_method" in sig else "arbitrary_beta",
                "http_status": entry.get("http_status", 0),
            })

    return signals


def load_signals_from_array(signals_data: List[Dict]) -> List[Dict]:
    """Load signals from a simple array (schema-compliant format)."""
    result = []
    for sig in signals_data:
        result.append({
            "signal_id": sig.get("signal_id", ""),
            "symbol": sig.get("symbol", ""),
            "direction": sig.get("direction", ""),
            "confidence": sig.get("confidence", 0.5),
            "horizon_hours": sig.get("horizon_hours", 24),
            "timestamp": sig.get("timestamp", ""),
            "regime": sig.get("regime", "UNKNOWN"),
            "method": sig.get("method", "unknown"),
        })
    return result


# ---------------------------------------------------------------------------
# Proof Maintainer
# ---------------------------------------------------------------------------

class ProofMaintainer:
    """
    Maintains a live, rolling proof surface for a Post Fiat signal producer.

    Resolves unresolved signals, computes rolling metrics per window,
    detects calibration drift via CUSUM, and manages atomic versioned snapshots.
    """

    def __init__(self,
                 producer_id: str = "unknown",
                 wallet: str = "",
                 staleness_hours: float = DEFAULT_STALENESS_HOURS,
                 cusum_allowance: float = DEFAULT_CUSUM_ALLOWANCE,
                 cusum_threshold: float = DEFAULT_CUSUM_THRESHOLD,
                 window_days: Optional[List[int]] = None):
        self.producer_id = producer_id
        self.wallet = wallet
        self.staleness_hours = staleness_hours
        self.window_days = window_days or DEFAULT_WINDOW_DAYS
        self.cusum = CUSUMDetector(
            target=0.5,  # updated during maintenance
            allowance=cusum_allowance,
            threshold=cusum_threshold,
        )
        self._all_resolved: List[Dict] = []
        self._previous_snapshot: Optional[Dict] = None

    def load_existing_proof(self, proof_path: str) -> int:
        """Load previously resolved signals from existing proof surface."""
        if not os.path.exists(proof_path):
            return 0
        try:
            with open(proof_path) as f:
                data = json.load(f)
            # Extract resolved signals if stored
            if "resolved_signals" in data:
                self._all_resolved = data["resolved_signals"]
            self._previous_snapshot = data
            return len(self._all_resolved)
        except (json.JSONDecodeError, KeyError):
            return 0

    def maintain(self, signals: List[Dict],
                 price_resolver: PriceResolver,
                 now: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Run one maintenance cycle:
        1. Identify unresolved signals
        2. Resolve them against price data
        3. Compute rolling window metrics
        4. Run CUSUM drift detection
        5. Build proof report

        Idempotent: re-running with the same signals produces the same result.
        Already-resolved signals are skipped.
        """
        if now is None:
            now = datetime.now(timezone.utc)
        now_str = now.isoformat()

        # Find signals needing resolution
        resolved_ids = {s.get("signal_id") or s.get("timestamp", "") + s.get("symbol", "")
                       for s in self._all_resolved}
        unresolved = []
        for sig in signals:
            sig_key = sig.get("signal_id") or sig.get("timestamp", "") + sig.get("symbol", "")
            if sig_key not in resolved_ids:
                unresolved.append(sig)

        # Resolve new signals
        newly_resolved = []
        n_failed = 0
        for sig in unresolved:
            result = price_resolver.resolve_signal(sig)
            if result.get("resolved"):
                newly_resolved.append(result)
            else:
                n_failed += 1

        # Merge with existing
        self._all_resolved.extend(newly_resolved)

        # Sort by timestamp for windowing
        self._all_resolved.sort(key=lambda s: s.get("timestamp", ""))

        # Build resolution delta
        cycle_accuracy = None
        if newly_resolved:
            correct = sum(1 for s in newly_resolved if s.get("direction_correct"))
            cycle_accuracy = {"point": round(correct / len(newly_resolved), 6),
                              "n": len(newly_resolved)}

        delta = {
            "cycle_start": now_str,
            "cycle_end": now_str,
            "n_attempted": len(unresolved),
            "n_resolved": len(newly_resolved),
            "n_failed": n_failed,
            "n_skipped": len(signals) - len(unresolved),
            "resolution_rate": round(len(newly_resolved) / max(1, len(unresolved)), 4),
            "accuracy_this_cycle": cycle_accuracy,
            "price_data_source": price_resolver.source,
        }

        # Compute rolling windows
        windows = self._compute_rolling_windows(now)

        # Run CUSUM drift detection
        all_time = [w for w in windows if w["window_id"] == "all_time"]
        if all_time and all_time[0]["n_resolved"] > 0:
            self.cusum.target = all_time[0]["accuracy"]["point"]

        for sig in newly_resolved:
            outcome = 1.0 if sig.get("direction_correct") else 0.0
            self.cusum.update(outcome, sig.get("timestamp"))

        # Build calibration trajectory
        trajectory = self._compute_trajectory()

        # Build freshness metadata
        freshness = self._compute_freshness(now, signals, len(unresolved) - len(newly_resolved) + n_failed)

        # Build snapshot metadata
        snapshot = self._build_snapshot(now, newly_resolved, n_failed + (len(unresolved) - len(newly_resolved) - n_failed))

        # Compute reputation
        reputation = self._compute_reputation(windows)

        # Detect limitations
        limitations = self._detect_limitations(windows, freshness, delta)

        report = {
            "meta": {
                "producer_id": self.producer_id,
                "wallet": self.wallet,
                "generated_at": now_str,
                "protocol_version": PROTOCOL_VERSION,
                "signal_schema_version": SIGNAL_SCHEMA_VERSION,
                "resolution_protocol_version": RESOLUTION_PROTOCOL_VERSION,
                "proof_surface_version": PROTOCOL_VERSION,
            },
            "freshness": freshness,
            "snapshot": snapshot,
            "resolution_delta": delta,
            "rolling_windows": windows,
            "drift": self.cusum.to_dict(),
            "calibration_trajectory": trajectory,
            "reputation": reputation,
            "protocol_version": PROTOCOL_VERSION,
            "limitations": limitations,
        }

        return report

    def _compute_rolling_windows(self, now: datetime) -> List[Dict]:
        """Compute metrics for each rolling window."""
        windows = []
        for days in self.window_days:
            wid = WINDOW_IDS.get(days, f"{days}d")
            if days == 0:
                # All time
                window_signals = [s for s in self._all_resolved if s.get("resolved")]
                start_dt = None
                if window_signals:
                    try:
                        start_dt = datetime.fromisoformat(
                            window_signals[0]["timestamp"].replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        pass
            else:
                cutoff = now - timedelta(days=days)
                window_signals = []
                start_dt = cutoff
                for s in self._all_resolved:
                    if not s.get("resolved"):
                        continue
                    try:
                        ts = datetime.fromisoformat(s["timestamp"].replace("Z", "+00:00"))
                        if ts >= cutoff:
                            window_signals.append(s)
                    except (ValueError, TypeError):
                        continue

            n_total = len([s for s in self._all_resolved
                          if self._in_window(s, now, days)])
            n_resolved = len(window_signals)

            if n_resolved == 0:
                windows.append({
                    "window_id": wid,
                    "window_days": days,
                    "start_date": start_dt.isoformat() if start_dt else now.isoformat(),
                    "end_date": now.isoformat(),
                    "n_signals": n_total,
                    "n_resolved": 0,
                    "resolution_rate": 0.0,
                    "accuracy": wilson_ci(0, 0),
                    "brier_score": 0.25,
                    "reliability": 0.0,
                    "resolution": 0.0,
                    "calibration_slope": 0.0,
                    "per_symbol": {},
                })
                continue

            forecasts = [s["confidence"] for s in window_signals]
            outcomes = [s["outcome"] for s in window_signals]
            correct = sum(1 for s in window_signals if s.get("direction_correct"))

            brier = brier_decomposition(forecasts, outcomes)
            cal_slope = calibration_slope(forecasts, outcomes)

            # Per-symbol breakdown
            per_sym: Dict[str, Dict] = {}
            by_sym: Dict[str, List[Dict]] = defaultdict(list)
            for s in window_signals:
                by_sym[s["symbol"]].append(s)
            for sym, sigs in by_sym.items():
                sym_correct = sum(1 for s in sigs if s.get("direction_correct"))
                sym_bs = brier_score(
                    [s["confidence"] for s in sigs],
                    [s["outcome"] for s in sigs]
                )
                per_sym[sym] = {
                    "accuracy": round(sym_correct / len(sigs), 6) if sigs else 0.0,
                    "n": len(sigs),
                    "brier_score": round(sym_bs, 6),
                }

            windows.append({
                "window_id": wid,
                "window_days": days,
                "start_date": start_dt.isoformat() if start_dt else now.isoformat(),
                "end_date": now.isoformat(),
                "n_signals": max(n_total, n_resolved),
                "n_resolved": n_resolved,
                "resolution_rate": round(n_resolved / max(1, max(n_total, n_resolved)), 4),
                "accuracy": wilson_ci(correct, n_resolved),
                "brier_score": brier["brier_score"],
                "reliability": brier["reliability"],
                "resolution": brier["resolution"],
                "calibration_slope": cal_slope,
                "per_symbol": per_sym,
            })

        return windows

    def _in_window(self, signal: Dict, now: datetime, days: int) -> bool:
        """Check if a signal falls within a time window."""
        if days == 0:
            return True
        try:
            ts = datetime.fromisoformat(signal["timestamp"].replace("Z", "+00:00"))
            return ts >= now - timedelta(days=days)
        except (ValueError, TypeError, KeyError):
            return False

    def _compute_trajectory(self) -> Dict[str, Any]:
        """Compute calibration trajectory with checkpoints."""
        resolved = [s for s in self._all_resolved if s.get("resolved")]
        if not resolved:
            return {"checkpoints": [], "trend": None}

        checkpoints = []
        for i in range(0, len(resolved), CHECKPOINT_INTERVAL):
            batch = resolved[:i + CHECKPOINT_INTERVAL]
            if not batch:
                continue
            correct = sum(1 for s in batch if s.get("direction_correct"))
            forecasts = [s["confidence"] for s in batch]
            outcomes = [s["outcome"] for s in batch]
            bs = brier_score(forecasts, outcomes)

            checkpoints.append({
                "timestamp": batch[-1].get("timestamp", ""),
                "n_cumulative": len(batch),
                "accuracy": round(correct / len(batch), 6),
                "brier_score": round(bs, 6),
                "reliability": brier_decomposition(forecasts, outcomes)["reliability"],
                "calibration_slope": calibration_slope(forecasts, outcomes),
            })

        # Always add final checkpoint if not at exact interval
        if len(resolved) % CHECKPOINT_INTERVAL != 0 and len(resolved) >= CHECKPOINT_INTERVAL:
            correct = sum(1 for s in resolved if s.get("direction_correct"))
            forecasts = [s["confidence"] for s in resolved]
            outcomes = [s["outcome"] for s in resolved]
            bs = brier_score(forecasts, outcomes)
            checkpoints.append({
                "timestamp": resolved[-1].get("timestamp", ""),
                "n_cumulative": len(resolved),
                "accuracy": round(correct / len(resolved), 6),
                "brier_score": round(bs, 6),
                "reliability": brier_decomposition(forecasts, outcomes)["reliability"],
                "calibration_slope": calibration_slope(forecasts, outcomes),
            })

        # Trend analysis
        trend = None
        if len(checkpoints) >= 3:
            accs = [c["accuracy"] for c in checkpoints]
            xs = list(range(len(accs)))
            n = len(xs)
            mean_x = sum(xs) / n
            mean_y = sum(accs) / n
            cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, accs)) / n
            var_x = sum((x - mean_x) ** 2 for x in xs) / n
            if var_x > EPS:
                slope = cov / var_x
                # Approximate days per checkpoint
                days_per_cp = CHECKPOINT_INTERVAL * (24 / 96)  # ~96 signals/day at 15min
                slope_per_day = slope / max(days_per_cp, 1)

                # Simple p-value approximation
                residuals = [y - (mean_y + slope * (x - mean_x)) for x, y in zip(xs, accs)]
                sse = sum(r ** 2 for r in residuals)
                se = math.sqrt(sse / max(1, n - 2)) / math.sqrt(var_x * n)
                t_stat = slope / max(se, EPS)
                p_approx = min(1.0, 2.0 * math.exp(-0.5 * t_stat ** 2))

                if abs(slope_per_day) < 0.001:
                    interp = "Stable: no significant accuracy trend detected."
                elif slope_per_day > 0:
                    interp = f"Improving: +{slope_per_day:.4f} accuracy per day."
                else:
                    interp = f"Degrading: {slope_per_day:.4f} accuracy per day."

                trend = {
                    "slope_per_day": round(slope_per_day, 6),
                    "p_value": round(p_approx, 6),
                    "interpretation": interp,
                }

        return {"checkpoints": checkpoints, "trend": trend}

    def _compute_freshness(self, now: datetime, all_signals: List[Dict],
                           remaining_unresolved: int) -> Dict[str, Any]:
        """Compute freshness metadata."""
        last_resolved_at = now.isoformat()
        if self._all_resolved:
            resolved_sorted = sorted(
                [s for s in self._all_resolved if s.get("resolved")],
                key=lambda s: s.get("timestamp", "")
            )
            if resolved_sorted:
                last_resolved_at = resolved_sorted[-1]["timestamp"]

        age_hours = 0.0
        try:
            last_ts = datetime.fromisoformat(last_resolved_at.replace("Z", "+00:00"))
            age_hours = (now - last_ts).total_seconds() / 3600
        except (ValueError, TypeError):
            pass

        # Freshness grade
        ratio = age_hours / max(self.staleness_hours, 1)
        if ratio <= 1.0:
            grade = "LIVE"
        elif ratio <= 2.0:
            grade = "RECENT"
        elif ratio <= 4.0:
            grade = "STALE"
        else:
            grade = "EXPIRED"

        next_due = now + timedelta(hours=self.staleness_hours)

        return {
            "last_resolved_at": last_resolved_at,
            "last_update_at": now.isoformat(),
            "staleness_threshold_hours": self.staleness_hours,
            "freshness_grade": grade,
            "age_hours": round(age_hours, 2),
            "signals_since_last_update": remaining_unresolved,
            "next_update_due": next_due.isoformat(),
        }

    def _build_snapshot(self, now: datetime,
                        newly_resolved: List[Dict],
                        remaining: int) -> Dict[str, Any]:
        """Build snapshot metadata for atomic versioned update."""
        version = 1
        parent_id = None
        prev_snap = (self._previous_snapshot or {}).get("snapshot", self._previous_snapshot)
        if prev_snap:
            version = prev_snap.get("snapshot_version", 0) + 1
            parent_id = prev_snap.get("snapshot_id")

        snapshot_id = now.strftime("%Y%m%dT%H%M%SZ")
        return {
            "snapshot_id": snapshot_id,
            "snapshot_version": version,
            "created_at": now.isoformat(),
            "parent_snapshot_id": parent_id,
            "n_resolved_total": len([s for s in self._all_resolved if s.get("resolved")]),
            "n_new_resolved": len(newly_resolved),
            "n_unresolved_remaining": max(0, remaining),
        }

    def _compute_reputation(self, windows: List[Dict]) -> Dict[str, Any]:
        """Compute current reputation score from all-time window."""
        all_time = [w for w in windows if w["window_id"] == "all_time"]
        if not all_time or all_time[0]["n_resolved"] == 0:
            return {"score": 0.0, "grade": "F", "previous_score": None, "score_delta": None}

        w = all_time[0]
        acc = w["accuracy"]["point"]
        rel = w["reliability"]
        cal = w["calibration_slope"]

        # Simplified reputation: weighted composite of accuracy, calibration, reliability
        # Using pf-resolution-protocol weights: calibration 0.20, accuracy 0.35, reliability 0.20
        cal_score = max(0, 1.0 - abs(cal - 1.0))
        rel_score = max(0, 1.0 - min(rel * 100, 1.0))
        acc_score = acc

        score = round(0.35 * acc_score + 0.20 * cal_score + 0.20 * rel_score + 0.25 * 0.5, 4)
        score = max(0.0, min(1.0, score))

        if score >= 0.80:
            grade = "A"
        elif score >= 0.65:
            grade = "B"
        elif score >= 0.50:
            grade = "C"
        elif score >= 0.35:
            grade = "D"
        else:
            grade = "F"

        prev = None
        delta = None
        if self._previous_snapshot and "reputation" in (self._previous_snapshot or {}):
            prev = self._previous_snapshot.get("reputation", {}).get("score")
            if prev is not None:
                delta = round(score - prev, 4)

        return {"score": score, "grade": grade, "previous_score": prev, "score_delta": delta}

    def _detect_limitations(self, windows: List[Dict],
                            freshness: Dict, delta: Dict) -> List[Dict]:
        """Auto-detect structural limitations."""
        limitations = []

        # Stale proof
        if freshness["freshness_grade"] in ("STALE", "EXPIRED"):
            limitations.append({
                "id": "STALE_PROOF",
                "description": f"Proof surface is {freshness['freshness_grade']}. "
                               f"Age: {freshness['age_hours']:.1f}h, threshold: {freshness['staleness_threshold_hours']}h.",
                "bias_direction": "UNKNOWN",
                "bias_magnitude": "Current metrics may not reflect recent producer behavior.",
            })

        # CUSUM drift
        drift = self.cusum.to_dict()
        if drift["drift_status"] in ("DRIFTING", "DEGRADED"):
            limitations.append({
                "id": "CALIBRATION_DRIFT",
                "description": f"CUSUM detected {drift['drift_direction']} calibration drift. "
                               f"Status: {drift['drift_status']}, run length: {drift['run_length']}.",
                "bias_direction": drift["drift_direction"].upper(),
                "bias_magnitude": f"CUSUM positive: {drift['cusum_pos']:.3f}, "
                                  f"negative: {drift['cusum_neg']:.3f} vs threshold {drift['threshold']}.",
            })

        # Small sample
        all_time = [w for w in windows if w["window_id"] == "all_time"]
        if all_time and all_time[0]["n_resolved"] < 100:
            limitations.append({
                "id": "SMALL_SAMPLE",
                "description": f"Only {all_time[0]['n_resolved']} resolved signals. "
                               f"Minimum 100 recommended for stable metrics.",
                "bias_direction": "INDETERMINATE",
                "bias_magnitude": "Wide confidence intervals, metrics may be unstable.",
            })

        # Price data gaps
        if delta["n_failed"] > 0 and delta["n_attempted"] > 0:
            fail_rate = delta["n_failed"] / delta["n_attempted"]
            if fail_rate > 0.1:
                limitations.append({
                    "id": "PRICE_DATA_GAPS",
                    "description": f"{delta['n_failed']}/{delta['n_attempted']} signals "
                                   f"could not be resolved ({fail_rate:.0%}) due to missing price data.",
                    "bias_direction": "INDETERMINATE",
                    "bias_magnitude": "Unresolved signals may have different accuracy distribution.",
                })

        # Window divergence (7d vs all-time accuracy differs significantly)
        w7d = [w for w in windows if w["window_id"] == "7d"]
        if all_time and w7d and all_time[0]["n_resolved"] > 0 and w7d[0]["n_resolved"] >= 20:
            diff = abs(w7d[0]["accuracy"]["point"] - all_time[0]["accuracy"]["point"])
            if diff > 0.05:
                direction = "IMPROVING" if w7d[0]["accuracy"]["point"] > all_time[0]["accuracy"]["point"] else "DEGRADING"
                limitations.append({
                    "id": "WINDOW_DIVERGENCE",
                    "description": f"7d accuracy ({w7d[0]['accuracy']['point']:.3f}) diverges from "
                                   f"all-time ({all_time[0]['accuracy']['point']:.3f}) by {diff:.3f}.",
                    "bias_direction": direction,
                    "bias_magnitude": f"Recent performance is {direction.lower()} relative to historical baseline.",
                })

        return limitations

    def get_resolved_signals(self) -> List[Dict]:
        """Return all resolved signals for persistence."""
        return self._all_resolved


# ---------------------------------------------------------------------------
# Atomic File Update
# ---------------------------------------------------------------------------

def atomic_write(path: str, data: Dict, include_signals: bool = True,
                 resolved_signals: Optional[List[Dict]] = None):
    """
    Write proof surface atomically using temp file + rename.
    Prevents corruption from interrupted writes (e.g. cron killed mid-write).
    """
    output = copy.deepcopy(data)
    if include_signals and resolved_signals is not None:
        output["resolved_signals"] = resolved_signals

    dir_name = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(output, f, indent=2, default=str)
        shutil.move(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Schema Validation
# ---------------------------------------------------------------------------

def validate_report(report: Dict) -> List[str]:
    """Validate a proof report against proof_protocol.json. Returns list of errors."""
    if not HAS_JSONSCHEMA:
        return ["jsonschema not installed"]
    if not os.path.exists(SCHEMA_PATH):
        return [f"Schema not found: {SCHEMA_PATH}"]

    with open(SCHEMA_PATH) as f:
        schema = json.load(f)

    validator = Draft202012Validator(schema)
    errors = []
    for error in sorted(validator.iter_errors(report), key=lambda e: list(e.path)):
        path = ".".join(str(p) for p in error.path) or "(root)"
        errors.append(f"{path}: {error.message}")
    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Continuous Proof Maintenance Protocol — maintain a live rolling proof surface."
    )
    parser.add_argument("signals", help="Path to signal log JSON (b1e55ed format or signal array)")
    parser.add_argument("--proof-surface", "-p", default=None,
                        help="Path to existing proof_surface.json to update")
    parser.add_argument("--price-csv", default=None,
                        help="CSV price data fallback (columns: timestamp,symbol,close_price)")
    parser.add_argument("--producer-id", default="post-fiat-signals",
                        help="Producer identifier")
    parser.add_argument("--wallet", default="",
                        help="Producer wallet address")
    parser.add_argument("--staleness", type=float, default=DEFAULT_STALENESS_HOURS,
                        help=f"Staleness threshold in hours (default: {DEFAULT_STALENESS_HOURS})")
    parser.add_argument("--cusum-allowance", type=float, default=DEFAULT_CUSUM_ALLOWANCE,
                        help=f"CUSUM allowance K (default: {DEFAULT_CUSUM_ALLOWANCE})")
    parser.add_argument("--cusum-threshold", type=float, default=DEFAULT_CUSUM_THRESHOLD,
                        help=f"CUSUM threshold H (default: {DEFAULT_CUSUM_THRESHOLD})")
    parser.add_argument("--fetch-start", default=None,
                        help="Yahoo Finance fetch start date (YYYY-MM-DD)")
    parser.add_argument("--fetch-end", default=None,
                        help="Yahoo Finance fetch end date (YYYY-MM-DD)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output report to file")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON to stdout")
    parser.add_argument("--cron", action="store_true",
                        help="Cron mode: silent unless drift detected")
    parser.add_argument("--validate", action="store_true",
                        help="Validate output against proof_protocol.json")
    parser.add_argument("--no-persist", action="store_true",
                        help="Do not persist resolved signals in output")

    args = parser.parse_args()

    # Load signals
    with open(args.signals) as f:
        raw = json.load(f)

    if isinstance(raw, list):
        signals = load_signals_from_array(raw)
    elif isinstance(raw, dict) and "runs" in raw:
        signals = load_signals_from_log(args.signals)
    else:
        print("Error: unrecognized signal log format", file=sys.stderr)
        sys.exit(1)

    # Set up price resolver
    resolver = PriceResolver(csv_path=args.price_csv)

    # If using Yahoo Finance, pre-fetch data
    if not args.price_csv and HAS_SCIENCE:
        start = args.fetch_start
        end = args.fetch_end
        if not start and signals:
            timestamps = [s["timestamp"] for s in signals if s.get("timestamp")]
            if timestamps:
                earliest = min(timestamps)[:10]
                start = earliest
        if not end:
            end = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%d")
        if start:
            for symbol in CRYPTO_SYMBOLS:
                try:
                    resolver._fetch_yf(symbol, start, end)
                except Exception as e:
                    print(f"Warning: failed to fetch {symbol}: {e}", file=sys.stderr)

    # Set up maintainer
    maintainer = ProofMaintainer(
        producer_id=args.producer_id,
        wallet=args.wallet,
        staleness_hours=args.staleness,
        cusum_allowance=args.cusum_allowance,
        cusum_threshold=args.cusum_threshold,
    )

    # Load existing proof surface
    if args.proof_surface:
        maintainer.load_existing_proof(args.proof_surface)

    # Run maintenance cycle
    report = maintainer.maintain(signals, resolver)

    # Validate
    if args.validate:
        errors = validate_report(report)
        if errors:
            print(f"VALIDATION FAILED ({len(errors)} errors):", file=sys.stderr)
            for e in errors[:20]:
                print(f"  {e}", file=sys.stderr)
            sys.exit(1)
        if not args.cron:
            print("VALIDATION: PASS", file=sys.stderr)

    # Output
    if args.output:
        atomic_write(args.output, report,
                     include_signals=not args.no_persist,
                     resolved_signals=maintainer.get_resolved_signals())
        if not args.cron:
            snap = report["snapshot"]
            fresh = report["freshness"]
            drift = report["drift"]
            print(f"Proof updated: v{snap['snapshot_version']}, "
                  f"{snap['n_resolved_total']} resolved (+{snap['n_new_resolved']}), "
                  f"freshness={fresh['freshness_grade']}, "
                  f"drift={drift['drift_status']}")

    if args.json:
        # Strip resolved_signals from JSON output (too large)
        output = {k: v for k, v in report.items() if k != "resolved_signals"}
        print(json.dumps(output, indent=2, default=str))

    if args.cron:
        drift = report["drift"]
        fresh = report["freshness"]
        if drift["drift_status"] in ("DRIFTING", "DEGRADED"):
            print(f"DRIFT ALERT: {drift['drift_status']} "
                  f"({drift['drift_direction']}) "
                  f"CUSUM+={drift['cusum_pos']:.3f} CUSUM-={drift['cusum_neg']:.3f}")
        if fresh["freshness_grade"] in ("STALE", "EXPIRED"):
            print(f"FRESHNESS ALERT: {fresh['freshness_grade']} "
                  f"(age: {fresh['age_hours']:.1f}h)")

    if not args.json and not args.output and not args.cron:
        # Default: print summary
        snap = report["snapshot"]
        fresh = report["freshness"]
        drift = report["drift"]
        delta = report["resolution_delta"]
        print(f"Proof Maintenance Report")
        print(f"  Producer: {report['meta']['producer_id']}")
        print(f"  Snapshot: v{snap['snapshot_version']}")
        print(f"  Resolved: {snap['n_resolved_total']} total (+{snap['n_new_resolved']} this cycle)")
        print(f"  Unresolved: {snap['n_unresolved_remaining']}")
        print(f"  Freshness: {fresh['freshness_grade']} (age: {fresh['age_hours']:.1f}h)")
        print(f"  Drift: {drift['drift_status']}")
        print(f"  Windows:")
        for w in report["rolling_windows"]:
            print(f"    {w['window_id']}: accuracy={w['accuracy']['point']:.4f} "
                  f"brier={w['brier_score']:.4f} n={w['n_resolved']}")
        rep = report.get("reputation", {})
        if rep:
            print(f"  Reputation: {rep.get('score', 0):.4f} ({rep.get('grade', 'F')})")


if __name__ == "__main__":
    main()
