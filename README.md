# Post Fiat Continuous Proof Maintenance Protocol

**Fifth companion protocol** in the Post Fiat signal quality stack:

| Layer | Repo | Purpose |
|-------|------|---------|
| 1. Signal Schema | [pf-signal-schema](https://github.com/sendoeth/pf-signal-schema) | Define what a signal looks like |
| 2. Signal Routing | [pf-routing-protocol](https://github.com/sendoeth/pf-routing-protocol) | Route signals to consumers |
| 3. Signal Resolution | [pf-resolution-protocol](https://github.com/sendoeth/pf-resolution-protocol) | Resolve signals against outcomes |
| 4. Signal Aggregation | [pf-aggregation-protocol](https://github.com/sendoeth/pf-aggregation-protocol) | Aggregate multi-producer signals |
| **5. Proof Maintenance** | **this repo** | **Keep proof surfaces live and trustworthy** |

## Problem

A proof surface is only trustworthy if it stays current. Static proof snapshots decay: new signals go unresolved, accuracy metrics become stale, calibration drift goes undetected. Consumers cannot trust a proof artifact that was generated once and never updated.

## Solution

This protocol defines how producers maintain a **live, rolling proof surface** with:

- **Automated signal resolution** against real price data (Yahoo Finance + CSV fallback)
- **Rolling window metrics** (7d / 14d / 30d / all-time) with Wilson score CIs
- **CUSUM calibration drift detection** (STABLE → WATCH → DRIFTING → DEGRADED)
- **Freshness guarantees** (LIVE / RECENT / STALE / EXPIRED grading)
- **Atomic versioned snapshots** with parent chain for audit trail
- **Auto-detected limitations** with bias direction and magnitude
- **Cron-deployable** — run every 15 minutes, idempotent, crash-safe

## Quick Start

```bash
# Resolve signals against CSV price data
python3 maintain_proof.py signals.json --price-csv prices.csv --validate --json

# Resolve signals against Yahoo Finance (requires yfinance)
python3 maintain_proof.py signals.json --validate -o proof_surface.json

# Cron mode: load existing proof, resolve new signals, update atomically
python3 maintain_proof.py signals.json --cron --proof-surface existing_proof.json -o proof_surface.json

# Validate an existing proof report against the schema
python3 maintain_proof.py signals.json --validate
```

### Cron Deployment

```cron
# Every 15 minutes: resolve new signals and update proof surface
*/15 * * * * cd /path/to/repo && python3 maintain_proof.py /path/to/signals.json --cron --proof-surface proof_surface.json -o proof_surface.json --validate >> /var/log/proof_maintenance.log 2>&1
```

## Schema

`proof_protocol.json` — JSON Schema (draft 2020-12) defining the proof maintenance report format.

### Top-Level Structure

```
proof_report
├── meta                    # Producer ID, wallet, protocol versions
├── freshness               # LIVE/RECENT/STALE/EXPIRED grading
├── snapshot                # Versioned snapshot with parent chain
├── resolution_delta        # This cycle's resolution results
├── rolling_windows[]       # 7d, 14d, 30d, all_time metrics
│   ├── accuracy            # Wilson score CI (point, lower, upper, n)
│   ├── brier_score         # Mean Brier score (0=perfect, 0.25=random)
│   ├── reliability         # Murphy-Winkler calibration error (lower=better)
│   ├── resolution          # Murphy-Winkler discrimination (higher=better)
│   ├── calibration_slope   # OLS slope (1.0=perfect calibration)
│   └── per_symbol          # Per-symbol breakdown
├── drift                   # CUSUM calibration drift detector state
├── calibration_trajectory  # Time series of calibration checkpoints
├── reputation              # Producer Reputation Score + grade
└── limitations[]           # Auto-detected with bias direction/magnitude
```

### Schema Definitions

| $def | Description |
|------|-------------|
| `freshness_grade` | LIVE (≤1x threshold), RECENT (≤2x), STALE (≤4x), EXPIRED (>4x) |
| `drift_status` | STABLE, WATCH (≥0.6×H), DRIFTING (≥H), DEGRADED (run_length ≥50) |
| `rolling_window` | Metrics over a time window with per-symbol breakdown |
| `cusum_state` | Two-sided CUSUM chart with target, allowance K, threshold H |
| `calibration_trajectory` | Ordered checkpoints + linear trend (slope_per_day, p-value) |
| `freshness_metadata` | Last resolved/updated timestamps, age, staleness grading |
| `snapshot_metadata` | Monotonic version, parent chain, resolution counts |
| `resolution_delta` | Single-cycle resolution results |
| `proof_report` | Complete report output |

## CUSUM Drift Detection

The protocol uses a two-sided Cumulative Sum (CUSUM) chart to detect systematic shifts in producer accuracy:

```
For each resolved signal:
  deviation = outcome - target_accuracy
  S⁺ = max(0, S⁺ + deviation - K)     # Detects improvement
  S⁻ = max(0, S⁻ - deviation - K)     # Detects degradation
```

**Parameters:**
- **Target (μ₀)**: All-time historical accuracy (auto-calibrated)
- **Allowance (K)**: Slack parameter. Default 0.02 — shifts must exceed K per observation to accumulate
- **Threshold (H)**: Decision boundary. Default 1.5 — triggers drift detection when exceeded

**Status progression:**
- `STABLE` — both S⁺ and S⁻ below 0.6×H
- `WATCH` — max(S⁺, S⁻) ≥ 0.6×H
- `DRIFTING` — max(S⁺, S⁻) ≥ H
- `DEGRADED` — sustained drift (run_length ≥ 50)

## Consumer Verification Guide

Downstream consumers should verify a proof surface by checking:

1. **Freshness**: Is `freshness.freshness_grade` LIVE or RECENT? STALE/EXPIRED means the proof may not reflect current producer behavior
2. **Sample size**: Check `rolling_windows[all_time].n_resolved` — at least 100 recommended for stable metrics
3. **Drift status**: Is `drift.drift_status` STABLE? WATCH is acceptable; DRIFTING/DEGRADED means accuracy is shifting
4. **Limitations**: Read `limitations[]` — each has `bias_direction` and `bias_magnitude` explaining what the limitation means
5. **Schema validation**: Validate the report against `proof_protocol.json` to ensure structural compliance
6. **Snapshot chain**: Verify `snapshot.snapshot_version` is monotonically increasing and `parent_snapshot_id` chains correctly

```python
import json
from maintain_proof import validate_report

with open("proof_surface.json") as f:
    report = json.load(f)

errors = validate_report(report)
if errors:
    print(f"Schema violations: {errors}")
else:
    grade = report["freshness"]["freshness_grade"]
    drift = report["drift"]["drift_status"]
    rep = report["reputation"]["grade"]
    print(f"Freshness: {grade}, Drift: {drift}, Reputation: {rep}")
```

## Auto-Detected Limitations

The protocol automatically detects and reports structural limitations:

| ID | Trigger | Bias Direction |
|----|---------|----------------|
| `STALE_PROOF` | freshness_grade is STALE or EXPIRED | UNKNOWN — metrics may not reflect recent behavior |
| `CALIBRATION_DRIFT` | drift_status is DRIFTING or DEGRADED | Reported by CUSUM direction (improving/degrading) |
| `SMALL_SAMPLE` | n_resolved < 100 | INDETERMINATE — wide confidence intervals |
| `PRICE_DATA_GAPS` | n_failed > 0 in resolution_delta | UPWARD — unresolvable signals may have been harder predictions |
| `WINDOW_DIVERGENCE` | 7d accuracy differs >15% from all-time | Indicates regime change or calibration shift |

## Tests

```bash
python3 -m pytest tests/ -v
# 87 tests across 22 test classes
```

**Test categories:**
- Wilson CI, Brier score, calibration slope (13 tests)
- CUSUM drift detection: no-drift, gradual, sudden, reset, parameters (12 tests)
- Rolling windows and metrics (8 tests)
- Atomic updates and file I/O (7 tests)
- Staleness grading (5 tests)
- Empty inputs and price gaps (6 tests)
- Schema validation and cross-schema compatibility (9 tests)
- Cron idempotency (3 tests)
- Report structure and limitations (7 tests)
- Calibration trajectory (3 tests)
- Signal loading (3 tests)
- End-to-end pipeline (4 tests)
- Reputation scoring (5 tests)
- Multi-symbol handling (2 tests)

## License

MIT
