# Changelog

## [1.0.0] - 2026-03-29

### Added
- `proof_protocol.json` — JSON Schema (draft 2020-12) defining continuous proof maintenance report format
  - 10 schema definitions ($defs): semver, iso_timestamp, freshness_grade, drift_status, rolling_window, cusum_state, calibration_trajectory, freshness_metadata, snapshot_metadata, resolution_delta, proof_report
- `maintain_proof.py` — Standalone CLI + library for continuous proof maintenance
  - PriceResolver: Yahoo Finance + CSV fallback price resolution
  - CUSUMDetector: Two-sided CUSUM calibration drift detection (STABLE/WATCH/DRIFTING/DEGRADED)
  - ProofMaintainer: Rolling window metrics, Brier decomposition, Wilson CIs, atomic versioned snapshots
  - Freshness grading (LIVE/RECENT/STALE/EXPIRED)
  - Auto-detected limitations with bias direction and magnitude
  - Cron-deployable with idempotent resolution and crash-safe atomic writes
- `example_proof/` — Worked example with 20 synthetic signals, CSV prices, and generated report
- `tests/test_maintain_proof.py` — 87 tests across 22 test classes
