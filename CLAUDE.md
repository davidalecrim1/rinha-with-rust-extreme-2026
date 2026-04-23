# rinha-with-rust-2026

Rust submission for rinha-de-backend-2026 — fraud detection via vector search.

## Why Rust

The scoring formula rewards p99 ≤ 1ms with maximum latency points. Go's GC introduces non-deterministic tail latency that is tunable but not eliminable. Rust has zero GC — latency is fully deterministic. On a slow test machine (Mac Mini 2014, 2.6 GHz), this difference is real under load.

See `docs/rust-vs-go.md` for the full trade-off analysis.

## Challenge summary

Build a fraud detection API that:
1. Receives a card transaction payload
2. Vectorizes it into 14 f32 dimensions (normalization rules in `rinha-de-backend-2026/docs/en/DETECTION_RULES.md`)
3. Finds the 5 nearest neighbors in a 100K-vector reference dataset using Euclidean distance
4. Returns `approved = fraud_score < 0.6` where `fraud_score = frauds_among_5 / 5`

Full spec: `rinha-de-backend-2026/docs/en/`

## API contract

- `GET /ready` — return 200 when HNSW index is built and ready, 503 otherwise
- `POST /fraud-score` — receive transaction, return `{ "approved": bool, "fraud_score": float }`
- Port: 9999 (nginx listens here, forwards to api1/api2)

## Architecture

```
nginx (0.05 CPU / 15MB)
  └── round-robin
        ├── api1 (0.475 CPU / 167MB)
        └── api2 (0.475 CPU / 167MB)
```

## Design decisions (shared with Go submission)

| Decision | Choice | Reason |
|---|---|---|
| Vector type | f32 | Halves memory vs f64, cache-friendly, SIMD-aligned |
| Vector index | HNSW in-memory | O(log N) queries, fits in 167MB easily |
| Resource files | Embedded in image via `COPY` | Self-contained, no volume mount dependencies |
| Index build | At container startup | Spec explicitly supports this |
| Docker | Multi-stage, `FROM scratch` | Tiny final image, statically linked binary |
| nginx | `worker_processes 1`, `keepalive 100`, `access_log off` | Matches CPU quota |

## Rust-specific decisions (TBD)

These need to be resolved during implementation:

- **HTTP framework**: `axum` (tokio-based, ergonomic) vs `actix-web` (high perf) vs `hyper` (bare metal)
- **HNSW library**: `instant-distance` vs `hora` vs hand-rolled
- **Async runtime**: `tokio` (standard) — thread count should respect the 0.475 CPU quota
- **JSON**: `serde_json` (standard) vs `simd-json` (faster parsing)
- **Cross-compilation**: `cross` crate or Docker builder for linux/amd64 from ARM Mac

## Resource files

Copy these from `rinha-de-backend-2026/resources/` into `resources/` at the repo root:

- `references.json.gz` — 100K labeled vectors (fraud/legit), ~1.6MB gzipped
- `mcc_risk.json` — MCC code → risk score mapping
- `normalization.json` — constants for the 14-dimension normalization formulas

## Submission structure

Two branches required:
- `main` — source code
- `submission` — only `docker-compose.yml`, `nginx.conf`, `info.json`

Docker images must be public and compatible with `linux/amd64`.

## Scoring

- `score_p99`: logarithmic, +1000 per 10x improvement in p99. Ceiling at ≤1ms (+3000), floor at >2000ms (-3000).
- `score_det`: based on FP (weight 1), FN (weight 3), HTTP errors (weight 5). Cutoff at >15% failure rate → -3000.
- `final_score = score_p99 + score_det`, range [-6000, +6000].

Key insight: HTTP 500s are the worst outcome (weight 5 + count toward failure rate). If something goes wrong, return `approved: true, fraud_score: 0.0` rather than 500.
