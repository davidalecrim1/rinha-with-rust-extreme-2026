# rinha-with-rust-2026

Rust submission for rinha-de-backend-2026 — fraud detection via vector search.

## Why Rust

The scoring formula rewards p99 ≤ 1ms with maximum latency points. Go's GC introduces non-deterministic tail latency that is tunable but not eliminable. Rust has zero GC — latency is fully deterministic. On a slow test machine (Mac Mini 2014, 2.6 GHz), this difference is real under load.

See `docs/rust-vs-go.md` for the full trade-off analysis. Build and validate correctness with the Go submission first, then use Rust to push p99 lower.

## Challenge summary

Build a fraud detection API that:
1. Receives a card transaction payload
2. Vectorizes it into 14 f32 dimensions (normalization rules in `rinha-de-backend-2026/docs/en/DETECTION_RULES.md`)
3. Finds the 5 nearest neighbors in a 100K-vector reference dataset using Euclidean distance
4. Returns `approved = fraud_score < 0.6` where `fraud_score = frauds_among_5 / 5`

Full spec: `rinha-de-backend-2026/docs/en/`

## API contract

- `GET /ready` — return 503 until HNSW index is built, then 200
- `POST /fraud-score` — receive transaction, return `{ "approved": bool, "fraud_score": float }`
- Internal port: 8080 (nginx on 9999 forwards here)

## Architecture

```
nginx (0.05 CPU / 15MB)  ← listens on :9999, round-robin
  ├── api1 (0.475 CPU / 167MB)  ← listens on :8080
  └── api2 (0.475 CPU / 167MB)  ← listens on :8080
```

## Design decisions (shared with Go submission)

| Decision | Choice | Reason |
|---|---|---|
| Vector type | f32 | Halves memory vs f64, cache-friendly, SIMD-aligned |
| Vector search | Brute-force KNN + SIMD auto-vectorization | 100K × 14 f32 = ~1.4M ops; AVX2 brings this to ~50-70µs, well under 1ms p99 — no ANN approximation error |
| Resource files | Embedded in image via `COPY` | Self-contained, no volume mount dependencies |
| Docker | Multi-stage → `FROM scratch` | Tiny final image, statically linked binary |
| nginx | TBD — template to be provided | `worker_processes 1`, `keepalive 100`, `access_log off` as baseline |

## Rust-specific decisions

| Decision | Choice | Reason |
|---|---|---|
| HTTP framework | `axum` | tokio-native, clean serde integration, tower overhead negligible at this scale |
| KNN search | Brute-force over `Vec<f32>` flat buffer, AVX2 via `RUSTFLAGS=-C target-feature=+avx2` | Compiler auto-vectorizes inner loop; no external crate, no C deps, deterministic recall |
| Top-5 selection | Fixed array of 5 slots, linear eviction scan, O(N) | Avoids full sort; single pass over distances |
| Async runtime | `tokio`, `worker_threads = 1` | Matches 0.475 CPU quota; eliminates thread contention, same reasoning as Go's `GOMAXPROCS=1` |
| JSON | `serde_json` | Payload is ~200 bytes — simd-json gains (~1-2µs) don't justify axum extractor incompatibility |
| Cross-compilation | Docker multi-stage builder | Build inside `FROM rust:alpine`; no host toolchain needed, matches competition infra |

## Module contract

Business logic must not leak into handlers. Each module has a single responsibility:

- **`main.rs`**: wires modules, builds the HNSW index at startup, sets the readiness flag
- **`handler.rs`**: deserialize → call `vectorizer::vectorize()` → call `index::search()` → serialize. No scoring logic.
- **`vectorizer.rs`**: transaction payload → `[f32; 14]`. Pure data transformation.
- **`index.rs`**: owns the flat `Vec<f32>` reference buffer and label list. Exposes only `fn search(vector: [f32; 14]) -> f32` returning `fraud_score`. The `approved` decision (`score < 0.6`) lives here. Swap the search strategy by touching only this file.

## Project structure

```
rinha-with-rust-2026/
├── src/
│   ├── main.rs          # startup, runtime config, readiness flag
│   ├── handler.rs       # axum handlers for /ready and /fraud-score
│   ├── vectorizer.rs    # 14-dim normalization
│   └── index.rs         # instant-distance wrapper, exposes search(vector) -> f32
├── resources/
│   ├── references.json.gz
│   ├── mcc_risk.json
│   └── normalization.json
├── Dockerfile
├── Cargo.toml
└── CLAUDE.md
```

## Vectorization notes

Same rules as Go submission:
- `minutes_since_last_tx`: delta between `requested_at` and `last_transaction.timestamp` in minutes, clamped. -1 sentinel only when `last_transaction` is null.
- `unknown_merchant`: 1 if `merchant.id` not in `customer.known_merchants`, else 0.
- `mcc_risk`: look up `merchant.mcc`, default 0.5 if not found.
- All values clamped to [0.0, 1.0] except indices 5 and 6 (-1 sentinel).

## Resource files

Copy from `rinha-de-backend-2026/resources/` into `resources/`:

- `references.json.gz` — 100K labeled vectors (fraud/legit), ~16MB gzipped
- `mcc_risk.json` — MCC code → risk score mapping
- `normalization.json` — constants for the 14-dimension normalization formulas

## Load test dataset

`scripts/test-data.json` is gitignored (22 MB). Run `make fetch-test-data` after a fresh clone to copy it from the `rinha-de-backend-2026` submodule.

## Submission structure

Two branches required:
- `main` — source code
- `submission` — only `docker-compose.yml`, `nginx.conf`, `info.json`

Docker images must be public and compatible with `linux/amd64`.

## Scoring

- `score_p99`: logarithmic, +1000 per 10x improvement. Ceiling at ≤1ms (+3000), floor at >2000ms (-3000).
- `score_det`: FP weight 1, FN weight 3, HTTP error weight 5. Cutoff at >15% failure rate → -3000.
- `final_score = score_p99 + score_det`, range [-6000, +6000].

HTTP 500s are the worst outcome — weight 5 and count toward the failure rate cutoff.

## Rust practices

**Panics**: `.expect()` / `.unwrap()` are acceptable only at startup on embedded data (e.g., `include_bytes!` resources parsed in `main`). Any code reachable from a live request must not panic — return an error type instead. A panic on a malformed request means connection reset for the client and a weight-5 penalty in scoring.

**Parse at the boundary**: validate and parse external input in the serde types, not in business logic. Use `DateTime<FixedOffset>` rather than `String` for timestamps so that invalid dates are rejected at deserialization with a clean 422, before reaching `vectorize`.

**Safe startup panics must have `.expect("message")`** — never bare `.unwrap()` on embedded resource parsing, so crash messages are actionable.
