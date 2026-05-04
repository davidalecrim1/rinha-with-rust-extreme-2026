# rinha-with-rust-2026

Rust submission for [rinha-de-backend-2026](https://github.com/zanfranceschi/rinha-de-backend-2026) — a fraud detection API competition scored on p99 latency and detection accuracy.

## What it does

Receives a card transaction payload, vectorizes it into 14 f32 dimensions, finds the 5 nearest neighbors in a 3M labeled reference dataset using Euclidean distance, and returns:

```json
{ "approved": true, "fraud_score": 0.2 }
```

`approved = fraud_score < 0.6`, where `fraud_score = frauds_among_5_neighbors / 5`.

## Architecture

```
nginx (0.05 CPU / 15MB)  — listens on :9999, round-robin
  ├── api1 (0.475 CPU / 167MB)  — listens on :8080
  └── api2 (0.475 CPU / 167MB)  — listens on :8080
```

Total resource budget: 1.0 CPU / 350MB.

## Search strategy

IVF (Inverted File Index) with K=1024 centroids and nprobe=50, built on top of the same quantized row format and SIMD scan kernel used for brute-force.

- 3M reference rows; IVF scans ~146K rows per query (~5% of the dataset)
- Centroid selection: distance to all 1024 centroids, partial sort to pick top nprobe
- Row scan: same 16-byte packed format + AVX2 SIMD distance kernel as brute-force
- nprobe tunable via `NPROBE`; bbox repair tunable via `IVF_REPAIR` and `IVF_REPAIR_MAX_EXTRA_CLUSTERS`

A fixed array of 5 slots tracks the nearest neighbors in a single O(N) pass per cluster — on each candidate, a linear scan of the 5 slots evicts the farthest.

## Key decisions

| Decision | Choice | Reason |
|---|---|---|
| HTTP framework | `axum` | tokio-native, minimal overhead |
| Runtime | `tokio`, `worker_threads = 1` | Matches 0.475 CPU quota; eliminates thread contention |
| Vector type | `f32` | Halves memory vs f64, cache-friendly, SIMD-aligned |
| Resource files | `include_bytes!` | Embedded at compile time — self-contained binary, no volume mounts |
| Docker | Multi-stage → `FROM scratch` | Minimal final image, statically linked musl binary |
| JSON | `serde_json` | Payload is ~200 bytes — simd-json gains don't justify extractor incompatibility |

## Module layout

```
src/
  main.rs        startup, resource loading, axum wiring
  types.rs       all request/response structs and NormConsts
  vectorizer.rs  14-dim normalization (pure function)
  index.rs       brute-force KNN, exposes search(vector) -> f32
  handler.rs     axum handlers for /ready and /fraud-score
```

## Development

```bash
cargo test                     # run unit tests
make build                     # build the latest local Docker image
make run                       # start local stack from the latest built image
bash scripts/check-health.sh   # smoke test with two known payloads
bash scripts/run-example-payloads.sh  # validate all 50 example payloads
```

## Submission branches

- `main` — source code (this branch)
- `submission` — only `docker-compose.yml`, `nginx.conf`, `info.json`
