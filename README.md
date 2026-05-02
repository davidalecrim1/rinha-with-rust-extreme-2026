# rinha-with-rust-extreme-2026

Rust submission for [rinha-de-backend-2026](https://github.com/zanfranceschi/rinha-de-backend-2026) — a fraud detection API competition scored on p99 latency and detection accuracy.

## What it does

Receives a card transaction payload, vectorizes it into 14 f32 dimensions, finds the 5 nearest neighbors in a 3M labeled reference dataset using Euclidean distance, and returns:

```json
{ "approved": true, "fraud_score": 0.2 }
```

`approved = fraud_score < 0.6`, where `fraud_score = frauds_among_5_neighbors / 5`.

## Architecture

```
nginx (0.05 CPU / 15MB)  — listens on :9999, round-robin over UDS
  ├── api1 (0.475 CPU / 167MB)  — listens on /var/run/api1.sock
  └── api2 (0.475 CPU / 167MB)  — listens on /var/run/api2.sock
```

Total resource budget: 1.0 CPU / 350MB.

## Search strategy

IVF (Inverted File Index) with `K=4096` centroids and an AVX2-friendly 8-wide structure-of-arrays block layout.

- 3M reference rows; fast path scans the nearest 16 clusters, fallback scans 24 clusters for boundary scores
- Centroid selection: distance to all 4096 centroids, then fixed top-N selection
- Block scan: each block stores 8 vectors as `14 dims x 8 i16` so AVX2 evaluates 8 candidates at once
- Probe counts are tunable with `FAST_NPROBE` and `FULL_NPROBE`

A fixed array of 5 slots tracks the nearest neighbors in a single O(N) pass per cluster — on each candidate, a linear scan of the 5 slots evicts the farthest.

Latest local results:

- `scripts/results/2026-05-02T18-28-19_v0.8.0_200vus_60s.json`: p99 89.82ms, p95 66.43ms, 0% error, 1239.86 RPS
- Official local test (`../rinha-de-backend-2026/test/results.json`): p99 4.43ms, 0% failure, 1 false negative, final score 5172.84

## Key decisions

| Decision | Choice | Reason |
|---|---|---|
| HTTP framework | `axum` | tokio-native, minimal overhead |
| Runtime | `tokio`, `worker_threads = 1` | Matches 0.475 CPU quota; eliminates thread contention |
| Vector storage | Quantized `i16` SoA blocks | Cuts bandwidth and lets AVX2 scan 8 candidates per block |
| Resource files | `include_bytes!` | Embedded at compile time — self-contained binary, no volume mounts |
| Docker | Multi-stage → `FROM scratch` | Minimal final image, statically linked musl binary |
| JSON | `serde_json` | Payload is ~200 bytes — simd-json gains don't justify extractor incompatibility |

## Module layout

```
src/
  main.rs        startup, resource loading, axum wiring
  types.rs       all request/response structs and NormConsts
  vectorizer.rs  14-dim normalization (pure function)
  index.rs       IVF + AVX2 block KNN, exposes search(vector) -> f32
  handler.rs     axum handlers for /ready and /fraud-score
```

## Development

```bash
cargo test                     # run unit tests
make run                       # start local stack via docker-compose.local.yml
bash scripts/check-health.sh   # smoke test with two known payloads
bash scripts/run-example-payloads.sh  # validate all 50 example payloads
```

## Submission branches

- `main` — source code (this branch)
- `submission` — only `docker-compose.yml`, `nginx.conf`, `info.json`
