# Changelog

## v0.1.1
- Offload KNN search to `spawn_blocking` — frees tokio thread for I/O under concurrent load
- Set `worker_processes 1` in nginx — prevents CPU starvation with 0.05 CPU budget
- p99 improved from 291ms → 129ms; score from 3535 → 3888

## v0.1.0
- Brute-force KNN over 100K reference vectors with AVX2/FMA auto-vectorization
- Full fraud detection API: `/ready` healthcheck + `/fraud-score` endpoint
- Static binary via `FROM busybox:musl`, resource files embedded at compile time
