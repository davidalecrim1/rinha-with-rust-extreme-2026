# Changelog

## v0.8.0
- Rework IVF references into 8-wide AVX2 SoA blocks with `K=4096`
- Add two-stage probing (`FAST_NPROBE=16`, `FULL_NPROBE=24`) for ambiguous scores
- Official local load test: p99 4.43ms, 0% failure rate, final score 5172.84

## v0.7.1
- IVF (Inverted File Index) vector search: k-means clustering at build time, scan only nearest 100 clusters at query time
- ~10x fewer rows scanned per query (293K vs 3M), targeting sub-3ms p99 on competition hardware
- Feature-flagged (`--features ivf`): brute-force remains the default fallback

## v0.7.0
- Updated to 3M reference dataset
- Quantize references to i16 and scan with SSE2 SIMD — halves memory bandwidth vs f32
- Replace TCP with Unix domain sockets between nginx and api instances

## v0.1.2
- Add k6 load test script, replace shell-based example-payloads with Python
- Include VUs and duration in load test result filenames

## v0.1.1
- Offload KNN search to `spawn_blocking` — frees tokio thread for I/O under concurrent load
- Set `worker_processes 1` in nginx — prevents CPU starvation with 0.05 CPU budget
- p99 improved from 291ms → 129ms; score from 3535 → 3888

## v0.1.0
- Brute-force KNN over 100K reference vectors with AVX2/FMA auto-vectorization
- Full fraud detection API: `/ready` healthcheck + `/fraud-score` endpoint
- Static binary via `FROM busybox:musl`, resource files embedded at compile time
