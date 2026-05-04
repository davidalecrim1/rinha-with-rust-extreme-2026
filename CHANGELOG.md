# Changelog

## v0.8.4
- Remove the brute-force production path and keep IVF as the only search mode
- Drop the uncapped overlay, add the compose helpers to the release flow, and simplify the build path by removing optional IVF feature toggles
- Capped official load test: final score 4416.92, p99 25.26ms, 0 HTTP errors, 3 false positives, 0 false negatives

## v0.8.3
- Add decision-aware IVF refinement: scan `IVF_PRIMARY_NPROBE=4`, skip repair for clear top-5 votes, and refine ambiguous votes with `IVF_REFINE_NPROBE=24`
- Bound refinement by both clusters and rows with `IVF_REPAIR_MAX_EXTRA_CLUSTERS=96` and `IVF_REPAIR_MAX_EXTRA_ROWS=180000`; rebalance CPU to nginx `0.10` and each API `0.45`
- Local official load test: final score 3069.62, p99 3.35ms, 0 HTTP errors, 366 false positives, 386 false negatives

## v0.8.2
- Bound IVF bbox repair with `IVF_REPAIR_MAX_EXTRA_CLUSTERS=32` and lower `NPROBE` to `4` to avoid official-limit backpressure
- Add explicit Haswell SIMD target features and strip symbols in the release Docker build
- Capped local Rinha load test: final score 2099.67, p99 31.36ms, 0 HTTP errors, 366 false positives, 385 false negatives

## v0.8.1
- Add IVF bounding-box repair pass to revisit only unscanned clusters whose 14-dim quantized bounds can still beat the current top-5 worst distance
- Official local Rinha load test: final score 5383.76, p99 3.36ms, zero HTTP errors
- Detection result: 1 false positive, 0 false negatives, weighted error 1 over 54,100 requests

## v0.7.1
- IVF (Inverted File Index) vector search: k-means clustering at build time, scan only nearest 100 clusters at query time
- ~10x fewer rows scanned per query (293K vs 3M), targeting sub-3ms p99 on competition hardware
- Initial rollout before IVF became the only production search mode

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
