#!/usr/bin/env python3
import json
import os
import statistics
import sys
import time
import urllib.request
from datetime import datetime

BASE_URL = os.environ.get("BASE_URL", "http://localhost:9999")
PAYLOADS_URL = "https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/example-payloads.json"

print("Fetching example-payloads.json...")
with urllib.request.urlopen(PAYLOADS_URL) as r:
    payloads = json.loads(r.read())
print(f"Loaded {len(payloads)} payloads\n")

passed = 0
failed = 0
approved_count = 0
declined_count = 0
latencies = []
failures = []

for payload in payloads:
    tx_id = payload["id"]
    body = json.dumps(payload).encode()

    req = urllib.request.Request(
        f"{BASE_URL}/fraud-score",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req) as r:
            status = r.status
            response = json.loads(r.read())
    except urllib.error.HTTPError as e:
        status = e.code
        response = None
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    latencies.append(elapsed_ms)

    if status != 200:
        print(f"FAIL [{tx_id}] HTTP {status} ({elapsed_ms}ms)")
        failures.append({"tx_id": tx_id, "reason": f"HTTP {status}"})
        failed += 1
        continue

    try:
        assert isinstance(response.get("approved"), bool), "approved must be bool"
        fs = response.get("fraud_score")
        assert isinstance(fs, (int, float)), "fraud_score must be number"
        assert 0.0 <= fs <= 1.0, f"fraud_score {fs} out of range [0,1]"
    except AssertionError as e:
        print(f"FAIL [{tx_id}] {e} — response: {response} ({elapsed_ms}ms)")
        failures.append({"tx_id": tx_id, "reason": str(e)})
        failed += 1
        continue

    approved = response["approved"]
    print(f"PASS [{tx_id}] approved={approved} fraud_score={fs} ({elapsed_ms}ms)")
    passed += 1
    if approved:
        approved_count += 1
    else:
        declined_count += 1

print(f"\n=== Results: {passed} passed, {failed} failed out of {len(payloads)} "
      f"(approved={approved_count}, declined={declined_count}) ===")

latency_stats = {}
if latencies:
    latencies.sort()
    n = len(latencies)
    def pct(p):
        return latencies[min(int(p / 100 * n), n - 1)]

    latency_stats = {
        "n": n,
        "min_ms": round(min(latencies), 2),
        "mean_ms": round(statistics.mean(latencies), 2),
        "p50_ms": round(pct(50), 2),
        "p90_ms": round(pct(90), 2),
        "p99_ms": round(pct(99), 2),
        "max_ms": round(max(latencies), 2),
    }

    print("\n=== Latency (sequential, no queuing) ===")
    for k, v in latency_stats.items():
        print(f"  {k}: {v}")

result = {
    "timestamp": datetime.now().isoformat(),
    "base_url": BASE_URL,
    "summary": {
        "total": len(payloads),
        "passed": passed,
        "failed": failed,
        "approved": approved_count,
        "declined": declined_count,
    },
    "latency": latency_stats,
    "failures": failures,
}

out_path = f"/tmp/rinha-{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nResults saved to {out_path}")

sys.exit(0 if failed == 0 else 1)
