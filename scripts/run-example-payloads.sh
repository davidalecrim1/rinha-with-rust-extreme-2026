#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:9999}"
PAYLOADS_URL="https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/example-payloads.json"

echo "Fetching example-payloads.json..."
payloads=$(curl -sf "$PAYLOADS_URL")
total=$(echo "$payloads" | python3 -c "import json,sys; print(len(json.load(sys.stdin)))")
echo "Loaded $total payloads"
echo ""

passed=0
failed=0
total_approved=0
total_declined=0

for i in $(seq 0 $((total - 1))); do
    payload=$(echo "$payloads" | python3 -c "import json,sys; data=json.load(sys.stdin); print(json.dumps(data[$i]))")
    tx_id=$(echo "$payload" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")

    http_code=$(curl -s -o /tmp/rinha_response.json -w "%{http_code}" \
        -X POST "$BASE_URL/fraud-score" \
        -H "Content-Type: application/json" \
        -d "$payload")

    if [[ "$http_code" != "200" ]]; then
        echo "FAIL [$tx_id] HTTP $http_code"
        failed=$((failed + 1))
        continue
    fi

    response=$(cat /tmp/rinha_response.json)

    # Validate response structure
    validation=$(echo "$response" | python3 -c "
import json, sys
try:
    r = json.load(sys.stdin)
    assert isinstance(r.get('approved'), bool), 'approved must be bool'
    fs = r.get('fraud_score')
    assert isinstance(fs, (int, float)), 'fraud_score must be number'
    assert 0.0 <= fs <= 1.0, f'fraud_score {fs} out of range [0,1]'
    print('ok|' + str(r['approved']) + '|' + str(r['fraud_score']))
except Exception as e:
    print('err|' + str(e))
" 2>&1)

    if [[ "$validation" == err* ]]; then
        reason="${validation#err|}"
        echo "FAIL [$tx_id] $reason — response: $response"
        failed=$((failed + 1))
    else
        rest="${validation#ok|}"
        approved="${rest%%|*}"
        echo "PASS [$tx_id] approved=$rest"
        passed=$((passed + 1))
        if [[ "$approved" == "True" ]]; then
            total_approved=$((total_approved + 1))
        else
            total_declined=$((total_declined + 1))
        fi
    fi
done

echo ""
echo "=== Results: $passed passed, $failed failed out of $total (approved=$total_approved, declined=$total_declined) ==="

[[ $failed -eq 0 ]]
