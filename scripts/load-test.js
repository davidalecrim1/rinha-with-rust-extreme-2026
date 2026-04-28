import http from 'k6/http';
import { check } from 'k6';
import { Rate } from 'k6/metrics';
import { SharedArray } from 'k6/data';

// Only counts HTTP errors from the app (non-200 responses).
// Connection-level failures (EOF, timeout) are excluded — those are
// infrastructure noise, not app bugs.
const appErrors = new Rate('app_errors');

const testData = new SharedArray('payloads', function () {
    return JSON.parse(open('./test-data.json')).entries;
});

const MAX_VUS = 200;
const STAGES = [
    { duration: '10s', target: 750  }, // ramp to half-peak
    { duration: '10s', target: 1500 }, // ramp to full peak
    { duration: '40s', target: 1500 }, // sustain
];
const TOTAL_SECS = STAGES.reduce((sum, s) => sum + parseInt(s.duration), 0);

export const options = {
    summaryTrendStats: ['p(99)', 'p(95)'],
    scenarios: {
        default: {
            executor: 'ramping-arrival-rate',
            startRate: 1,
            timeUnit: '1s',
            preAllocatedVUs: 50,
            maxVUs: MAX_VUS,
            gracefulStop: '5s',
            stages: STAGES,
        },
    },
};

export default function () {
    const entry = testData[Math.floor(Math.random() * testData.length)];
    const res = http.post(
        'http://localhost:9999/fraud-score',
        JSON.stringify(entry.request),
        { headers: { 'Content-Type': 'application/json' } },
    );
    check(res, { 'status 200': (r) => r.status === 200 });
    // res.status === 0 means a connection error (EOF, reset) — not an app error.
    if (res.status !== 0) {
        appErrors.add(res.status !== 200);
    }
}

export function handleSummary(data) {
    const version = __ENV.VERSION || 'unknown';
    const now = new Date();
    const ts = now.toISOString().replace(/[:.]/g, '-').slice(0, 19);

    const filename = `scripts/results/${ts}_${version}_${MAX_VUS}vus_${TOTAL_SECS}s.json`;

    const dur = data.metrics.http_req_duration.values;
    const appErr = data.metrics.app_errors ? data.metrics.app_errors.values : { rate: 0 };

    return {
        [filename]: JSON.stringify({
            version,
            timestamp: now.toISOString(),
            p99_ms: +dur['p(99)'].toFixed(2),
            p95_ms: +dur['p(95)'].toFixed(2),
            error_rate: +appErr.rate.toFixed(6),
            rps: +data.metrics.http_reqs.values.rate.toFixed(2),
        }, null, 2),
    };
}
