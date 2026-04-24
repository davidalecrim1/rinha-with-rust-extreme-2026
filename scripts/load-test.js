import http from 'k6/http';
import { check } from 'k6';
import { SharedArray } from 'k6/data';

const testData = new SharedArray('payloads', function () {
    return JSON.parse(open('../../rinha-de-backend-2026/test/test-data.json')).entries;
});

export const options = {
    summaryTrendStats: ['p(99)', 'p(95)'],
    scenarios: {
        default: {
            executor: 'ramping-arrival-rate',
            startRate: 1,
            timeUnit: '1s',
            preAllocatedVUs: 50,
            maxVUs: 200,
            gracefulStop: '5s',
            stages: [
                { duration: '10s', target: 750  }, // ramp to half-peak
                { duration: '10s', target: 1500 }, // ramp to full peak
                { duration: '40s', target: 1500 }, // sustain
            ],
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
}

export function handleSummary(data) {
    const version = __ENV.VERSION || 'unknown';
    const now = new Date();
    const ts = now.toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const filename = `scripts/results/${ts}_${version}.json`;

    const dur = data.metrics.http_req_duration.values;
    const failed = data.metrics.http_req_failed.values;

    return {
        [filename]: JSON.stringify({
            version,
            timestamp: now.toISOString(),
            p99_ms: +dur['p(99)'].toFixed(2),
            p95_ms: +dur['p(95)'].toFixed(2),
            error_rate: +failed.rate.toFixed(6),
            rps: +data.metrics.http_reqs.values.rate.toFixed(2),
        }, null, 2),
    };
}
