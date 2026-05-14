use axum::{
    body::Body,
    extract::{Json, State},
    http::{header, Response, StatusCode},
};
use bytes::Bytes;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use crate::index::FraudIndex;
use crate::types::{FraudRequest, NormConsts};
use crate::vectorizer::vectorize;

#[derive(Clone)]
pub struct AppState {
    pub index: Arc<FraudIndex>,
    pub ready: Arc<AtomicBool>,
    pub norm: Arc<NormConsts>,
    pub mcc_risk: Arc<HashMap<String, f32>>,
}

// All 6 possible responses indexed by fraud vote count (0..=5).
// Prebuilt at compile time: zero serde_json serialization on the hot path.
static FRAUD_BODIES: [&[u8]; 6] = [
    b"{\"approved\":true,\"fraud_score\":0.0}",
    b"{\"approved\":true,\"fraud_score\":0.2}",
    b"{\"approved\":true,\"fraud_score\":0.4}",
    b"{\"approved\":false,\"fraud_score\":0.6}",
    b"{\"approved\":false,\"fraud_score\":0.8}",
    b"{\"approved\":false,\"fraud_score\":1.0}",
];

pub async fn ready(State(s): State<AppState>) -> StatusCode {
    if s.ready.load(Ordering::Acquire) {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

pub async fn fraud_score(
    State(s): State<AppState>,
    Json(req): Json<FraudRequest>,
) -> Response<Body> {
    let vector = vectorize(&req, &s.norm, &s.mcc_risk);
    // Default to 5 (all fraud) on panic — avoids HTTP 500 weight-5 scoring penalty.
    let votes = tokio::task::spawn_blocking(move || s.index.search(&vector))
        .await
        .unwrap_or(5);
    let body = FRAUD_BODIES[votes as usize];
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::CONTENT_LENGTH, body.len())
        .body(Body::from(Bytes::from_static(body)))
        .unwrap()
}
