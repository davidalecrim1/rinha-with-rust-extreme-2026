use chrono::{DateTime, FixedOffset};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct FraudRequest {
    #[allow(dead_code)]
    pub id: String,
    pub transaction: Transaction,
    pub customer: Customer,
    pub merchant: Merchant,
    pub terminal: Terminal,
    pub last_transaction: Option<LastTransaction>,
}

#[derive(Deserialize)]
pub struct Transaction {
    pub amount: f64,
    pub installments: u32,
    pub requested_at: DateTime<FixedOffset>,
}

#[derive(Deserialize)]
pub struct Customer {
    pub avg_amount: f64,
    pub tx_count_24h: u32,
    pub known_merchants: Vec<String>,
}

#[derive(Deserialize)]
pub struct Merchant {
    pub id: String,
    pub mcc: String,
    pub avg_amount: f64,
}

#[derive(Deserialize)]
pub struct Terminal {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f64,
}

#[derive(Deserialize)]
pub struct LastTransaction {
    pub timestamp: DateTime<FixedOffset>,
    pub km_from_current: f64,
}

#[derive(Deserialize)]
pub struct NormConsts {
    pub max_amount: f64,
    pub max_installments: f64,
    pub amount_vs_avg_ratio: f64,
    pub max_minutes: f64,
    pub max_km: f64,
    pub max_tx_count_24h: f64,
    pub max_merchant_avg_amount: f64,
}
