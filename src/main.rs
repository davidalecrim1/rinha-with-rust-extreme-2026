mod handler;
mod index;
mod types;
mod vectorizer;

use axum::routing::{get, post};
use axum::Router;
use handler::AppState;
use index::FraudIndex;
use std::collections::HashMap;
use std::sync::{atomic::AtomicBool, Arc};
use tokio::net::TcpListener;
use types::NormConsts;

#[tokio::main(flavor = "multi_thread", worker_threads = 1)]
async fn main() {
    let gz = include_bytes!("../resources/references.json.gz");
    let mcc_raw = include_bytes!("../resources/mcc_risk.json");
    let norm_raw = include_bytes!("../resources/normalization.json");

    let index = Arc::new(FraudIndex::build(gz));
    let mcc_risk = Arc::new(
        serde_json::from_slice::<HashMap<String, f32>>(mcc_raw)
            .expect("mcc_risk.json embedded in binary is invalid"),
    );
    let norm = Arc::new(
        serde_json::from_slice::<NormConsts>(norm_raw)
            .expect("normalization.json embedded in binary is invalid"),
    );
    let ready = Arc::new(AtomicBool::new(true));

    let state = AppState {
        index,
        ready,
        norm,
        mcc_risk,
    };

    let app = Router::new()
        .route("/ready", get(handler::ready))
        .route("/fraud-score", post(handler::fraud_score))
        .with_state(state);

    #[cfg(feature = "profiling")]
    let _profiler = {
        let guard = pprof::ProfilerGuardBuilder::default()
            .frequency(4000)
            .blocklist(&["libc", "libgcc", "pthread", "vdso"])
            .build()
            .expect("failed to start profiler");

        let guard_ref = std::sync::Arc::new(std::sync::Mutex::new(Some(guard)));
        let g = guard_ref.clone();
        tokio::spawn(async move {
            use tokio::signal::unix::{signal, SignalKind};
            let mut sigterm = signal(SignalKind::terminate()).expect("failed to register SIGTERM");
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = sigterm.recv() => {}
            }
            if let Some(guard) = g.lock().unwrap().take() {
                let report = guard.report().build().expect("failed to build profile report");

                let mut file = std::fs::File::create("/profile/profile.folded").unwrap();
                use std::io::Write;
                use std::fmt::Write as FmtWrite;
                for (key, count) in report.data.iter() {
                    let mut line = key.thread_name_or_id();
                    line.push(';');
                    for frame in key.frames.iter().rev() {
                        for symbol in frame.iter().rev() {
                            write!(&mut line, "{};", symbol).unwrap();
                        }
                    }
                    line.pop();
                    writeln!(file, "{} {}", line, count).unwrap();
                }

                let mut svg = std::fs::File::create("/profile/profile.svg").unwrap();
                report.flamegraph(&mut svg).unwrap();

                eprintln!("Profile written to /profile/profile.folded and /profile/profile.svg");
                std::process::exit(0);
            }
        });
        guard_ref
    };

    let listener = TcpListener::bind("0.0.0.0:8080").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
