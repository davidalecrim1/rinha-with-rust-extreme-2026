use flate2::read::GzDecoder;
use serde::Deserialize;
use std::io::Read;
use std::path::PathBuf;

#[derive(Deserialize)]
struct RefEntry {
    vector: [f32; D],
    label: String,
}

const D: usize = 14;
const LANES: usize = 8;
const BLOCK_I16S: usize = D * LANES;
const K: usize = 4096;
const KMEANS_MAX_ITERS: usize = 25;
const SCALE: f32 = 10_000.0;

fn quantize(v: f32) -> i16 {
    (v * SCALE).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

fn l2_f32(a: &[f32; D], b: &[f32; D]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..D {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

fn kmeans(vectors: &[[f32; D]]) -> (Vec<[f32; D]>, Vec<u16>) {
    let n = vectors.len();
    let mut centroids = kmeans_plus_plus_init(vectors, K, 0xdeadbeef_cafebabe);
    let mut assignments = vec![0u16; n];

    for iter in 0..KMEANS_MAX_ITERS {
        let changed = assign_parallel(vectors, &centroids, &mut assignments);
        update_centroids(vectors, &assignments, &mut centroids);

        let pct = changed as f64 / n as f64 * 100.0;
        eprintln!(
            "cargo:warning=k-means iter {}: {} changed ({:.2}%)",
            iter + 1,
            changed,
            pct
        );
        if pct < 0.1 {
            eprintln!("cargo:warning=k-means converged at iteration {}", iter + 1);
            break;
        }
    }

    let mut sizes = vec![0u32; K];
    for &a in &assignments {
        sizes[a as usize] += 1;
    }
    let min_sz = *sizes.iter().min().unwrap();
    let max_sz = *sizes.iter().max().unwrap();
    let avg_sz = n as f64 / K as f64;
    let empty = sizes.iter().filter(|&&s| s == 0).count();
    eprintln!(
        "cargo:warning=k-means cluster sizes: min={}, max={}, avg={:.0}, empty={}",
        min_sz, max_sz, avg_sz, empty
    );

    (centroids, assignments)
}

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() >> 33) as usize % n
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn kmeans_plus_plus_init(vectors: &[[f32; D]], k: usize, seed: u64) -> Vec<[f32; D]> {
    let n = vectors.len();
    let mut rng = Lcg::new(seed);
    let sample_size = n.min(50_000);
    let sample: Vec<usize> = (0..sample_size).map(|_| rng.next_usize(n)).collect();
    let mut centroids = Vec::with_capacity(k);
    centroids.push(vectors[sample[rng.next_usize(sample_size)]]);

    let mut min_dists = vec![f32::INFINITY; sample_size];
    for _ in 1..k {
        let last = *centroids.last().unwrap();
        for (i, &vi) in sample.iter().enumerate() {
            let d = l2_f32(&vectors[vi], &last);
            if d < min_dists[i] {
                min_dists[i] = d;
            }
        }

        let total: f64 = min_dists.iter().map(|&x| x as f64).sum();
        let r = rng.next_f64() * total;
        let mut cum = 0.0f64;
        let mut chosen = sample_size - 1;
        for (i, &d) in min_dists.iter().enumerate() {
            cum += d as f64;
            if cum >= r {
                chosen = i;
                break;
            }
        }
        centroids.push(vectors[sample[chosen]]);
    }

    centroids
}

fn nearest_centroid(v: &[f32; D], centroids: &[[f32; D]]) -> u16 {
    let mut best_dist = f32::INFINITY;
    let mut best_idx = 0u16;
    for (i, c) in centroids.iter().enumerate() {
        let d = l2_f32(v, c);
        if d < best_dist {
            best_dist = d;
            best_idx = i as u16;
        }
    }
    best_idx
}

fn assign_parallel(vectors: &[[f32; D]], centroids: &[[f32; D]], assignments: &mut [u16]) -> usize {
    let n_threads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(4)
        .min(16);
    let chunk = vectors.len().div_ceil(n_threads);
    let total_changed = std::sync::atomic::AtomicUsize::new(0);

    std::thread::scope(|s| {
        for (v_chunk, a_chunk) in vectors.chunks(chunk).zip(assignments.chunks_mut(chunk)) {
            let total_changed = &total_changed;
            s.spawn(move || {
                let mut changed = 0usize;
                for (v, a) in v_chunk.iter().zip(a_chunk.iter_mut()) {
                    let best = nearest_centroid(v, centroids);
                    if best != *a {
                        changed += 1;
                        *a = best;
                    }
                }
                total_changed.fetch_add(changed, std::sync::atomic::Ordering::Relaxed);
            });
        }
    });

    total_changed.load(std::sync::atomic::Ordering::Relaxed)
}

fn update_centroids(vectors: &[[f32; D]], assignments: &[u16], centroids: &mut [[f32; D]]) {
    let mut sums = vec![[0.0f64; D]; K];
    let mut counts = vec![0u32; K];
    for (v, &a) in vectors.iter().zip(assignments.iter()) {
        let c = a as usize;
        counts[c] += 1;
        for d in 0..D {
            sums[c][d] += v[d] as f64;
        }
    }
    for c in 0..K {
        if counts[c] == 0 {
            continue;
        }
        for d in 0..D {
            centroids[c][d] = (sums[c][d] / counts[c] as f64) as f32;
        }
    }
}

fn write_blocks(
    out_dir: &PathBuf,
    entries: &[RefEntry],
    order_by_cluster: Option<(&[usize], &[u16])>,
    centroids: Option<&[[f32; D]]>,
) {
    let mut block_offsets = Vec::new();
    let mut labels = Vec::new();
    let mut blocks = Vec::new();

    match order_by_cluster {
        Some((order, assignments)) => {
            let mut pos = 0usize;
            for c in 0..K {
                block_offsets.push((blocks.len() / BLOCK_I16S) as u32);
                while pos < order.len() && assignments[order[pos]] as usize == c {
                    let end = (pos + LANES).min(order.len());
                    let mut lane_indices = Vec::with_capacity(LANES);
                    while pos < end && assignments[order[pos]] as usize == c {
                        lane_indices.push(order[pos]);
                        pos += 1;
                    }
                    write_one_block(entries, &lane_indices, &mut labels, &mut blocks);
                }
            }
            block_offsets.push((blocks.len() / BLOCK_I16S) as u32);
        }
        None => {
            block_offsets.push(0);
            for chunk_start in (0..entries.len()).step_by(LANES) {
                let lane_indices: Vec<usize> =
                    (chunk_start..(chunk_start + LANES).min(entries.len())).collect();
                write_one_block(entries, &lane_indices, &mut labels, &mut blocks);
            }
            block_offsets.push((blocks.len() / BLOCK_I16S) as u32);
        }
    }

    std::fs::write(out_dir.join("labels.bin"), &labels).expect("failed to write labels.bin");

    let mut block_blob = Vec::with_capacity(blocks.len() * 2);
    for v in blocks {
        block_blob.extend_from_slice(&v.to_ne_bytes());
    }
    std::fs::write(out_dir.join("blocks.bin"), &block_blob).expect("failed to write blocks.bin");

    let mut off_blob = Vec::with_capacity(block_offsets.len() * 4);
    for o in block_offsets {
        off_blob.extend_from_slice(&o.to_ne_bytes());
    }
    std::fs::write(out_dir.join("block_offsets.bin"), &off_blob)
        .expect("failed to write block_offsets.bin");

    if let Some(centroids) = centroids {
        let mut cent_blob = Vec::with_capacity(K * D * 4);
        for d in 0..D {
            for centroid in centroids {
                cent_blob.extend_from_slice(&centroid[d].to_ne_bytes());
            }
        }
        std::fs::write(out_dir.join("centroids.bin"), &cent_blob)
            .expect("failed to write centroids.bin");
    }
}

fn write_one_block(
    entries: &[RefEntry],
    lane_indices: &[usize],
    labels: &mut Vec<u8>,
    blocks: &mut Vec<i16>,
) {
    for lane in 0..LANES {
        let label = lane_indices
            .get(lane)
            .map(|&i| (entries[i].label == "fraud") as u8)
            .unwrap_or(0);
        labels.push(label);
    }

    for d in 0..D {
        for lane in 0..LANES {
            let value = lane_indices
                .get(lane)
                .map(|&i| quantize(entries[i].vector[d]))
                .unwrap_or(i16::MAX);
            blocks.push(value);
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=resources/references.json.gz");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let gz = std::fs::read("resources/references.json.gz")
        .expect("resources/references.json.gz not found");
    let mut decoder = GzDecoder::new(gz.as_slice());
    let mut json = String::new();
    decoder
        .read_to_string(&mut json)
        .expect("failed to decompress references");

    let entries: Vec<RefEntry> =
        serde_json::from_str(&json).expect("failed to parse references JSON");

    if std::env::var("CARGO_FEATURE_IVF").is_ok() {
        eprintln!(
            "cargo:warning=IVF enabled: running k-means (K={}) on {} vectors...",
            K,
            entries.len()
        );

        let vectors: Vec<[f32; D]> = entries.iter().map(|e| e.vector).collect();
        let (centroids, assignments) = kmeans(&vectors);
        let mut order: Vec<usize> = (0..entries.len()).collect();
        order.sort_by_key(|&i| assignments[i]);

        write_blocks(
            &out_dir,
            &entries,
            Some((&order, &assignments)),
            Some(&centroids),
        );

        eprintln!(
            "cargo:warning=build.rs: packed {} references into AVX2 IVF blocks ({} clusters)",
            entries.len(),
            K
        );
    } else {
        write_blocks(&out_dir, &entries, None, None);
        eprintln!(
            "cargo:warning=build.rs: packed {} references into AVX2 blocks",
            entries.len()
        );
    }
}
