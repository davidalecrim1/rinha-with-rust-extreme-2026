const D: usize = 14;
const LANES: usize = 8;
const BLOCK_I16S: usize = D * LANES;
const SCALE: f32 = 0.0001;

#[cfg(feature = "ivf")]
const K: usize = 4096;
#[cfg(feature = "ivf")]
const FAST_NPROBE: usize = 16;
#[cfg(feature = "ivf")]
const FULL_NPROBE: usize = 24;

pub struct FraudIndex {
    blocks: &'static [u8],
    labels: &'static [u8],
    #[cfg(feature = "ivf")]
    centroids: Vec<f32>,
    #[cfg(feature = "ivf")]
    offsets: Vec<u32>,
    #[cfg(feature = "ivf")]
    fast_nprobe: usize,
    #[cfg(feature = "ivf")]
    full_nprobe: usize,
}

impl FraudIndex {
    pub fn build() -> Self {
        let blocks_raw: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/blocks.bin"));
        let labels_raw: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/labels.bin"));

        assert_eq!(
            blocks_raw.len() % 2,
            0,
            "blocks.bin must contain i16 values"
        );
        assert_eq!(
            (blocks_raw.len() / 2) % BLOCK_I16S,
            0,
            "blocks.bin must contain complete {BLOCK_I16S}-i16 blocks"
        );
        assert_eq!(
            labels_raw.len(),
            blocks_raw.len() / 2 / D,
            "labels.bin must contain {LANES} labels per block"
        );

        #[cfg(feature = "ivf")]
        let centroids = Self::load_centroids();
        #[cfg(feature = "ivf")]
        let offsets = Self::load_offsets(K + 1);

        #[cfg(feature = "ivf")]
        let fast_nprobe = env_usize("FAST_NPROBE", FAST_NPROBE).min(K);
        #[cfg(feature = "ivf")]
        let full_nprobe = env_usize("FULL_NPROBE", FULL_NPROBE)
            .min(K)
            .max(fast_nprobe);

        Self {
            blocks: blocks_raw,
            labels: labels_raw,
            #[cfg(feature = "ivf")]
            centroids,
            #[cfg(feature = "ivf")]
            offsets,
            #[cfg(feature = "ivf")]
            fast_nprobe,
            #[cfg(feature = "ivf")]
            full_nprobe,
        }
    }

    #[cfg(feature = "ivf")]
    fn load_centroids() -> Vec<f32> {
        let raw: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/centroids.bin"));
        assert_eq!(raw.len(), K * D * 4, "centroids.bin size mismatch");
        let mut centroids = Vec::with_capacity(K * D);
        for chunk in raw.chunks_exact(4) {
            centroids.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        centroids
    }

    #[cfg(feature = "ivf")]
    fn load_offsets(expected_len: usize) -> Vec<u32> {
        let raw: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/block_offsets.bin"));
        assert_eq!(
            raw.len(),
            expected_len * 4,
            "block_offsets.bin size mismatch"
        );
        let mut offsets = Vec::with_capacity(expected_len);
        for chunk in raw.chunks_exact(4) {
            offsets.push(u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        offsets
    }

    pub fn search(&self, q: &[f32; D]) -> f32 {
        #[cfg(feature = "ivf")]
        {
            self.search_ivf(q)
        }

        #[cfg(not(feature = "ivf"))]
        {
            self.search_all(q)
        }
    }

    #[cfg(any(not(feature = "ivf"), test))]
    fn search_all(&self, q: &[f32; D]) -> f32 {
        let mut top = [(f32::INFINITY, 0u8); 5];
        let mut worst_idx = 0usize;
        scan_blocks(
            q,
            &self.blocks,
            &self.labels,
            0,
            self.blocks.len() / 2 / BLOCK_I16S,
            &mut top,
            &mut worst_idx,
        );
        fraud_score(&top)
    }

    #[cfg(feature = "ivf")]
    fn search_ivf(&self, q: &[f32; D]) -> f32 {
        let mut centroid_dists = vec![0.0f32; K];
        compute_centroid_dists(q, &self.centroids, &mut centroid_dists);

        let fast = top_n_centroids::<FAST_NPROBE>(&centroid_dists);
        let fast_score = self.scan_probes(q, &fast[..self.fast_nprobe.min(FAST_NPROBE)]);
        if fast_score != 0.4 && fast_score != 0.6 {
            return fast_score;
        }

        let full = top_n_centroids::<FULL_NPROBE>(&centroid_dists);
        self.scan_probes(q, &full[..self.full_nprobe.min(FULL_NPROBE)])
    }

    #[cfg(feature = "ivf")]
    fn scan_probes(&self, q: &[f32; D], probes: &[usize]) -> f32 {
        let mut top = [(f32::INFINITY, 0u8); 5];
        let mut worst_idx = 0usize;

        for &cluster_id in probes {
            let start = self.offsets[cluster_id] as usize;
            let end = self.offsets[cluster_id + 1] as usize;
            scan_blocks(
                q,
                &self.blocks,
                &self.labels,
                start,
                end,
                &mut top,
                &mut worst_idx,
            );
        }

        fraud_score(&top)
    }
}

#[cfg(feature = "ivf")]
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

#[cfg(feature = "ivf")]
fn compute_centroid_dists(q: &[f32; D], centroids: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        compute_centroid_dists_avx2(q, centroids, out);
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        compute_centroid_dists_scalar(q, centroids, out);
    }
}

#[cfg(all(feature = "ivf", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn compute_centroid_dists_avx2(q: &[f32; D], centroids: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;

    let cp = centroids.as_ptr();
    let dp = out.as_mut_ptr();

    let q0 = _mm256_set1_ps(q[0]);
    let mut ci = 0usize;
    while ci + 8 <= K {
        let c = _mm256_loadu_ps(cp.add(ci));
        let d = _mm256_sub_ps(c, q0);
        _mm256_storeu_ps(dp.add(ci), _mm256_mul_ps(d, d));
        ci += 8;
    }

    for dim in 1..D {
        let base = dim * K;
        let qd = _mm256_set1_ps(q[dim]);
        let mut ci = 0usize;
        while ci + 8 <= K {
            let c = _mm256_loadu_ps(cp.add(base + ci));
            let d = _mm256_sub_ps(c, qd);
            let acc = _mm256_loadu_ps(dp.add(ci));
            _mm256_storeu_ps(dp.add(ci), _mm256_fmadd_ps(d, d, acc));
            ci += 8;
        }
    }
}

#[cfg(all(feature = "ivf", not(target_arch = "x86_64")))]
fn compute_centroid_dists_scalar(q: &[f32; D], centroids: &[f32], out: &mut [f32]) {
    for ci in 0..out.len() {
        let mut sum = 0.0f32;
        for dim in 0..D {
            let d = centroids[dim * out.len() + ci] - q[dim];
            sum += d * d;
        }
        out[ci] = sum;
    }
}

#[cfg(feature = "ivf")]
fn top_n_centroids<const N: usize>(dists: &[f32]) -> [usize; N] {
    let mut top_dists = [f32::INFINITY; N];
    let mut top_idx = [0usize; N];

    for (idx, &dist) in dists.iter().enumerate() {
        if dist < top_dists[N - 1] {
            let pos = top_dists.partition_point(|&x| x < dist);
            top_dists[pos..N].rotate_right(1);
            top_dists[pos] = dist;
            top_idx[pos..N].rotate_right(1);
            top_idx[pos] = idx;
        }
    }

    top_idx
}

fn scan_blocks(
    q: &[f32; D],
    blocks: &[u8],
    labels: &[u8],
    start_block: usize,
    end_block: usize,
    top: &mut [(f32, u8); 5],
    worst_idx: &mut usize,
) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        scan_blocks_avx2(q, blocks, labels, start_block, end_block, top, worst_idx);
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        scan_blocks_scalar(q, blocks, labels, start_block, end_block, top, worst_idx);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scan_blocks_avx2(
    q: &[f32; D],
    blocks: &[u8],
    labels: &[u8],
    start_block: usize,
    end_block: usize,
    top: &mut [(f32, u8); 5],
    worst_idx: &mut usize,
) {
    use std::arch::x86_64::*;

    let scale = _mm256_set1_ps(SCALE);
    let q_vecs = [
        _mm256_set1_ps(q[0]),
        _mm256_set1_ps(q[1]),
        _mm256_set1_ps(q[2]),
        _mm256_set1_ps(q[3]),
        _mm256_set1_ps(q[4]),
        _mm256_set1_ps(q[5]),
        _mm256_set1_ps(q[6]),
        _mm256_set1_ps(q[7]),
        _mm256_set1_ps(q[8]),
        _mm256_set1_ps(q[9]),
        _mm256_set1_ps(q[10]),
        _mm256_set1_ps(q[11]),
        _mm256_set1_ps(q[12]),
        _mm256_set1_ps(q[13]),
    ];

    let blocks_ptr = blocks.as_ptr();
    let labels_ptr = labels.as_ptr();

    for block_i in start_block..end_block {
        let prefetch = block_i + 8;
        if prefetch < end_block {
            _mm_prefetch(
                blocks_ptr.add(prefetch * BLOCK_I16S * 2) as *const i8,
                _MM_HINT_T0,
            );
            _mm_prefetch(
                blocks_ptr.add(prefetch * BLOCK_I16S * 2 + 112) as *const i8,
                _MM_HINT_T0,
            );
        }

        let base = block_i * BLOCK_I16S;
        let threshold = _mm256_set1_ps(top[*worst_idx].0);
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        dim_pair(&mut acc0, &mut acc1, blocks_ptr, base, &q_vecs, scale, 0);
        dim_pair(&mut acc0, &mut acc1, blocks_ptr, base, &q_vecs, scale, 2);
        dim_pair(&mut acc0, &mut acc1, blocks_ptr, base, &q_vecs, scale, 4);
        dim_pair(&mut acc0, &mut acc1, blocks_ptr, base, &q_vecs, scale, 6);

        let partial = _mm256_add_ps(acc0, acc1);
        if _mm256_movemask_ps(_mm256_cmp_ps(partial, threshold, _CMP_LT_OQ)) == 0 {
            continue;
        }

        dim_pair(&mut acc0, &mut acc1, blocks_ptr, base, &q_vecs, scale, 8);
        dim_pair(&mut acc0, &mut acc1, blocks_ptr, base, &q_vecs, scale, 10);
        dim_pair(&mut acc0, &mut acc1, blocks_ptr, base, &q_vecs, scale, 12);

        let acc = _mm256_add_ps(acc0, acc1);
        let mut mask = _mm256_movemask_ps(_mm256_cmp_ps(acc, threshold, _CMP_LT_OQ)) as u32;
        if mask == 0 {
            continue;
        }

        let mut dists = [0.0f32; LANES];
        _mm256_storeu_ps(dists.as_mut_ptr(), acc);
        let label_base = block_i * LANES;
        while mask != 0 {
            let lane = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            update_top(
                top,
                worst_idx,
                dists[lane],
                *labels_ptr.add(label_base + lane),
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dim_pair(
    acc0: &mut std::arch::x86_64::__m256,
    acc1: &mut std::arch::x86_64::__m256,
    blocks_ptr: *const u8,
    base: usize,
    q_vecs: &[std::arch::x86_64::__m256; D],
    scale: std::arch::x86_64::__m256,
    dim: usize,
) {
    use std::arch::x86_64::*;

    let r0 = _mm_loadu_si128(blocks_ptr.add((base + dim * LANES) * 2) as *const __m128i);
    let r1 = _mm_loadu_si128(blocks_ptr.add((base + (dim + 1) * LANES) * 2) as *const __m128i);
    let v0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(r0)), scale);
    let v1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(r1)), scale);
    let d0 = _mm256_sub_ps(v0, q_vecs[dim]);
    let d1 = _mm256_sub_ps(v1, q_vecs[dim + 1]);
    *acc0 = _mm256_fmadd_ps(d0, d0, *acc0);
    *acc1 = _mm256_fmadd_ps(d1, d1, *acc1);
}

#[cfg(not(target_arch = "x86_64"))]
fn scan_blocks_scalar(
    q: &[f32; D],
    blocks: &[u8],
    labels: &[u8],
    start_block: usize,
    end_block: usize,
    top: &mut [(f32, u8); 5],
    worst_idx: &mut usize,
) {
    for block_i in start_block..end_block {
        let base = block_i * BLOCK_I16S;
        for lane in 0..LANES {
            let mut dist = 0.0f32;
            for dim in 0..D {
                let offset = (base + dim * LANES + lane) * 2;
                let rv = i16::from_ne_bytes([blocks[offset], blocks[offset + 1]]) as f32 * SCALE;
                let d = rv - q[dim];
                dist += d * d;
            }
            update_top(top, worst_idx, dist, labels[block_i * LANES + lane]);
        }
    }
}

#[inline(always)]
fn update_top(top: &mut [(f32, u8); 5], worst_idx: &mut usize, dist: f32, label: u8) {
    if dist >= top[*worst_idx].0 {
        return;
    }

    top[*worst_idx] = (dist, label);
    let mut wi = 0usize;
    let mut wd = top[0].0;
    for (i, &(d, _)) in top.iter().enumerate().skip(1) {
        if d > wd {
            wd = d;
            wi = i;
        }
    }
    *worst_idx = wi;
}

fn fraud_score(top: &[(f32, u8); 5]) -> f32 {
    top.iter().filter(|(_, label)| *label != 0).count() as f32 / 5.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_index(entries: &[([f32; D], bool)]) -> FraudIndex {
        let mut blocks_i16 = Vec::new();
        let mut labels = Vec::new();

        for chunk in entries.chunks(LANES) {
            for lane in 0..LANES {
                labels.push(chunk.get(lane).map(|(_, f)| *f as u8).unwrap_or(0));
            }
            for dim in 0..D {
                for lane in 0..LANES {
                    blocks_i16.push(
                        chunk
                            .get(lane)
                            .map(|(v, _)| (v[dim] * 10_000.0).round() as i16)
                            .unwrap_or(i16::MAX),
                    );
                }
            }
        }

        let mut blocks = Vec::with_capacity(blocks_i16.len() * 2);
        for value in blocks_i16 {
            blocks.extend_from_slice(&value.to_ne_bytes());
        }

        FraudIndex {
            blocks: Box::leak(blocks.into_boxed_slice()),
            labels: Box::leak(labels.into_boxed_slice()),
            #[cfg(feature = "ivf")]
            centroids: Vec::new(),
            #[cfg(feature = "ivf")]
            offsets: Vec::new(),
            #[cfg(feature = "ivf")]
            fast_nprobe: FAST_NPROBE,
            #[cfg(feature = "ivf")]
            full_nprobe: FULL_NPROBE,
        }
    }

    const ZERO: [f32; D] = [0.0; D];

    #[test]
    fn search_all_fraud() {
        let idx = make_index(&[
            (ZERO, true),
            (ZERO, true),
            (ZERO, true),
            (ZERO, true),
            (ZERO, true),
        ]);
        assert_eq!(idx.search_all(&ZERO), 1.0);
    }

    #[test]
    fn search_mixed_3_fraud_2_legit() {
        let idx = make_index(&[
            (ZERO, true),
            (ZERO, true),
            (ZERO, true),
            (ZERO, false),
            (ZERO, false),
        ]);
        assert_eq!(idx.search_all(&ZERO), 0.6);
    }

    #[test]
    fn padded_lanes_do_not_enter_top5() {
        let close = [0.01f32; D];
        let far = [0.9f32; D];
        let idx = make_index(&[
            (close, true),
            (close, true),
            (close, true),
            (close, true),
            (close, true),
            (far, false),
        ]);
        assert_eq!(idx.search_all(&ZERO), 1.0);
    }
}
