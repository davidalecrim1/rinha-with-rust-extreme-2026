use crate::packed_ref::quantize;
use crate::packed_ref::{query_cont_bytes, unpack_bits, PartialDists};
use crate::simd;
#[cfg(test)]
use std::time::Instant;

// 16-byte row layout (matches build.rs output):
//   bytes  0-11 : [i16; 6] continuous dims (0, 2, 5, 6, 7, 13), native-endian
//   bytes 12-14 : [u8; 3]  packed low-cardinality indices + binary flags
//   byte  15    : u8        fraud label (0 = legit, 1 = fraud)
const ROW: usize = 16;

// When two binary dims differ (one is 0, the other is 8192 quantized), the
// squared distance is (8192 - 0)^2 = 67_108_864.
const BIT_DIST_SQ: u64 = 8192 * 8192;
const K: usize = 1024;
const DEFAULT_PRIMARY_NPROBE: usize = 4;
const DEFAULT_REFINE_NPROBE: usize = 24;
const DEFAULT_REPAIR_MAX_EXTRA_CLUSTERS: usize = 96;
const DEFAULT_REPAIR_MAX_EXTRA_ROWS: usize = 120_000;
type IvfMetadata = (Vec<[f32; 14]>, Vec<u32>, Vec<[i16; 14]>, Vec<[i16; 14]>);

pub struct FraudIndex {
    rows: Vec<[u8; 16]>,
    centroids: Vec<[f32; 14]>,
    offsets: Vec<u32>,
    bbox_min: Vec<[i16; 14]>,
    bbox_max: Vec<[i16; 14]>,
    primary_nprobe: usize,
    refine_nprobe: usize,
    ivf_repair: bool,
    repair_max_extra_clusters: usize,
    repair_max_extra_rows: usize,
}

impl FraudIndex {
    /// Load the pre-packed binary blob produced by build.rs.
    /// No decompression, no JSON parsing, single memcpy from the binary's
    /// read-only data segment.
    pub fn build() -> Self {
        let raw: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/packed_refs.bin"));

        assert!(
            raw.len().is_multiple_of(ROW),
            "packed_refs.bin size {} is not a multiple of {ROW}",
            raw.len()
        );

        let mut rows: Vec<[u8; 16]> = Vec::with_capacity(raw.len() / ROW);
        for chunk in raw.chunks_exact(ROW) {
            rows.push(chunk.try_into().unwrap());
        }

        let (centroids, offsets, bbox_min, bbox_max) = Self::load_ivf_metadata();

        let legacy_nprobe = std::env::var("NPROBE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());

        let primary_nprobe = std::env::var("IVF_PRIMARY_NPROBE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .or(legacy_nprobe)
            .unwrap_or(DEFAULT_PRIMARY_NPROBE);

        let refine_nprobe = std::env::var("IVF_REFINE_NPROBE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_REFINE_NPROBE);

        let ivf_repair = std::env::var("IVF_REPAIR")
            .map(|v| v != "0")
            .unwrap_or(true);

        let repair_max_extra_clusters = std::env::var("IVF_REPAIR_MAX_EXTRA_CLUSTERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_REPAIR_MAX_EXTRA_CLUSTERS)
            .min(K);

        let repair_max_extra_rows = std::env::var("IVF_REPAIR_MAX_EXTRA_ROWS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_REPAIR_MAX_EXTRA_ROWS);

        FraudIndex {
            rows,
            centroids,
            offsets,
            bbox_min,
            bbox_max,
            primary_nprobe,
            refine_nprobe,
            ivf_repair,
            repair_max_extra_clusters,
            repair_max_extra_rows,
        }
    }

    fn load_ivf_metadata() -> IvfMetadata {
        let cent_raw: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/centroids.bin"));
        let off_raw: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cluster_offsets.bin"));
        let bbox_min_raw: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bbox_min.bin"));
        let bbox_max_raw: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bbox_max.bin"));

        assert_eq!(
            cent_raw.len(),
            K * 14 * 4,
            "centroids.bin size mismatch: expected {}, got {}",
            K * 14 * 4,
            cent_raw.len()
        );
        assert_eq!(
            off_raw.len(),
            (K + 1) * 4,
            "cluster_offsets.bin size mismatch: expected {}, got {}",
            (K + 1) * 4,
            off_raw.len()
        );
        assert_eq!(
            bbox_min_raw.len(),
            K * 14 * 2,
            "bbox_min.bin size mismatch: expected {}, got {}",
            K * 14 * 2,
            bbox_min_raw.len()
        );
        assert_eq!(
            bbox_max_raw.len(),
            K * 14 * 2,
            "bbox_max.bin size mismatch: expected {}, got {}",
            K * 14 * 2,
            bbox_max_raw.len()
        );

        let mut centroids = Vec::with_capacity(K);
        for i in 0..K {
            let mut centroid = [0.0f32; 14];
            for (d, value) in centroid.iter_mut().enumerate() {
                let offset = (i * 14 + d) * 4;
                *value = f32::from_ne_bytes(cent_raw[offset..offset + 4].try_into().unwrap());
            }
            centroids.push(centroid);
        }

        let mut offsets = Vec::with_capacity(K + 1);
        for i in 0..=K {
            let offset = i * 4;
            offsets.push(u32::from_ne_bytes(
                off_raw[offset..offset + 4].try_into().unwrap(),
            ));
        }

        let mut bbox_min = Vec::with_capacity(K);
        let mut bbox_max = Vec::with_capacity(K);
        for i in 0..K {
            let mut mins = [0i16; 14];
            let mut maxes = [0i16; 14];
            for d in 0..14 {
                let offset = (i * 14 + d) * 2;
                mins[d] = i16::from_ne_bytes(bbox_min_raw[offset..offset + 2].try_into().unwrap());
                maxes[d] = i16::from_ne_bytes(bbox_max_raw[offset..offset + 2].try_into().unwrap());
            }
            bbox_min.push(mins);
            bbox_max.push(maxes);
        }

        (centroids, offsets, bbox_min, bbox_max)
    }

    /// Returns the fraud score for query vector `q` using k-NN with k=5.
    pub fn search(&self, q: &[f32; 14]) -> f32 {
        self.search_ivf(q).score
    }

    fn search_ivf(&self, q: &[f32; 14]) -> IvfSearchStats {
        self.search_ivf_decision_aware(q, self.primary_nprobe, self.refine_nprobe, self.ivf_repair)
    }

    fn search_ivf_decision_aware(
        &self,
        q: &[f32; 14],
        primary_nprobe: usize,
        refine_nprobe: usize,
        repair: bool,
    ) -> IvfSearchStats {
        #[cfg(test)]
        let started = Instant::now();
        let primary_nprobe = primary_nprobe.clamp(1, K);
        let refine_nprobe = refine_nprobe.clamp(primary_nprobe, K);

        let mut cdists: Vec<(f32, u16)> = Vec::with_capacity(K);
        for (i, centroid) in self.centroids.iter().enumerate() {
            cdists.push((l2_f32(q, centroid), i as u16));
        }

        cdists.sort_unstable_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        let q_cont = query_cont_bytes(q);
        let pd = PartialDists::compute(q);

        #[cfg(target_arch = "x86_64")]
        let (q_simd, zero_lanes_mask) = unsafe {
            use std::arch::x86_64::*;
            let q_v = simd::load_m128(&q_cont);
            let mask = _mm_set_epi32(0, 0, -1i32, -1i32);
            (q_v, mask)
        };

        let mut neighbors: [Neighbor; 5] = [Neighbor::empty(); 5];
        let (mut worst_slot, mut worst_dist) = (0usize, u64::MAX);
        let mut scanned = [false; K];
        #[cfg(test)]
        let mut clusters_scanned = 0usize;
        #[cfg(test)]
        let mut rows_scanned = 0usize;
        let mut extra_clusters_scanned = 0usize;
        let mut extra_rows_scanned = 0usize;

        for &(_, cluster_id) in &cdists[..primary_nprobe] {
            let cluster_id = cluster_id as usize;
            let start = self.offsets[cluster_id] as usize;
            let end = self.offsets[cluster_id + 1] as usize;
            scanned[cluster_id] = true;
            #[cfg(test)]
            {
                clusters_scanned += 1;
                rows_scanned += end - start;
            }

            scan_rows(
                &self.rows[start..end],
                start as u32,
                &q_cont,
                &pd,
                #[cfg(target_arch = "x86_64")]
                q_simd,
                #[cfg(target_arch = "x86_64")]
                zero_lanes_mask,
                &mut neighbors,
                &mut worst_slot,
                &mut worst_dist,
            );
        }

        let primary_fraud_votes = fraud_votes(&neighbors);

        if repair
            && self.repair_max_extra_clusters > 0
            && self.repair_max_extra_rows > 0
            && is_ambiguous(primary_fraud_votes)
        {
            let q_quantized = quantized_search_vector(q);
            let mut repair_candidates: Vec<(u64, u16)> =
                Vec::with_capacity(K.saturating_sub(primary_nprobe));

            for (rank, &(_, cluster_id)) in cdists.iter().enumerate() {
                let cluster_id = cluster_id as usize;
                if scanned[cluster_id] {
                    continue;
                }

                let lower_bound = bbox_lower_bound_sq(
                    &q_quantized,
                    &self.bbox_min[cluster_id],
                    &self.bbox_max[cluster_id],
                );
                if rank >= refine_nprobe && lower_bound > worst_dist {
                    continue;
                }

                repair_candidates.push((lower_bound, cluster_id as u16));
            }

            repair_candidates
                .sort_unstable_by_key(|&(lower_bound, cluster_id)| (lower_bound, cluster_id));

            for (lower_bound, cluster_id) in repair_candidates {
                if extra_clusters_scanned >= self.repair_max_extra_clusters {
                    break;
                }
                if lower_bound > worst_dist {
                    break;
                }

                let cluster_id = cluster_id as usize;
                let start = self.offsets[cluster_id] as usize;
                let end = self.offsets[cluster_id + 1] as usize;
                let cluster_rows = end - start;
                if extra_rows_scanned + cluster_rows > self.repair_max_extra_rows {
                    break;
                }

                scanned[cluster_id] = true;
                #[cfg(test)]
                {
                    clusters_scanned += 1;
                    rows_scanned += cluster_rows;
                }
                extra_clusters_scanned += 1;
                extra_rows_scanned += cluster_rows;

                scan_rows(
                    &self.rows[start..end],
                    start as u32,
                    &q_cont,
                    &pd,
                    #[cfg(target_arch = "x86_64")]
                    q_simd,
                    #[cfg(target_arch = "x86_64")]
                    zero_lanes_mask,
                    &mut neighbors,
                    &mut worst_slot,
                    &mut worst_dist,
                );
            }
        }

        IvfSearchStats {
            score: fraud_votes(&neighbors) as f32 / 5.0,
            #[cfg(test)]
            refined: extra_clusters_scanned > 0,
            #[cfg(test)]
            primary_fraud_votes,
            #[cfg(test)]
            clusters_scanned,
            #[cfg(test)]
            rows_scanned,
            #[cfg(test)]
            extra_clusters_scanned,
            #[cfg(test)]
            extra_rows_scanned,
            #[cfg(test)]
            elapsed_nanos: started.elapsed().as_nanos(),
        }
    }

    #[cfg(test)]
    fn search_exact(&self, q: &[f32; 14]) -> f32 {
        let q_cont = query_cont_bytes(q);
        let pd = PartialDists::compute(q);

        let mut neighbors: [Neighbor; 5] = [Neighbor::empty(); 5];
        let (mut worst_slot, mut worst_dist) = (0usize, u64::MAX);

        #[cfg(target_arch = "x86_64")]
        let (q_simd, zero_lanes_mask) = unsafe {
            use std::arch::x86_64::*;
            let q_v = simd::load_m128(&q_cont);
            let mask = _mm_set_epi32(0, 0, -1i32, -1i32);
            (q_v, mask)
        };

        scan_rows(
            &self.rows,
            0,
            &q_cont,
            &pd,
            #[cfg(target_arch = "x86_64")]
            q_simd,
            #[cfg(target_arch = "x86_64")]
            zero_lanes_mask,
            &mut neighbors,
            &mut worst_slot,
            &mut worst_dist,
        );

        fraud_votes(&neighbors) as f32 / 5.0
    }
}

struct IvfSearchStats {
    score: f32,
    #[cfg(test)]
    refined: bool,
    #[cfg(test)]
    primary_fraud_votes: usize,
    #[cfg(test)]
    clusters_scanned: usize,
    #[cfg(test)]
    rows_scanned: usize,
    #[cfg(test)]
    extra_clusters_scanned: usize,
    #[cfg(test)]
    extra_rows_scanned: usize,
    #[cfg(test)]
    elapsed_nanos: u128,
}

#[derive(Clone, Copy)]
struct Neighbor {
    dist: u64,
    id: u32,
    is_fraud: bool,
}

impl Neighbor {
    const fn empty() -> Self {
        Self {
            dist: u64::MAX,
            id: u32::MAX,
            is_fraud: false,
        }
    }
}

fn fraud_votes(neighbors: &[Neighbor; 5]) -> usize {
    neighbors.iter().filter(|n| n.is_fraud).count()
}

fn is_ambiguous(fraud_votes: usize) -> bool {
    fraud_votes == 2 || fraud_votes == 3
}

/// Shared inner scan loop used by IVF and exact-search tests.
/// Scans `rows` computing distance = SIMD_cont + table_lookup_discrete + binary,
/// updating the top-5 neighbor tracker.
#[inline(always)]
fn scan_rows(
    rows: &[[u8; 16]],
    base_id: u32,
    #[cfg(not(target_arch = "x86_64"))] q_cont: &[u8; 16],
    #[cfg(target_arch = "x86_64")] _q_cont: &[u8; 16],
    pd: &PartialDists,
    #[cfg(target_arch = "x86_64")] q_simd: std::arch::x86_64::__m128i,
    #[cfg(target_arch = "x86_64")] zero_lanes_mask: std::arch::x86_64::__m128i,
    neighbors: &mut [Neighbor; 5],
    worst_slot: &mut usize,
    worst_dist: &mut u64,
) {
    for (offset, row) in rows.iter().enumerate() {
        #[cfg(target_arch = "x86_64")]
        let cont = unsafe {
            use std::arch::x86_64::*;
            let r_masked = _mm_and_si128(simd::load_m128(row), zero_lanes_mask);
            simd::dist_cont(q_simd, r_masked) as u64
        };

        #[cfg(not(target_arch = "x86_64"))]
        let cont = {
            let r_buf: &[u8; 12] = row[0..12].try_into().unwrap();
            simd::dist_cont_scalar(q_cont, r_buf) as u64
        };

        let bits = [row[12], row[13], row[14]];
        let (i1, i3, i4, i8, i12, b9, b10, b11) = unpack_bits(&bits);

        let discrete: u64 = pd.d1[i1] as u64
            + pd.d3[i3] as u64
            + pd.d4[i4] as u64
            + pd.d8[i8] as u64
            + pd.d12[i12] as u64;

        let binary: u64 = bit_sq(b9, pd.q9) + bit_sq(b10, pd.q10) + bit_sq(b11, pd.q11);
        let dist = cont + discrete + binary;
        let id = base_id + offset as u32;

        if dist < *worst_dist || (dist == *worst_dist && id < neighbors[*worst_slot].id) {
            neighbors[*worst_slot] = Neighbor {
                dist,
                id,
                is_fraud: row[15] != 0,
            };
            (*worst_slot, *worst_dist) = find_worst(neighbors);
        }
    }
}

#[inline(always)]
fn bit_sq(r: u8, q: u8) -> u64 {
    if r == q {
        0
    } else {
        BIT_DIST_SQ
    }
}

fn find_worst(neighbors: &[Neighbor; 5]) -> (usize, u64) {
    let mut worst_slot = 0usize;
    let mut worst = neighbors[0];
    for (i, &neighbor) in neighbors.iter().enumerate().skip(1) {
        if neighbor.dist > worst.dist || (neighbor.dist == worst.dist && neighbor.id > worst.id) {
            worst_slot = i;
            worst = neighbor;
        }
    }
    (worst_slot, worst.dist)
}

fn quantized_search_vector(q: &[f32; 14]) -> [i16; 14] {
    [
        quantize(q[0]),
        quantize(q[1]),
        quantize(q[2]),
        quantize(q[3]),
        quantize(q[4]),
        quantize(q[5]),
        quantize(q[6]),
        quantize(q[7]),
        quantize(q[8]),
        if q[9] > 0.5 { 8192 } else { 0 },
        if q[10] > 0.5 { 8192 } else { 0 },
        if q[11] > 0.5 { 8192 } else { 0 },
        quantize(q[12]),
        quantize(q[13]),
    ]
}

fn bbox_lower_bound_sq(q: &[i16; 14], mins: &[i16; 14], maxes: &[i16; 14]) -> u64 {
    let mut sum = 0u64;
    for d in 0..14 {
        let qd = q[d] as i32;
        let diff = if qd < mins[d] as i32 {
            mins[d] as i32 - qd
        } else if qd > maxes[d] as i32 {
            qd - maxes[d] as i32
        } else {
            0
        };
        sum += (diff * diff) as u64;
    }
    sum
}

#[inline(always)]
fn l2_f32(a: &[f32; 14], b: &[f32; 14]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..14 {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed_ref::pack_row_for_test;

    fn make_index(entries: &[([f32; 14], bool)]) -> FraudIndex {
        let rows = entries
            .iter()
            .map(|(v, is_fraud)| pack_row_for_test(v, *is_fraud))
            .collect();
        FraudIndex {
            rows,
            centroids: Vec::new(),
            offsets: Vec::new(),
            bbox_min: Vec::new(),
            bbox_max: Vec::new(),
            primary_nprobe: DEFAULT_PRIMARY_NPROBE,
            refine_nprobe: DEFAULT_REFINE_NPROBE,
            ivf_repair: true,
            repair_max_extra_clusters: DEFAULT_REPAIR_MAX_EXTRA_CLUSTERS,
            repair_max_extra_rows: DEFAULT_REPAIR_MAX_EXTRA_ROWS,
        }
    }

    const ZERO: [f32; 14] = [0.0; 14];

    #[test]
    fn search_all_fraud() {
        let idx = make_index(&[
            (ZERO, true),
            (ZERO, true),
            (ZERO, true),
            (ZERO, true),
            (ZERO, true),
        ]);
        assert_eq!(idx.search_exact(&ZERO), 1.0);
    }

    #[test]
    fn search_all_legit() {
        let idx = make_index(&[
            (ZERO, false),
            (ZERO, false),
            (ZERO, false),
            (ZERO, false),
            (ZERO, false),
        ]);
        assert_eq!(idx.search_exact(&ZERO), 0.0);
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
        assert_eq!(idx.search_exact(&ZERO), 0.6);
    }

    #[test]
    fn search_nearest_neighbors_win() {
        let close = [0.01f32; 14];
        let far = [0.9f32; 14];
        let idx = make_index(&[
            (close, true),
            (close, true),
            (close, true),
            (close, true),
            (close, true),
            (far, false),
        ]);
        assert_eq!(idx.search_exact(&ZERO), 1.0);
    }

    #[test]
    fn bbox_lower_bound_zero_when_query_inside_box() {
        let q = [10i16; 14];
        let mins = [0i16; 14];
        let maxes = [20i16; 14];

        assert_eq!(bbox_lower_bound_sq(&q, &mins, &maxes), 0);
    }

    #[test]
    fn bbox_lower_bound_uses_nearest_bound_below_and_above() {
        let mut q = [15i16; 14];
        let mins = [10i16; 14];
        let maxes = [20i16; 14];
        q[0] = 7;
        q[1] = 25;

        assert_eq!(bbox_lower_bound_sq(&q, &mins, &maxes), 9 + 25);
    }

    #[test]
    fn bbox_lower_bound_sums_multiple_dimensions() {
        let mut q = [15i16; 14];
        let mins = [10i16; 14];
        let maxes = [20i16; 14];
        q[0] = 5;
        q[2] = 24;
        q[3] = 15;

        assert_eq!(bbox_lower_bound_sq(&q, &mins, &maxes), 25 + 16);
    }

    #[test]
    fn bbox_lower_bound_is_conservative_for_point_in_box() {
        let q = [0i16; 14];
        let point = [10i16; 14];
        let mins = point;
        let maxes = point;

        let exact = point
            .iter()
            .map(|&v| {
                let d = v as i32;
                (d * d) as u64
            })
            .sum::<u64>();

        assert!(bbox_lower_bound_sq(&q, &mins, &maxes) <= exact);
    }

    fn make_two_cluster_ivf_index(
        primary_entries: &[([f32; 14], bool)],
        repair_entries: &[([f32; 14], bool)],
        repair_max_extra_clusters: usize,
        repair_max_extra_rows: usize,
    ) -> FraudIndex {
        let mut rows = Vec::new();
        rows.extend(
            primary_entries
                .iter()
                .map(|(v, is_fraud)| pack_row_for_test(v, *is_fraud)),
        );
        rows.extend(
            repair_entries
                .iter()
                .map(|(v, is_fraud)| pack_row_for_test(v, *is_fraud)),
        );

        let mut centroids = vec![[1.0f32; 14]; K];
        centroids[0] = [0.0f32; 14];
        centroids[1] = [1.0f32; 14];

        let primary_len = primary_entries.len() as u32;
        let total_len = rows.len() as u32;
        let mut offsets = vec![total_len; K + 1];
        offsets[0] = 0;
        offsets[1] = primary_len;
        offsets[2] = total_len;

        let mut bbox_min = vec![[0i16; 14]; K];
        let mut bbox_max = vec![[0i16; 14]; K];
        set_bbox(&mut bbox_min[0], &mut bbox_max[0], primary_entries);
        set_bbox(&mut bbox_min[1], &mut bbox_max[1], repair_entries);

        FraudIndex {
            rows,
            centroids,
            offsets,
            bbox_min,
            bbox_max,
            primary_nprobe: 1,
            refine_nprobe: 2,
            ivf_repair: true,
            repair_max_extra_clusters,
            repair_max_extra_rows,
        }
    }

    fn set_bbox(mins: &mut [i16; 14], maxes: &mut [i16; 14], entries: &[([f32; 14], bool)]) {
        *mins = [i16::MAX; 14];
        *maxes = [i16::MIN; 14];
        for (v, _) in entries {
            let qv = quantized_search_vector(v);
            for d in 0..14 {
                mins[d] = mins[d].min(qv[d]);
                maxes[d] = maxes[d].max(qv[d]);
            }
        }
    }

    fn ambiguous_primary_entries() -> Vec<([f32; 14], bool)> {
        let far = [0.9f32; 14];
        vec![
            (far, true),
            (far, true),
            (far, false),
            (far, false),
            (far, false),
        ]
    }

    fn near_repair_entries() -> Vec<([f32; 14], bool)> {
        vec![
            (ZERO, true),
            (ZERO, true),
            (ZERO, true),
            (ZERO, true),
            (ZERO, true),
        ]
    }

    #[test]
    fn clear_primary_vote_skips_repair() {
        let far = [0.9f32; 14];
        let primary = vec![(far, false); 5];
        let repair = near_repair_entries();
        let idx = make_two_cluster_ivf_index(
            &primary,
            &repair,
            DEFAULT_REPAIR_MAX_EXTRA_CLUSTERS,
            DEFAULT_REPAIR_MAX_EXTRA_ROWS,
        );

        let stats = idx.search_ivf_decision_aware(&ZERO, 1, 2, true);

        assert_eq!(stats.score, 0.0);
        assert_eq!(stats.primary_fraud_votes, 0);
        assert!(!stats.refined);
        assert_eq!(stats.extra_clusters_scanned, 0);
        assert_eq!(stats.extra_rows_scanned, 0);
    }

    #[test]
    fn ambiguous_primary_vote_triggers_refine() {
        let primary = ambiguous_primary_entries();
        let repair = near_repair_entries();
        let idx = make_two_cluster_ivf_index(
            &primary,
            &repair,
            DEFAULT_REPAIR_MAX_EXTRA_CLUSTERS,
            DEFAULT_REPAIR_MAX_EXTRA_ROWS,
        );

        let pure = idx.search_ivf_decision_aware(&ZERO, 1, 2, false);
        let repaired = idx.search_ivf_decision_aware(&ZERO, 1, 2, true);

        assert_eq!(pure.score, 0.4);
        assert_eq!(repaired.primary_fraud_votes, 2);
        assert!(repaired.refined);
        assert_eq!(repaired.score, idx.search_exact(&ZERO));
        assert_eq!(repaired.score, 1.0);
        assert!(repaired.clusters_scanned > pure.clusters_scanned);
    }

    #[test]
    fn row_budget_stops_repair() {
        let primary = ambiguous_primary_entries();
        let repair = near_repair_entries();
        let idx = make_two_cluster_ivf_index(&primary, &repair, 1, repair.len() - 1);

        let stats = idx.search_ivf_decision_aware(&ZERO, 1, 2, true);

        assert!(!stats.refined);
        assert_eq!(stats.extra_clusters_scanned, 0);
        assert_eq!(stats.extra_rows_scanned, 0);
        assert_eq!(stats.score, 0.4);
    }

    #[test]
    fn cluster_cap_stops_repair() {
        let primary = ambiguous_primary_entries();
        let repair = near_repair_entries();
        let idx = make_two_cluster_ivf_index(&primary, &repair, 0, DEFAULT_REPAIR_MAX_EXTRA_ROWS);

        let stats = idx.search_ivf_decision_aware(&ZERO, 1, 2, true);

        assert!(!stats.refined);
        assert_eq!(stats.extra_clusters_scanned, 0);
        assert_eq!(stats.score, 0.4);
    }

    #[test]
    fn equal_distance_neighbors_keep_lower_stable_ids() {
        let idx = make_index(&[
            (ZERO, true),
            (ZERO, true),
            (ZERO, true),
            (ZERO, false),
            (ZERO, false),
            (ZERO, false),
        ]);

        assert_eq!(idx.search_exact(&ZERO), 0.6);
    }

    /// IVF recall benchmark: compare IVF at various nprobe values against
    /// exact-search ground truth on the full compiled-in dataset.
    /// Run with: cargo test --release -- --nocapture recall_benchmark
    #[test]
    #[ignore = "benchmark-style test over the full embedded dataset"]
    fn recall_benchmark() {
        let idx = FraudIndex::build();

        let mut queries: Vec<[f32; 14]> = Vec::new();

        for dim in 0..14 {
            for &val in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
                let mut q = [0.5f32; 14];
                q[dim] = val;
                queries.push(q);
            }
        }

        queries.push([0.0; 14]);
        queries.push([1.0; 14]);
        queries.push([0.5; 14]);

        let mut seed: u64 = 42;
        for _ in 0..500 {
            let mut q = [0.0f32; 14];
            for value in &mut q {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *value = ((seed >> 33) as f32) / (u32::MAX as f32 / 2.0);
                *value = value.clamp(0.0, 1.0);
            }
            queries.push(q);
        }

        let n = queries.len();

        eprintln!("\nComputing exact-search ground truth for {} queries...", n);
        let bf_scores: Vec<f32> = queries.iter().map(|q| idx.search_exact(q)).collect();
        eprintln!("Ground truth computed.");

        let row_budgets = [80_000usize, 120_000, 180_000];

        eprintln!("\n=== IVF Recall Benchmark ({} queries, K={}) ===", n, K);
        eprintln!(
            "{:>10} {:>12} {:>12} {:>11} {:>14} {:>12} {:>12} {:>12}",
            "extra_rows",
            "score_match",
            "decision_eq",
            "refine_rate",
            "avg_extra_rows",
            "avg_clusters",
            "avg_rows",
            "p99_us"
        );

        for &row_budget in &row_budgets {
            let mut idx = FraudIndex::build();
            idx.repair_max_extra_rows = row_budget;

            let mut score_matches = 0usize;
            let mut decision_matches = 0usize;
            let mut refined = 0usize;
            let mut clusters_scanned = 0usize;
            let mut rows_scanned = 0usize;
            let mut extra_rows_scanned = 0usize;
            let mut latencies = Vec::with_capacity(n);

            for (i, q) in queries.iter().enumerate() {
                let stats = idx.search_ivf_decision_aware(q, 4, 24, true);
                let ivf_score = stats.score;
                refined += stats.refined as usize;
                clusters_scanned += stats.clusters_scanned;
                rows_scanned += stats.rows_scanned;
                extra_rows_scanned += stats.extra_rows_scanned;
                latencies.push(stats.elapsed_nanos);

                if (bf_scores[i] - ivf_score).abs() < 1e-6 {
                    score_matches += 1;
                }
                if (bf_scores[i] < 0.6) == (ivf_score < 0.6) {
                    decision_matches += 1;
                }
            }

            latencies.sort_unstable();
            let p99_idx = ((n as f64 * 0.99).ceil() as usize).saturating_sub(1);
            let p99_us = latencies[p99_idx] as f64 / 1_000.0;

            eprintln!(
                "{:>10} {:>11.2}% {:>11.2}% {:>10.2}% {:>14.0} {:>12.1} {:>12.0} {:>12.1}",
                row_budget,
                score_matches as f64 * 100.0 / n as f64,
                decision_matches as f64 * 100.0 / n as f64,
                refined as f64 * 100.0 / n as f64,
                extra_rows_scanned as f64 / n as f64,
                clusters_scanned as f64 / n as f64,
                rows_scanned as f64 / n as f64,
                p99_us
            );
        }
    }
}
