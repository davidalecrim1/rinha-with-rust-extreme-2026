use crate::packed_ref::{query_cont_bytes, unpack_bits, PartialDists};
use crate::simd;

// 16-byte row layout (matches build.rs output):
//   bytes  0–11 : [i16; 6] continuous dims (0, 2, 5, 6, 7, 13), native-endian
//   bytes 12–14 : [u8; 3]  packed low-cardinality indices + binary flags
//   byte  15    : u8        fraud label (0 = legit, 1 = fraud)
const ROW: usize = 16;

// When two binary dims differ (one is 0, the other is 8192 quantized), the
// squared distance is (8192 - 0)^2 = 67_108_864.
const BIT_DIST_SQ: u64 = 8192 * 8192;

pub struct FraudIndex {
    rows: Vec<[u8; 16]>,
}

impl FraudIndex {
    /// Load the pre-packed binary blob produced by build.rs.
    /// No decompression, no JSON parsing — single memcpy from the binary's
    /// read-only data segment.
    pub fn build() -> Self {
        let raw: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/packed_refs.bin"));

        assert!(
            raw.len() % ROW == 0,
            "packed_refs.bin size {} is not a multiple of {ROW}",
            raw.len()
        );

        let mut rows: Vec<[u8; 16]> = Vec::with_capacity(raw.len() / ROW);
        for chunk in raw.chunks_exact(ROW) {
            // SAFETY: chunk is exactly ROW (16) bytes from a valid slice.
            rows.push(chunk.try_into().unwrap());
        }

        FraudIndex { rows }
    }

    /// Returns the fraud score for query vector `q` using k-NN with k=5.
    ///
    /// Precomputes partial distances for all low-cardinality dictionary entries
    /// (~208 bytes, stays in L1 cache), then scans all 100K rows computing
    /// distance = SIMD_cont_dist + table_lookup_discrete_dist + binary_dist.
    pub fn search(&self, q: &[f32; 14]) -> f32 {
        let q_cont = query_cont_bytes(q);
        let pd = PartialDists::compute(q);

        #[cfg(target_arch = "x86_64")]
        // SAFETY: q_cont is a valid 16-byte buffer; SSE2 is present on all x86_64.
        let (q_simd, zero_lanes_mask) = unsafe {
            use std::arch::x86_64::*;
            let q_v = simd::load_m128(&q_cont);
            // Bytes 12-15 of each row hold packed bits and a label, not distance
            // data. This mask zeroes those bytes (SIMD lanes 6-7) so they don't
            // corrupt the squared distance sum.
            let mask = _mm_set_epi32(0, 0, -1i32, -1i32);
            (q_v, mask)
        };

        let mut neighbors: [(u64, bool); 5] = [(u64::MAX, false); 5];
        let (mut worst_slot, mut worst_dist) = (0usize, u64::MAX);

        for row in &self.rows {
            #[cfg(target_arch = "x86_64")]
            let cont = unsafe {
                use std::arch::x86_64::*;
                // Load the full 16-byte row. Bytes 12-15 land in SIMD lanes 6-7;
                // we zero them with the precomputed mask before computing distance.
                //
                // SAFETY: row is a valid 16-byte buffer; SSE2 is guaranteed on x86_64.
                let r_masked = _mm_and_si128(simd::load_m128(row), zero_lanes_mask);
                simd::dist_cont(q_simd, r_masked) as u64
            };

            #[cfg(not(target_arch = "x86_64"))]
            let cont = {
                let r_buf: &[u8; 12] = row[0..12].try_into().unwrap();
                simd::dist_cont_scalar(&q_cont, r_buf) as u64
            };

            let bits = [row[12], row[13], row[14]];
            let (i1, i3, i4, i8, i12, b9, b10, b11) = unpack_bits(&bits);

            // Table lookup replaces arithmetic for 8 of the 14 dimensions.
            let discrete: u64 = pd.d1[i1] as u64
                + pd.d3[i3] as u64
                + pd.d4[i4] as u64
                + pd.d8[i8] as u64
                + pd.d12[i12] as u64;

            let binary: u64 = bit_sq(b9, pd.q9)
                + bit_sq(b10, pd.q10)
                + bit_sq(b11, pd.q11);

            let dist = cont + discrete + binary;

            if dist < worst_dist {
                let is_fraud = row[15] != 0;
                neighbors[worst_slot] = (dist, is_fraud);
                (worst_slot, worst_dist) = find_worst(&neighbors);
            }
        }

        neighbors.iter().filter(|(_, f)| *f).count() as f32 / 5.0
    }
}

/// Returns BIT_DIST_SQ when binary dims differ, 0 when equal.
#[inline(always)]
fn bit_sq(r: u8, q: u8) -> u64 {
    if r == q { 0 } else { BIT_DIST_SQ }
}

/// Returns the index and distance of the farthest neighbor in the top-5.
fn find_worst(neighbors: &[(u64, bool); 5]) -> (usize, u64) {
    neighbors
        .iter()
        .enumerate()
        .fold((0, 0), |(wi, wd), (i, &(d, _))| {
            if d > wd { (i, d) } else { (wi, wd) }
        })
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
        FraudIndex { rows }
    }

    const ZERO: [f32; 14] = [0.0; 14];

    #[test]
    fn search_all_fraud() {
        let idx = make_index(&[
            (ZERO, true), (ZERO, true), (ZERO, true), (ZERO, true), (ZERO, true),
        ]);
        assert_eq!(idx.search(&ZERO), 1.0);
    }

    #[test]
    fn search_all_legit() {
        let idx = make_index(&[
            (ZERO, false), (ZERO, false), (ZERO, false), (ZERO, false), (ZERO, false),
        ]);
        assert_eq!(idx.search(&ZERO), 0.0);
    }

    #[test]
    fn search_mixed_3_fraud_2_legit() {
        let idx = make_index(&[
            (ZERO, true), (ZERO, true), (ZERO, true), (ZERO, false), (ZERO, false),
        ]);
        assert_eq!(idx.search(&ZERO), 0.6);
    }

    #[test]
    fn search_nearest_neighbors_win() {
        // 5 fraud vectors close to origin; 1 legit far away — top-5 must all be fraud.
        let close = [0.01f32; 14];
        let far   = [0.9f32; 14];
        let idx = make_index(&[
            (close, true), (close, true), (close, true), (close, true), (close, true),
            (far, false),
        ]);
        assert_eq!(idx.search(&ZERO), 1.0);
    }
}
