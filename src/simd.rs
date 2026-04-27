// SIMD distance computation for the 6 continuous dimensions.
//
// The competition runs on x86_64 (Mac Mini Intel 2014, Haswell). On that
// target we use SSE2 integer intrinsics which are baseline for all x86_64
// CPUs — no runtime feature detection needed. On other architectures (e.g.
// Apple Silicon during local development) we fall back to a plain scalar loop.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Load 8 i16 values from a 16-byte buffer into a 128-bit SIMD register.
///
/// # Safety
/// `buf` must be a valid 16-byte slice. Alignment is not required: `_mm_loadu_si128`
/// accepts unaligned addresses. The two padding slots (bytes 12–15) are loaded
/// but their lanes (6, 7) are always zero in both query and reference, so they
/// contribute 0 to the squared distance.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn load_m128(buf: &[u8; 16]) -> __m128i {
    // SAFETY: buf is 16 bytes; _mm_loadu_si128 reads exactly 16 bytes from
    // the pointer without requiring alignment.
    _mm_loadu_si128(buf.as_ptr() as *const __m128i)
}

/// Compute the squared distance between two sets of 6 continuous i16 dimensions
/// using SSE2 integer instructions.
///
/// Both `q` and `r` are 128-bit registers holding 8 i16 lanes each. Lanes 0–5
/// carry the continuous dimensions; lanes 6–7 are zero-padded and contribute 0.
///
/// # Safety
/// Caller must have verified (or compiled for) x86_64 — SSE2 is always present
/// on x86_64. `q` and `r` must have been loaded with `load_m128`.
///
/// # Instruction breakdown
/// 1. `_mm_sub_epi16`  — subtract each i16 lane: diff[k] = q[k] - r[k]
/// 2. `_mm_madd_epi16` — multiply adjacent pairs and accumulate to i32:
///                       result[k] = diff[2k]*diff[2k] + diff[2k+1]*diff[2k+1]
///                       This gives 4 partial sums (i32) for 8 input lanes.
/// 3. Two `_mm_add_epi32` + `_mm_srli_si128` steps fold the 4 i32 lanes into 1.
/// 4. `_mm_cvtsi128_si32` extracts the scalar result from lane 0.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
pub unsafe fn dist_cont(q: __m128i, r: __m128i) -> u32 {
    let diff = _mm_sub_epi16(q, r);
    let sq   = _mm_madd_epi16(diff, diff);

    // Horizontal sum of 4 i32 lanes.
    let s = _mm_add_epi32(sq, _mm_srli_si128(sq, 8));
    let s = _mm_add_epi32(s,  _mm_srli_si128(s,  4));

    _mm_cvtsi128_si32(s) as u32
}

// Used on non-x86_64 hosts (Apple Silicon dev machines). Functionally identical
// to the SIMD path: reads 6 i16 values from the 12-byte prefix of each buffer.
#[cfg(not(target_arch = "x86_64"))]
pub fn dist_cont_scalar(q_buf: &[u8; 16], r_buf: &[u8; 12]) -> u32 {
    let mut sum = 0i64;
    for i in 0..6 {
        let qv = i16::from_ne_bytes([q_buf[i * 2], q_buf[i * 2 + 1]]) as i64;
        let rv = i16::from_ne_bytes([r_buf[i * 2], r_buf[i * 2 + 1]]) as i64;
        let d = qv - rv;
        sum += d * d;
    }
    sum as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a 16-byte query buffer with a single i16 value at `slot` (0-based),
    // all other slots zero.
    fn q_buf_with(slot: usize, val: i16) -> [u8; 16] {
        let mut buf = [0u8; 16];
        buf[slot * 2..slot * 2 + 2].copy_from_slice(&val.to_ne_bytes());
        buf
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn r_buf_from(q: &[u8; 16]) -> [u8; 12] {
        q[0..12].try_into().unwrap()
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[test]
    fn scalar_identical_vectors_give_zero() {
        let q = q_buf_with(0, 100);
        let r = r_buf_from(&q);
        assert_eq!(dist_cont_scalar(&q, &r), 0);
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[test]
    fn scalar_single_dim_delta() {
        // q[slot 0] = 10, r[slot 0] = 7 → diff = 3, sq = 9
        let q = q_buf_with(0, 10);
        let mut r = [0u8; 12];
        r[0..2].copy_from_slice(&7i16.to_ne_bytes());
        assert_eq!(dist_cont_scalar(&q, &r), 9);
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[test]
    fn scalar_bytes_12_15_are_ignored() {
        // Bytes 12-15 carry packed bits and a label; the scalar path only reads
        // the first 12 bytes via r_buf, so those bytes must not contribute.
        let mut q = q_buf_with(0, 0);
        q[12] = 0xFF; // would add noise if misread as a distance dimension
        q[13] = 0xFF;
        q[14] = 0xFF;
        q[15] = 0xFF;
        let r = [0u8; 12];
        assert_eq!(dist_cont_scalar(&q, &r), 0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_identical_vectors_give_zero() {
        let buf = [0u8; 16];
        let dist = unsafe {
            let q = load_m128(&buf);
            let r = load_m128(&buf);
            dist_cont(q, r)
        };
        assert_eq!(dist, 0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_single_dim_delta() {
        // q[slot 0] = 10, r[slot 0] = 7 → diff = 3, sq = 9
        let q = q_buf_with(0, 10);
        let mut r_buf = [0u8; 16];
        r_buf[0..2].copy_from_slice(&7i16.to_ne_bytes());

        // Zero out bytes 12-15 to simulate the mask applied in index.rs.
        // Without the mask those bytes would be loaded as two extra i16 lanes
        // and corrupt the sum; here we zero them in the buffer directly.
        let dist = unsafe {
            let q_v = load_m128(&q);
            let r_v = load_m128(&r_buf);
            dist_cont(q_v, r_v)
        };
        assert_eq!(dist, 9);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_all_six_dims_contribute() {
        // Each continuous slot (0-5) has diff = 1 → total sq_dist = 6.
        // Slots 6-7 are zero in both buffers, so they add 0.
        let mut q_buf = [0u8; 16];
        let r_buf    = [0u8; 16];
        for slot in 0..6 {
            q_buf[slot * 2..slot * 2 + 2].copy_from_slice(&1i16.to_ne_bytes());
        }
        let dist = unsafe {
            let q = load_m128(&q_buf);
            let r = load_m128(&r_buf);
            dist_cont(q, r)
        };
        assert_eq!(dist, 6);
    }
}
