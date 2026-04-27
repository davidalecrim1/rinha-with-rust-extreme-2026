// Runs at `cargo build` time. Decompresses resources/references.json.gz,
// parses the 100K labeled vectors, quantizes every value to i16 (scale × 8192),
// and packs each row into exactly 16 bytes. The resulting binary blob is written
// to $OUT_DIR/packed_refs.bin and a Rust source file with the dictionary
// constants to $OUT_DIR/dicts.rs. At runtime, index.rs loads the blob via
// include_bytes! — zero I/O, zero JSON parsing, zero decompression.

use flate2::read::GzDecoder;
use serde::Deserialize;
use std::io::Read;
use std::path::PathBuf;

#[derive(Deserialize)]
struct RefEntry {
    vector: [f32; 14],
    label: String,
}

const SCALE: f32 = 8192.0;

fn quantize(v: f32) -> i16 {
    (v * SCALE)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

// Dictionaries are derived analytically from the vectorization formulas, not
// from the dataset, so they are always correct regardless of which reference
// dataset is used:
//   dim 1  (installments)  : k/12   for k in 0..=12   → 13 entries, 4 bits
//   dim 3  (hour of day)   : k/23   for k in 0..=23   → 24 entries, 5 bits
//   dim 4  (day of week)   : k/6    for k in 0..=6    → 7 entries,  3 bits
//   dim 8  (tx_count_24h)  : k/20   for k in 0..=20   → 21 entries, 5 bits
//   dim 12 (mcc_risk)      : 10 fixed values from mcc_risk.json  → 4 bits

fn dict_dim1() -> Vec<i16> {
    (0u32..=12).map(|k| quantize((k as f32 / 12.0).clamp(0.0, 1.0))).collect()
}

fn dict_dim3() -> Vec<i16> {
    (0u32..=23).map(|k| quantize(k as f32 / 23.0)).collect()
}

fn dict_dim4() -> Vec<i16> {
    (0u32..=6).map(|k| quantize(k as f32 / 6.0)).collect()
}

fn dict_dim8() -> Vec<i16> {
    (0u32..=20).map(|k| quantize((k as f32 / 20.0).clamp(0.0, 1.0))).collect()
}

fn dict_dim12() -> Vec<i16> {
    // Distinct values present in resources/mcc_risk.json.
    // Default when MCC is unknown is 0.5, which is already in the list.
    [0.15f32, 0.20, 0.25, 0.30, 0.35, 0.45, 0.50, 0.75, 0.80, 0.85]
        .iter()
        .map(|&v| quantize(v))
        .collect()
}

fn find_dict_index(dict: &[i16], val: i16) -> u8 {
    dict.iter()
        .enumerate()
        .min_by_key(|(_, &d)| (d as i32 - val as i32).abs())
        .map(|(i, _)| i as u8)
        .unwrap()
}

// 24-bit layout (3 bytes), LSB first — mirrors unpack_bits in src/packed_ref.rs:
//   bits  0– 3  (4 bits) : dim 1  index  (installments, 0-12)
//   bits  4– 8  (5 bits) : dim 3  index  (hour, 0-23)
//   bits  9–11  (3 bits) : dim 4  index  (dow, 0-6)
//   bits 12–16  (5 bits) : dim 8  index  (tx_count, 0-20)
//   bits 17–20  (4 bits) : dim 12 index  (mcc_risk, 0-9)
//   bit  21     (1 bit)  : dim 9  binary (is_online)
//   bit  22     (1 bit)  : dim 10 binary (card_present)
//   bit  23     (1 bit)  : dim 11 binary (unknown_merchant)
fn pack_bits(i1: u8, i3: u8, i4: u8, i8: u8, i12: u8, b9: u8, b10: u8, b11: u8) -> [u8; 3] {
    let bits: u32 = (i1 as u32)
        | ((i3 as u32) << 4)
        | ((i4 as u32) << 9)
        | ((i8 as u32) << 12)
        | ((i12 as u32) << 17)
        | ((b9 as u32) << 21)
        | ((b10 as u32) << 22)
        | ((b11 as u32) << 23);
    [bits as u8, (bits >> 8) as u8, (bits >> 16) as u8]
}

// 16-byte row layout — mirrors index.rs:
//   bytes  0–11 : [i16; 6] continuous dims (0, 2, 5, 6, 7, 13), native-endian
//   bytes 12–14 : [u8; 3]  packed low-cardinality indices + binary flags
//   byte  15    : u8        fraud label (0 = legit, 1 = fraud)
fn pack_row(
    v: &[f32; 14],
    d1: &[i16], d3: &[i16], d4: &[i16], d8: &[i16], d12: &[i16],
    is_fraud: bool,
) -> [u8; 16] {
    let c0  = quantize(v[0]);
    let c2  = quantize(v[2]);
    let c5  = quantize(v[5]);
    let c6  = quantize(v[6]);
    let c7  = quantize(v[7]);
    let c13 = quantize(v[13]);

    let i1  = find_dict_index(d1,  quantize(v[1]));
    let i3  = find_dict_index(d3,  quantize(v[3]));
    let i4  = find_dict_index(d4,  quantize(v[4]));
    let i8  = find_dict_index(d8,  quantize(v[8]));
    let i12 = find_dict_index(d12, quantize(v[12]));

    let b9  = (v[9]  > 0.5) as u8;
    let b10 = (v[10] > 0.5) as u8;
    let b11 = (v[11] > 0.5) as u8;

    let bits = pack_bits(i1, i3, i4, i8, i12, b9, b10, b11);

    let mut row = [0u8; 16];
    row[0..2].copy_from_slice(&c0.to_ne_bytes());
    row[2..4].copy_from_slice(&c2.to_ne_bytes());
    row[4..6].copy_from_slice(&c5.to_ne_bytes());
    row[6..8].copy_from_slice(&c6.to_ne_bytes());
    row[8..10].copy_from_slice(&c7.to_ne_bytes());
    row[10..12].copy_from_slice(&c13.to_ne_bytes());
    row[12..15].copy_from_slice(&bits);
    row[15] = is_fraud as u8;
    row
}

fn write_dict(out: &mut String, name: &str, values: &[i16]) {
    use std::fmt::Write;
    let items: Vec<String> = values.iter().map(|v| v.to_string()).collect();
    writeln!(
        out,
        "pub const {}: [i16; {}] = [{}];",
        name,
        values.len(),
        items.join(", ")
    )
    .unwrap();
}

fn main() {
    println!("cargo:rerun-if-changed=resources/references.json.gz");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let d1  = dict_dim1();
    let d3  = dict_dim3();
    let d4  = dict_dim4();
    let d8  = dict_dim8();
    let d12 = dict_dim12();

    // Emit dicts.rs — include!()-d by src/packed_ref.rs.
    let mut dicts = String::new();
    write_dict(&mut dicts, "DICT01", &d1);
    write_dict(&mut dicts, "DICT03", &d3);
    write_dict(&mut dicts, "DICT04", &d4);
    write_dict(&mut dicts, "DICT08", &d8);
    write_dict(&mut dicts, "DICT12", &d12);
    std::fs::write(out_dir.join("dicts.rs"), &dicts)
        .expect("failed to write dicts.rs");

    let gz = std::fs::read("resources/references.json.gz")
        .expect("resources/references.json.gz not found");
    let mut decoder = GzDecoder::new(gz.as_slice());
    let mut json = String::new();
    decoder.read_to_string(&mut json).expect("failed to decompress references");

    let entries: Vec<RefEntry> =
        serde_json::from_str(&json).expect("failed to parse references JSON");

    let mut blob: Vec<u8> = Vec::with_capacity(entries.len() * 16);
    for e in &entries {
        let row = pack_row(&e.vector, &d1, &d3, &d4, &d8, &d12, e.label == "fraud");
        blob.extend_from_slice(&row);
    }

    std::fs::write(out_dir.join("packed_refs.bin"), &blob)
        .expect("failed to write packed_refs.bin");

    eprintln!(
        "cargo:warning=build.rs: packed {} references into {} bytes",
        entries.len(),
        blob.len()
    );
}
