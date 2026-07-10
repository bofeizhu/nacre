//! Deterministic dedup machinery: normalization, entropy gating, MinHash/LSH.
//!
//! Ports `graphiti_core/utils/maintenance/dedup_helpers.py` (pinned
//! v0.29.2). The MinHash signatures are bit-identical to Python's (blake2b
//! with 8-byte digests over `"{seed}:{shingle}"`), so resolution decisions
//! match the oracle exactly.

use std::collections::HashMap;

use blake2::digest::consts::U8;
use blake2::{Blake2b, Digest};

use super::ExistingNode;

pub(crate) const NAME_ENTROPY_THRESHOLD: f64 = 1.5;
pub(crate) const MIN_NAME_LENGTH: usize = 6;
pub(crate) const MIN_TOKEN_COUNT: usize = 2;
pub(crate) const FUZZY_JACCARD_THRESHOLD: f64 = 0.9;
const MINHASH_PERMUTATIONS: u64 = 32;
const MINHASH_BAND_SIZE: usize = 4;

/// Lowercase text and collapse whitespace so equal names map to the same key.
// ports: dedup_helpers.py::_normalize_string_exact
pub fn normalize_string_exact(name: &str) -> String {
    name.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Fuzzier form keeping ASCII alphanumerics and apostrophes for shingling.
// ports: dedup_helpers.py::_normalize_name_for_fuzzy
pub fn normalize_name_for_fuzzy(name: &str) -> String {
    let exact = normalize_string_exact(name);
    let replaced: String = exact
        .chars()
        .map(|c| {
            if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '\'' || c == ' ' {
                c
            } else {
                ' '
            }
        })
        .collect();
    replaced.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Approximate text specificity via Shannon entropy over characters
/// (spaces stripped).
// ports: dedup_helpers.py::_name_entropy
pub fn name_entropy(normalized_name: &str) -> f64 {
    let mut counts: HashMap<char, usize> = HashMap::new();
    for c in normalized_name.chars().filter(|c| *c != ' ') {
        *counts.entry(c).or_insert(0) += 1;
    }
    let total: usize = counts.values().sum();
    if total == 0 {
        return 0.0;
    }
    let total = total as f64;
    -counts
        .values()
        .map(|&count| {
            let p = count as f64 / total;
            p * p.log2()
        })
        .sum::<f64>()
}

/// Whether a name is reliable enough for fuzzy matching.
// ports: dedup_helpers.py::_has_high_entropy
pub fn has_high_entropy(normalized_name: &str) -> bool {
    let token_count = normalized_name.split_whitespace().count();
    if normalized_name.chars().count() < MIN_NAME_LENGTH && token_count < MIN_TOKEN_COUNT {
        return false;
    }
    name_entropy(normalized_name) >= NAME_ENTROPY_THRESHOLD
}

/// Character 3-gram shingles of the space-stripped name.
// ports: dedup_helpers.py::_shingles
pub fn shingles(normalized_name: &str) -> Vec<String> {
    let cleaned: Vec<char> = normalized_name.chars().filter(|c| *c != ' ').collect();
    if cleaned.len() < 2 {
        return if cleaned.is_empty() {
            Vec::new()
        } else {
            vec![cleaned.iter().collect()]
        };
    }
    let mut out: Vec<String> = cleaned
        .windows(3)
        .map(|window| window.iter().collect())
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Deterministic 64-bit hash for a shingle under a permutation seed —
/// bit-identical to Python's `blake2b(f'{seed}:{shingle}', digest_size=8)`.
// ports: dedup_helpers.py::_hash_shingle
fn hash_shingle(shingle: &str, seed: u64) -> u64 {
    let mut hasher = Blake2b::<U8>::new();
    hasher.update(format!("{seed}:{shingle}").as_bytes());
    u64::from_be_bytes(hasher.finalize().into())
}

/// MinHash signature across the fixed permutations; empty for no shingles.
// ports: dedup_helpers.py::_minhash_signature
pub fn minhash_signature(shingle_set: &[String]) -> Vec<u64> {
    if shingle_set.is_empty() {
        return Vec::new();
    }
    (0..MINHASH_PERMUTATIONS)
        .map(|seed| {
            shingle_set
                .iter()
                .map(|s| hash_shingle(s, seed))
                .min()
                .expect("non-empty shingle set")
        })
        .collect()
}

/// Split a signature into fixed-size LSH bands (partial bands dropped).
// ports: dedup_helpers.py::_lsh_bands
pub fn lsh_bands(signature: &[u64]) -> Vec<[u64; MINHASH_BAND_SIZE]> {
    signature
        .chunks_exact(MINHASH_BAND_SIZE)
        .map(|chunk| chunk.try_into().expect("chunks_exact yields full bands"))
        .collect()
}

/// Jaccard similarity between two shingle sets (both-empty counts as 1.0).
// ports: dedup_helpers.py::_jaccard_similarity
pub fn jaccard_similarity(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    // Inputs are sorted-deduped (see `shingles`).
    let intersection = a.iter().filter(|s| b.binary_search(s).is_ok()).count();
    let union = a.len() + b.len() - intersection;
    intersection as f64 / union as f64
}

/// Precomputed lookup structures for one dedup run.
// ports: dedup_helpers.py::DedupCandidateIndexes
pub struct CandidateIndexes {
    /// The candidate pool, in supplied order.
    pub existing_nodes: Vec<ExistingNode>,
    pub(crate) by_normalized_name: HashMap<String, Vec<usize>>,
    pub(crate) shingles_by_candidate: Vec<Vec<String>>,
    pub(crate) lsh_buckets: HashMap<(usize, [u64; MINHASH_BAND_SIZE]), Vec<usize>>,
}

/// Precompute exact and fuzzy lookup structures once per dedupe run.
// ports: dedup_helpers.py::_build_candidate_indexes
pub fn build_candidate_indexes(existing_nodes: Vec<ExistingNode>) -> CandidateIndexes {
    let mut by_normalized_name: HashMap<String, Vec<usize>> = HashMap::new();
    let mut shingles_by_candidate = Vec::with_capacity(existing_nodes.len());
    let mut lsh_buckets: HashMap<(usize, [u64; MINHASH_BAND_SIZE]), Vec<usize>> = HashMap::new();

    for (i, candidate) in existing_nodes.iter().enumerate() {
        by_normalized_name
            .entry(normalize_string_exact(&candidate.name))
            .or_default()
            .push(i);
        let candidate_shingles = shingles(&normalize_name_for_fuzzy(&candidate.name));
        let signature = minhash_signature(&candidate_shingles);
        for (band_index, band) in lsh_bands(&signature).into_iter().enumerate() {
            lsh_buckets.entry((band_index, band)).or_default().push(i);
        }
        shingles_by_candidate.push(candidate_shingles);
    }

    CandidateIndexes {
        existing_nodes,
        by_normalized_name,
        shingles_by_candidate,
        lsh_buckets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_fuzzy_strips_punctuation_and_unicode() {
        assert_eq!(
            normalize_name_for_fuzzy("Priya-Raman  Engineer"),
            "priya raman engineer"
        );
        assert_eq!(normalize_name_for_fuzzy("Nisha's dad"), "nisha's dad");
        assert_eq!(normalize_name_for_fuzzy("佐藤さん cafe"), "cafe");
    }

    #[test]
    fn entropy_gate_matches_upstream_thresholds() {
        // Short single-token names are gated out before entropy is checked.
        assert!(!has_high_entropy("nyc"));
        // Repetitive names have low entropy.
        assert!(!has_high_entropy("aaaaaaaa"));
        // Ordinary multi-token names pass.
        assert!(has_high_entropy("priya raman engineer"));
    }

    #[test]
    fn shingles_edge_cases_match_python() {
        assert!(shingles("").is_empty());
        assert_eq!(shingles("a"), vec!["a".to_owned()]);
        // Two chars -> range(0) in Python -> empty set.
        assert!(shingles("ab").is_empty());
        let three = shingles("abcd");
        assert_eq!(three, vec!["abc".to_owned(), "bcd".to_owned()]);
    }

    #[test]
    fn jaccard_and_signature_are_consistent() {
        let a = shingles("priyaramanengineer");
        let b = shingles("priyaramanengineer");
        assert!((jaccard_similarity(&a, &b) - 1.0).abs() < f64::EPSILON);
        assert_eq!(minhash_signature(&a), minhash_signature(&b));
        assert_eq!(lsh_bands(&minhash_signature(&a)).len(), 8);
    }

    #[test]
    fn shingle_hash_is_pinned_to_python_blake2b() {
        // Python: int.from_bytes(hashlib.blake2b(b'0:abc', digest_size=8).digest(), 'big')
        assert_eq!(hash_shingle("abc", 0), 7705568351334315143);
    }
}
