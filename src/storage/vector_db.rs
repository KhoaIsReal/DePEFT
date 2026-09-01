use crate::blockchain::types::AccountId;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

/// Metadata stored alongside an adapter embedding vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterVectorRecord {
    pub adapter_cid: String,
    pub miner_address: AccountId,
    pub task_id: u64,
    pub round: usize,
    pub signature: Vec<f32>,
}

/// Similarity search match.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorSearchResult {
    pub record: AdapterVectorRecord,
    pub similarity: f32,
}

/// Embedded lightweight vector database for indexing adapter signatures at Validator layer.
#[derive(Debug, Clone)]
pub struct EmbeddedVectorDb {
    dimension: usize,
    records: Arc<RwLock<Vec<AdapterVectorRecord>>>,
}

impl EmbeddedVectorDb {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            records: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Insert an adapter signature into the vector database safely.
    pub fn insert(&self, record: AdapterVectorRecord) {
        if record.signature.len() != self.dimension {
            return;
        }
        // Sanitize vector: reject signatures containing NaN or Infinity
        if record.signature.iter().any(|x| !x.is_finite()) {
            return;
        }
        let mut recs = self.records.write().unwrap();
        recs.push(record);
    }

    /// Compute cosine similarity between two unit-normalized vectors: $u \cdot v$.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let mut dot: f32 = 0.0;
        let mut norm_a: f32 = 0.0;
        let mut norm_b: f32 = 0.0;
        for (x, y) in a.iter().zip(b) {
            if !x.is_finite() || !y.is_finite() {
                return 0.0;
            }
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }
        norm_a = norm_a.sqrt();
        norm_b = norm_b.sqrt();
        if norm_a < 1e-6 || norm_b < 1e-6 {
            0.0
        } else {
            (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
        }
    }

    /// Search the Top-K most similar adapter signatures in the database.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<VectorSearchResult> {
        let recs = self.records.read().unwrap();
        let mut results: Vec<VectorSearchResult> = recs
            .iter()
            .map(|r| {
                let sim = Self::cosine_similarity(query, &r.signature);
                VectorSearchResult {
                    record: r.clone(),
                    similarity: sim,
                }
            })
            .collect();

        // Sort descending by similarity
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Detect potential weight plagiarism or collusion (> 0.999 cosine similarity between different miners).
    pub fn check_plagiarism(&self, query: &[f32], current_miner: &AccountId) -> Option<VectorSearchResult> {
        let results = self.search(query, 5);
        for res in results {
            if &res.record.miner_address != current_miner && res.similarity > 0.999 {
                return Some(res);
            }
        }
        None
    }

    /// Total number of indexed vectors.
    pub fn count(&self) -> usize {
        let recs = self.records.read().unwrap();
        recs.len()
    }
}
