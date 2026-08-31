use crate::blockchain::types::{AccountId, ValidatorEvaluation};
use crate::ml::model::DePEFTModel;
use crate::storage::ipfs::IpfsStorage;
use crate::storage::safetensors::deserialize_safetensors;
use crate::storage::vector_db::{AdapterVectorRecord, EmbeddedVectorDb};
use crate::validator::tee::TeeSandbox;
use anyhow::Result;

/// Off-chain evaluation engine for validator nodes.
pub struct OffChainEvaluator;

impl OffChainEvaluator {
    /// Download revealed adapters from IPFS, evaluate in TEE Sandbox, index in Vector DB,
    /// and generate ranked evaluation vector with floating-point drift simulation.
    pub fn evaluate_candidates(
        validator_address: &AccountId,
        hardware_info: &str,
        hardware_drift: f32,
        base_model: &DePEFTModel,
        reveals: &[(AccountId, String)], // (miner_address, adapter_cid)
        tee_sandbox: &TeeSandbox,
        vector_db: Option<&EmbeddedVectorDb>,
        task_id: u64,
        round: usize,
    ) -> Result<ValidatorEvaluation> {
        let mut scores: Vec<(AccountId, f64, f64)> = Vec::new();

        for (miner, cid) in reveals {
            // Fetch .safetensors from IPFS CAS
            let ipfs = IpfsStorage::new(); // or passed in
            let bytes = match ipfs.get(cid) {
                Some(b) => b,
                None => {
                    // Fallback to error loss if file not accessible
                    scores.push((miner.clone(), 999.0, 0.0));
                    continue;
                }
            };

            let adapter_pkg = deserialize_safetensors(&bytes)?;

            // Index adapter signature into Vector DB for similarity / plagiarism tracking
            if let Some(vdb) = vector_db {
                let signature = adapter_pkg.generate_signature_vector(64);
                let record = AdapterVectorRecord {
                    adapter_cid: cid.clone(),
                    miner_address: miner.clone(),
                    task_id,
                    round,
                    signature,
                };
                vdb.insert(record);
            }

            // Run evaluation in secure TEE Sandbox
            let (loss, accuracy) = tee_sandbox.evaluate_adapter(base_model, &adapter_pkg, hardware_drift);
            scores.push((miner.clone(), loss, accuracy));
        }

        // Sort candidates by Loss (ascending: lowest loss is 1st place). NaN / Inf go to the bottom.
        scores.sort_by(|a, b| {
            let la = if a.1.is_finite() { a.1 } else { 9999.0 };
            let lb = if b.1.is_finite() { b.1 } else { 9999.0 };
            la.partial_cmp(&lb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let ranking: Vec<AccountId> = scores.iter().map(|(m, _, _)| m.clone()).collect();
        let loss_scores: Vec<(AccountId, f64)> = scores.iter().map(|(m, l, _)| (m.clone(), *l)).collect();
        let accuracy_scores: Vec<(AccountId, f64)> = scores.iter().map(|(m, _, a)| (m.clone(), *a)).collect();

        Ok(ValidatorEvaluation {
            validator_address: validator_address.clone(),
            ranking,
            loss_scores,
            accuracy_scores,
            hardware_info: hardware_info.to_string(),
            attestation_quote: None,
        })
    }
}
