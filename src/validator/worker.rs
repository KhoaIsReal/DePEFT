use crate::blockchain::transactions::Transaction;
use crate::blockchain::types::{AccountId, ValidatorEvaluation};
use crate::ml::model::DePEFTModel;
use crate::storage::ipfs::IpfsStorage;
use crate::storage::safetensors::deserialize_safetensors;
use crate::storage::vector_db::{AdapterVectorRecord, EmbeddedVectorDb};
use crate::tee::{HardwareTeeEnclave, TeeType};
use crate::validator::tee::TeeSandbox;
use anyhow::Result;

/// Validator Off-Chain Worker Node.
#[derive(Clone)]
pub struct ValidatorNode {
    pub account_id: AccountId,
    pub hardware_info: String,
    pub hardware_drift: f32,
    pub tee_sandbox: TeeSandbox,
    pub hardware_enclave: Option<HardwareTeeEnclave>,
    pub vector_db: Option<EmbeddedVectorDb>,
}

impl ValidatorNode {
    pub fn new(
        account_id: impl Into<String>,
        hardware_info: impl Into<String>,
        hardware_drift: f32,
        tee_sandbox: TeeSandbox,
        vector_db: Option<EmbeddedVectorDb>,
    ) -> Self {
        let enclave = HardwareTeeEnclave::official(TeeType::IntelSgxDcap);
        Self {
            account_id: AccountId::new(account_id),
            hardware_info: hardware_info.into(),
            hardware_drift,
            tee_sandbox,
            hardware_enclave: Some(enclave),
            vector_db,
        }
    }

    /// Perform off-chain evaluation of all revealed candidate adapters for the current round.
    pub fn evaluate_round(
        &self,
        ipfs: &IpfsStorage,
        task_id: u64,
        round: usize,
        nonce: u64,
        base_model: &DePEFTModel,
        reveals: &[(AccountId, String)], // list of (miner_id, adapter_cid)
    ) -> Result<Transaction> {
        let mut scores: Vec<(AccountId, f64, f64)> = Vec::new();

        for (miner, cid) in reveals {
            let bytes = ipfs
                .get(cid)
                .ok_or_else(|| anyhow::anyhow!("Adapter CID not found on IPFS: {}", cid))?;

            let adapter_pkg = deserialize_safetensors(&bytes)?;

            // Index in vector DB if enabled
            if let Some(vdb) = &self.vector_db {
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

            // Run evaluation in secure TEE Sandbox Enclave
            let (loss, accuracy) =
                self.tee_sandbox
                    .evaluate_adapter(base_model, &adapter_pkg, self.hardware_drift);
            scores.push((miner.clone(), loss, accuracy));
        }

        // Sort ranking by loss ascending (lowest loss = rank 1). NaN / Inf go to the bottom.
        scores.sort_by(|a, b| {
            let la = if a.1.is_finite() { a.1 } else { 9999.0 };
            let lb = if b.1.is_finite() { b.1 } else { 9999.0 };
            la.partial_cmp(&lb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let ranking: Vec<AccountId> = scores.iter().map(|(m, _, _)| m.clone()).collect();
        let loss_scores: Vec<(AccountId, f64)> = scores.iter().map(|(m, l, _)| (m.clone(), *l)).collect();
        let accuracy_scores: Vec<(AccountId, f64)> = scores.iter().map(|(m, _, a)| (m.clone(), *a)).collect();

        // Generate Hardware TEE Attestation Quote cryptographically bound to ranking
        let attestation_quote = if let Some(enclave) = &self.hardware_enclave {
            Some(enclave.generate_quote(task_id, round, &ranking)?)
        } else {
            None
        };

        let evaluation = ValidatorEvaluation {
            validator_address: self.account_id.clone(),
            ranking,
            loss_scores,
            accuracy_scores,
            hardware_info: self.hardware_info.clone(),
            attestation_quote,
        };

        Ok(Transaction::SubmitEvaluation {
            task_id,
            round,
            nonce,
            evaluation,
        })
    }
}
