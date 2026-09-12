use crate::blockchain::transactions::Transaction;
use crate::blockchain::types::{AccountId, TaskSpec};
use crate::miner::trainer::{MinerHyperparams, MinerTrainer, MinerTrainingArtifact};
use crate::ml::dataset::Dataset;
use crate::ml::model::DePEFTModel;
use crate::storage::ipfs::IpfsStorage;
use anyhow::Result;
use rand::Rng;

/// A decentralized Miner Node participating in DePEFT ReLoRA Tournaments.
#[derive(Debug, Clone)]
pub struct MinerNode {
    pub account_id: AccountId,
    pub hyperparams: MinerHyperparams,
    pub cached_artifact: Option<MinerTrainingArtifact>,
}

impl MinerNode {
    pub fn new(account_id: impl Into<String>, hyperparams: MinerHyperparams) -> Self {
        Self {
            account_id: AccountId::new(account_id),
            hyperparams,
            cached_artifact: None,
        }
    }

    /// Phase 2 (Commit Phase): Run QLoRA training and create commit transaction.
    pub fn run_training_and_commit(
        &mut self,
        task_id: u64,
        round: usize,
        nonce: u64,
        task_spec: &TaskSpec,
        base_model: &DePEFTModel,
        dataset: &Dataset,
        rng: &mut impl Rng,
    ) -> Result<Transaction> {
        let artifact = MinerTrainer::train(
            base_model,
            dataset,
            task_spec,
            round,
            &self.hyperparams,
            rng,
        )?;

        let commit_hash = artifact.commit_hash;
        self.cached_artifact = Some(artifact);

        Ok(Transaction::CommitAdapter {
            task_id,
            round,
            miner: self.account_id.clone(),
            nonce,
            commit_hash,
        })
    }

    /// Phase 3 (Reveal Phase): Upload .safetensors to IPFS and create reveal transaction.
    pub fn reveal_adapter(
        &mut self,
        task_id: u64,
        round: usize,
        nonce: u64,
        ipfs: &IpfsStorage,
    ) -> Result<(Transaction, String)> {
        let artifact = self
            .cached_artifact
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No cached training artifact found to reveal"))?;

        // Upload .safetensors bytes to IPFS CAS
        let adapter_cid = ipfs.put(&artifact.safetensors_bytes);

        let tx = Transaction::RevealAdapter {
            task_id,
            round,
            miner: self.account_id.clone(),
            nonce,
            adapter_cid: adapter_cid.clone(),
            salt: artifact.salt.clone(),
            adapter_hash: artifact.adapter_hash,
        };

        Ok((tx, adapter_cid))
    }
}
