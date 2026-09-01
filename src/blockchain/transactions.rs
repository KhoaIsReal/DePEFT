use crate::blockchain::types::{
    AccountId, MergeStrategy, PeftType, RewardDistribution, ValidatorEvaluation,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Transaction {
    /// Client deposits bounty and registers a new PEFT task.
    CreateTask {
        client: AccountId,
        nonce: u64,
        base_model_id: Vec<u8>,
        base_model_hash: [u8; 32],
        dataset_cid: Vec<u8>,
        peft_method: PeftType,
        max_rank: u16,
        target_modules: Vec<Vec<u8>>,
        bounty_pool: u128,
        epoch_blocks: u32,
        #[serde(default)]
        reward_distribution: RewardDistribution,
        #[serde(default)]
        merge_strategy: MergeStrategy,
    },
    /// Miner submits commit hash before deadline: commit_hash = SHA256(adapter_bytes || salt).
    CommitAdapter {
        task_id: u64,
        round: usize,
        miner: AccountId,
        nonce: u64,
        commit_hash: [u8; 32],
    },
    /// Miner reveals adapter file on IPFS and provides salt.
    RevealAdapter {
        task_id: u64,
        round: usize,
        miner: AccountId,
        nonce: u64,
        adapter_cid: String,
        salt: Vec<u8>,
        adapter_hash: [u8; 32],
    },
    /// Validator submits ranking and benchmark evaluation.
    SubmitEvaluation {
        task_id: u64,
        round: usize,
        nonce: u64,
        evaluation: ValidatorEvaluation,
    },
}

impl Transaction {
    /// Get the sender AccountId of the transaction.
    pub fn sender(&self) -> &AccountId {
        match self {
            Transaction::CreateTask { client, .. } => client,
            Transaction::CommitAdapter { miner, .. } => miner,
            Transaction::RevealAdapter { miner, .. } => miner,
            Transaction::SubmitEvaluation { evaluation, .. } => &evaluation.validator_address,
        }
    }

    /// Get the transaction nonce.
    pub fn nonce(&self) -> u64 {
        match self {
            Transaction::CreateTask { nonce, .. } => *nonce,
            Transaction::CommitAdapter { nonce, .. } => *nonce,
            Transaction::RevealAdapter { nonce, .. } => *nonce,
            Transaction::SubmitEvaluation { nonce, .. } => *nonce,
        }
    }
}
