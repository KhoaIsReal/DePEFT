use crate::blockchain::types::{AccountId, PeftType, ValidatorEvaluation};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Transaction {
    /// Client deposits bounty and registers a new PEFT task.
    CreateTask {
        client: AccountId,
        base_model_id: Vec<u8>,
        base_model_hash: [u8; 32],
        dataset_cid: Vec<u8>,
        peft_method: PeftType,
        max_rank: u16,
        target_modules: Vec<Vec<u8>>,
        bounty_pool: u128,
        epoch_blocks: u32,
    },
    /// Miner submits commit hash before deadline: commit_hash = SHA256(adapter_bytes || salt).
    CommitAdapter {
        task_id: u64,
        round: usize,
        miner: AccountId,
        commit_hash: [u8; 32],
    },
    /// Miner reveals adapter file on IPFS and provides salt.
    RevealAdapter {
        task_id: u64,
        round: usize,
        miner: AccountId,
        adapter_cid: String,
        salt: Vec<u8>,
        adapter_hash: [u8; 32],
    },
    /// Validator submits ranking and benchmark evaluation.
    SubmitEvaluation {
        task_id: u64,
        round: usize,
        evaluation: ValidatorEvaluation,
    },
}
