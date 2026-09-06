use serde::{Deserialize, Serialize};
use std::fmt;

/// Account identifier on the DePEFT App-Chain.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AccountId(pub String);

impl AccountId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AccountId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for AccountId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// PEFT (Parameter-Efficient Fine-Tuning) method specification.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PeftType {
    LoRA,
    QLoRA_NF4,
    QLoRA_INT4,
}

impl fmt::Display for PeftType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LoRA => write!(f, "LoRA (FP32)"),
            Self::QLoRA_NF4 => write!(f, "QLoRA (NF4 4-bit)"),
            Self::QLoRA_INT4 => write!(f, "QLoRA (INT4 4-bit)"),
        }
    }
}

/// Reward distribution strategy chosen by the task creator (Client).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum RewardDistribution {
    /// 100% of the round bounty awarded to the Top-1 winner.
    #[default]
    WinnerTakesAll,
    /// Award Top-K miners with exponential decay (e.g. top_k = 10, decay_rate = 0.5).
    TopKDecay { top_k: usize, decay_rate: f64 },
    /// Award Top-K miners proportionally based on their Borda consensus scores.
    TopKBordaWeighted { top_k: usize },
}

/// Weight merging strategy chosen by the task creator (Client).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum MergeStrategy {
    /// Merge only the Top-1 winner's adapter into the base model: W_{N+1} = W_N + Delta W*.
    #[default]
    SingleWinner,
    /// Ensemble weighted merge from Top-K adapters: W_{N+1} = W_N + sum(alpha_i * Delta W_i).
    EnsembleWeighted { top_k: usize },
}

/// TaskSpec matching the on-chain data specification in section 3 of DePEFT architecture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskSpec {
    pub task_id: u64,
    pub client_address: AccountId,

    // Base Model & Dataset specs
    pub base_model_id: Vec<u8>,     // e.g. "Qwen/Qwen2.5-7B" or "Llama-3-8B"
    pub base_model_hash: [u8; 32],  // Checksum against version mismatch
    pub dataset_cid: Vec<u8>,       // IPFS CID of training dataset

    // PEFT standard constraints
    pub peft_method: PeftType,        // Enum: LoRA, QLoRA_NF4, QLoRA_INT4
    pub max_rank: u16,                // Hyperparameter constraint (r <= 64)
    pub target_modules: Vec<Vec<u8>>, // e.g. ["q_proj", "v_proj"]

    // Economic & Tournament policy
    pub bounty_pool: u128,    // Total token reward for this epoch
    pub epoch_end_block: u32, // Block closing adapter submission
    #[serde(default)]
    pub reward_distribution: RewardDistribution, // Configurable bounty sharing strategy
    #[serde(default)]
    pub merge_strategy: MergeStrategy,           // Configurable adapter merge strategy
}

impl TaskSpec {
    pub fn base_model_id_str(&self) -> &str {
        std::str::from_utf8(&self.base_model_id).unwrap_or("unknown_model")
    }

    pub fn dataset_cid_str(&self) -> &str {
        std::str::from_utf8(&self.dataset_cid).unwrap_or("unknown_cid")
    }

    pub fn target_modules_str(&self) -> Vec<String> {
        self.target_modules
            .iter()
            .map(|m| String::from_utf8_lossy(m).to_string())
            .collect()
    }
}

/// Lifecycle phases of a ReLoRA Tournament round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoundPhase {
    /// Initialized round, Base model $W_N$ published, training started.
    CommitPhase,
    /// Deadline reached, Miners reveal .safetensors files on IPFS and commit proofs.
    RevealPhase,
    /// Validators download adapters, evaluate on Private Test Set, submit rankings.
    EvaluationPhase,
    /// Top-1 winner selected via Relative Consensus, merged into $W_{N+1}$, reward paid.
    MergePhase,
    /// Round finalized.
    Completed,
}

impl fmt::Display for RoundPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CommitPhase => write!(f, "Commit Phase (Miners Training & Hash Commits)"),
            Self::RevealPhase => write!(f, "Reveal Phase (Miners Upload .safetensors)"),
            Self::EvaluationPhase => write!(f, "Evaluation Phase (Validators Private Test)"),
            Self::MergePhase => write!(f, "Merge Phase (Top 1 Weight Evolution $W_{{N+1}}$)"),
            Self::Completed => write!(f, "Completed"),
        }
    }
}

/// Commit submission by a miner during Commit Phase.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitRecord {
    pub miner_address: AccountId,
    pub commit_hash: [u8; 32],
    pub submitted_at_block: u32,
}

/// Reveal submission by a miner during Reveal Phase.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevealRecord {
    pub miner_address: AccountId,
    pub adapter_cid: String,
    pub salt: Vec<u8>,
    pub adapter_hash: [u8; 32],
}

use crate::tee::AttestationQuote;

/// Individual evaluation result by a single validator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidatorEvaluation {
    pub validator_address: AccountId,
    /// Ordered list of miners from best (lowest loss) to worst (highest loss).
    pub ranking: Vec<AccountId>,
    /// Raw loss observed by this validator (may have slight floating-point drift).
    pub loss_scores: Vec<(AccountId, f64)>,
    /// Accuracy observed on private test set.
    pub accuracy_scores: Vec<(AccountId, f64)>,
    /// Validator hardware profile description (e.g., "NVIDIA RTX 4090 / CUDA", "AMD RX 7900 / ROCm", "CPU AVX512")
    pub hardware_info: String,
    /// Optional cryptographic Hardware TEE Attestation Quote
    pub attestation_quote: Option<AttestationQuote>,
}

/// Summary result of a ReLoRA Tournament Round.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoundSummary {
    pub round_number: usize,
    pub base_model_cid: String,
    pub evolved_model_cid: String,
    pub winning_miner: AccountId,
    pub winning_adapter_cid: String,
    pub consensus_ranking: Vec<AccountId>,
    pub borda_scores: Vec<(AccountId, usize)>,
    pub pre_merge_loss: f64,
    pub post_merge_loss: f64,
    pub bounty_awarded: u128,
    #[serde(default)]
    pub reward_distributions: Vec<(AccountId, u128)>,
    #[serde(default)]
    pub validator_rewards: Vec<(AccountId, u128)>,
    #[serde(default)]
    pub node_rewards: Vec<(AccountId, u128)>,
}
