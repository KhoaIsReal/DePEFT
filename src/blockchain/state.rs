use crate::blockchain::relative_consensus::{ConsensusResult, RelativeConsensusEngine};
use crate::blockchain::transactions::Transaction;
use crate::blockchain::types::{
    AccountId, CommitRecord, RevealRecord, RoundPhase, RoundSummary, TaskSpec, ValidatorEvaluation,
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet, VecDeque};

/// Runtime state of a specific tournament round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundContext {
    pub task_id: u64,
    pub round_number: usize,
    pub phase: RoundPhase,
    pub base_model_cid: String,
    #[serde(default)]
    pub epoch_end_block: u32,
    pub commits: HashMap<AccountId, CommitRecord>,
    pub reveals: HashMap<AccountId, RevealRecord>,
    pub evaluations: HashMap<AccountId, ValidatorEvaluation>,
    pub consensus: Option<ConsensusResult>,
}

impl RoundContext {
    pub fn new(task_id: u64, round_number: usize, base_model_cid: String, epoch_end_block: u32) -> Self {
        Self {
            task_id,
            round_number,
            phase: RoundPhase::CommitPhase,
            base_model_cid,
            epoch_end_block,
            commits: HashMap::new(),
            reveals: HashMap::new(),
            evaluations: HashMap::new(),
            consensus: None,
        }
    }
}

use crate::tee::OnChainTeeVerifier;

/// The App-Chain State Machine managing state, escrows, task specs, and tournament consensus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppChainState {
    pub block_height: u32,
    pub next_task_id: u64,
    pub balances: HashMap<AccountId, u128>,
    pub nonces: HashMap<AccountId, u64>,
    pub escrows: HashMap<u64, u128>,
    pub tasks: HashMap<u64, TaskSpec>,
    pub round_contexts: HashMap<(u64, usize), RoundContext>,
    pub round_history: Vec<RoundSummary>,
    #[serde(default)]
    pub total_burned: u128,
    #[serde(default)]
    pub slashed_validators: HashSet<AccountId>,
    #[serde(skip)]
    pub tee_verifier: OnChainTeeVerifier,
    /// On-Chain Emergency Circuit Breaker (Safe Mode) to halt runaway drainage
    #[serde(default)]
    pub circuit_breaker_active: bool,
    #[serde(default)]
    pub circuit_breaker_triggered_at_block: Option<u32>,
    #[serde(default)]
    pub rolling_payout_history: VecDeque<(u32, u128)>,
    #[serde(default = "default_payout_velocity_limit")]
    pub max_payout_velocity_per_window: u128,
    #[serde(default = "default_circuit_breaker_window")]
    pub circuit_breaker_window_blocks: u32,
}

fn default_payout_velocity_limit() -> u128 {
    1_000_000
}

fn default_circuit_breaker_window() -> u32 {
    100
}

impl Default for AppChainState {
    fn default() -> Self {
        Self::new()
    }
}

impl AppChainState {
    const MAX_ADAPTER_CID_LEN: usize = 128;
    const MAX_REVEAL_SALT_LEN: usize = 1024;
    const MAX_HARDWARE_INFO_LEN: usize = 256;

    pub fn new() -> Self {
        Self {
            block_height: 1,
            next_task_id: 1,
            balances: HashMap::new(),
            nonces: HashMap::new(),
            escrows: HashMap::new(),
            tasks: HashMap::new(),
            round_contexts: HashMap::new(),
            round_history: Vec::new(),
            total_burned: 0,
            slashed_validators: HashSet::new(),
            tee_verifier: OnChainTeeVerifier::default(),
            circuit_breaker_active: false,
            circuit_breaker_triggered_at_block: None,
            rolling_payout_history: VecDeque::new(),
            max_payout_velocity_per_window: default_payout_velocity_limit(),
            circuit_breaker_window_blocks: default_circuit_breaker_window(),
        }
    }

    /// Resolve an account identifier to its canonical representation (e.g. storage gateway aliases).
    pub fn canonical_account(account: &AccountId) -> AccountId {
        if account.0 == "ipfs-storage-gateway" {
            AccountId::storage_gateway()
        } else {
            account.clone()
        }
    }

    /// Slash and permanently ban a Byzantine or fraudulent validator/account.
    pub fn slash_validator(&mut self, validator: &AccountId, reason: &str) {
        let canonical = Self::canonical_account(validator);
        self.slashed_validators.insert(validator.clone());
        self.slashed_validators.insert(canonical.clone());
        let bal = self.balances.remove(&canonical).unwrap_or(0);
        self.total_burned += bal;
        eprintln!(
            "[SLASHED] Account {} has been slashed and permanently banned. Reason: {}",
            validator, reason
        );
    }

    /// Check if an account has been slashed.
    pub fn is_slashed(&self, account: &AccountId) -> bool {
        let canonical = Self::canonical_account(account);
        self.slashed_validators.contains(account) || self.slashed_validators.contains(&canonical)
    }

    /// Check and record a proposed bounty payout against the rolling window velocity cap.
    /// If velocity is exceeded, automatically triggers the Emergency Circuit Breaker (Safe Mode).
    pub fn check_and_record_payout(&mut self, amount: u128) -> Result<()> {
        if self.circuit_breaker_active {
            anyhow::bail!(
                "Emergency Safe Mode is ACTIVE! Circuit breaker triggered at block #{:?}. Payouts halted.",
                self.circuit_breaker_triggered_at_block
            );
        }

        if amount == 0 {
            return Ok(());
        }

        // Prune entries outside the rolling block window
        let window_start = self.block_height.saturating_sub(self.circuit_breaker_window_blocks);
        while let Some(&(block, _)) = self.rolling_payout_history.front() {
            if block < window_start {
                self.rolling_payout_history.pop_front();
            } else {
                break;
            }
        }

        let current_velocity: u128 = self.rolling_payout_history.iter().map(|&(_, amt)| amt).sum();
        if current_velocity.saturating_add(amount) > self.max_payout_velocity_per_window {
            self.circuit_breaker_active = true;
            self.circuit_breaker_triggered_at_block = Some(self.block_height);
            anyhow::bail!(
                "Emergency Circuit Breaker Triggered: Payout velocity ({} + {}) exceeds limit {} in rolling window of {} blocks. Safe Mode Activated!",
                current_velocity, amount, self.max_payout_velocity_per_window, self.circuit_breaker_window_blocks
            );
        }

        self.rolling_payout_history.push_back((self.block_height, amount));
        Ok(())
    }

    /// Reset emergency circuit breaker (admin/governance intervention).
    pub fn reset_circuit_breaker(&mut self) {
        self.circuit_breaker_active = false;
        self.circuit_breaker_triggered_at_block = None;
        self.rolling_payout_history.clear();
        eprintln!("[SECURITY] Emergency Circuit Breaker manually reset. Safe Mode deactivated.");
    }

    /// Update rolling payout velocity limit.
    pub fn set_circuit_breaker_velocity_limit(&mut self, limit: u128) {
        self.max_payout_velocity_per_window = limit;
    }

    /// Credit native tokens to an account balance.
    pub fn credit(&mut self, account: &AccountId, amount: u128) {
        let canonical = Self::canonical_account(account);
        *self.balances.entry(canonical).or_insert(0) += amount;
    }

    /// Debit tokens from an account balance with overflow and underflow protection.
    pub fn debit(&mut self, account: &AccountId, amount: u128) -> Result<()> {
        let canonical = Self::canonical_account(account);
        let cur = self.balances.entry(canonical).or_insert(0);
        ensure!(
            *cur >= amount,
            "Insufficient balance: have {}, need {}",
            *cur,
            amount
        );
        *cur -= amount;
        Ok(())
    }

    /// Deposit native tokens to an account balance.
    pub fn mint(&mut self, account: AccountId, amount: u128) {
        self.credit(&account, amount);
    }

    /// Get current balance of an account.
    pub fn balance_of(&self, account: &AccountId) -> u128 {
        let canonical = Self::canonical_account(account);
        self.balances.get(&canonical).copied().unwrap_or(0)
    }

    /// Get next expected nonce of an account.
    pub fn nonce_of(&self, account: &AccountId) -> u64 {
        self.nonces.get(account).copied().unwrap_or(0)
    }

    /// Advance block height by 1 and auto-transition round phases based on block progression.
    pub fn advance_block(&mut self) -> u32 {
        self.block_height += 1;
        self.tick_round_phases();
        self.block_height
    }

    /// Check and transition round phases according to epoch blocks and submission state.
    pub fn tick_round_phases(&mut self) {
        let current_block = self.block_height;
        for ((task_id, _round_num), ctx) in self.round_contexts.iter_mut() {
            let deadline = if ctx.epoch_end_block > 0 {
                ctx.epoch_end_block
            } else if let Some(task) = self.tasks.get(task_id) {
                task.epoch_end_block
            } else {
                0
            };

            // Determine phase progression intervals
            // 1. In CommitPhase: move to RevealPhase only when epoch deadline is reached
            if ctx.phase == RoundPhase::CommitPhase && !ctx.commits.is_empty() {
                if current_block >= deadline {
                    ctx.phase = RoundPhase::RevealPhase;
                }
            }
            // 2. In RevealPhase: if all committed miners revealed, or time elapsed, move to EvaluationPhase
            else if ctx.phase == RoundPhase::RevealPhase && !ctx.reveals.is_empty() {
                let all_revealed = ctx.commits.keys().all(|m| ctx.reveals.contains_key(m));
                if all_revealed || current_block >= deadline + 2 {
                    ctx.phase = RoundPhase::EvaluationPhase;
                }
            }
        }
    }

    /// Helper to compute commit hash: SHA256(adapter_hash || salt).
    pub fn compute_commit_hash(adapter_hash: &[u8; 32], salt: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(adapter_hash);
        hasher.update(salt);
        hasher.finalize().into()
    }

    /// Reject malformed CIDs before they are persisted in consensus state.  This
    /// prevents invalid identifiers from later reaching storage backends that may
    /// interpret them as paths or URLs.
    fn is_safe_cid(cid: &str) -> bool {
        !cid.is_empty()
            && cid.len() <= Self::MAX_ADAPTER_CID_LEN
            && cid
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
    }

    /// An evaluation is only meaningful when it ranks every and only every
    /// revealed candidate once.  In particular, accepting partial rankings lets
    /// a validator assign arbitrary worst ranks in Borda aggregation, while
    /// accepting unknown accounts makes the rank scale attacker-controlled.
    fn validate_evaluation(ctx: &RoundContext, evaluation: &ValidatorEvaluation) -> Result<()> {
        ensure!(
            evaluation.ranking.len() == ctx.reveals.len(),
            "Evaluation ranking must contain every revealed miner exactly once"
        );
        ensure!(
            evaluation.hardware_info.len() <= Self::MAX_HARDWARE_INFO_LEN,
            "Validator hardware info exceeds {} bytes",
            Self::MAX_HARDWARE_INFO_LEN
        );

        let mut ranked = std::collections::HashSet::with_capacity(evaluation.ranking.len());
        for miner in &evaluation.ranking {
            ensure!(
                ctx.reveals.contains_key(miner),
                "Evaluation ranking contains miner without a revealed adapter: {}",
                miner
            );
            ensure!(
                ranked.insert(miner),
                "Evaluation ranking contains duplicate miner: {}",
                miner
            );
        }

        for (miner, loss) in &evaluation.loss_scores {
            ensure!(
                ranked.contains(miner),
                "Loss score contains unranked miner: {}",
                miner
            );
            ensure!(
                loss.is_finite() && *loss >= 0.0,
                "Loss score must be finite and non-negative"
            );
        }
        for (miner, accuracy) in &evaluation.accuracy_scores {
            ensure!(
                ranked.contains(miner),
                "Accuracy score contains unranked miner: {}",
                miner
            );
            ensure!(
                accuracy.is_finite() && (0.0..=1.0).contains(accuracy),
                "Accuracy score must be finite and within [0, 1]"
            );
        }

        Ok(())
    }

    /// Initialize a new round context for a task.
    pub fn start_round(
        &mut self,
        task_id: u64,
        round_number: usize,
        base_model_cid: String,
    ) -> Result<()> {
        let task = self
            .tasks
            .get_mut(&task_id)
            .ok_or_else(|| anyhow::anyhow!("Task ID does not exist"))?;
        ensure!(
            !self.round_contexts.contains_key(&(task_id, round_number)),
            "Round {} already initialized for task {}",
            round_number,
            task_id
        );
        let epoch_end_block = self.block_height + task.epoch_blocks;
        task.epoch_end_block = epoch_end_block;
        let context = RoundContext::new(task_id, round_number, base_model_cid, epoch_end_block);
        self.round_contexts.insert((task_id, round_number), context);
        Ok(())
    }

    /// Transition round phase.
    pub fn set_round_phase(
        &mut self,
        task_id: u64,
        round_number: usize,
        phase: RoundPhase,
    ) -> Result<()> {
        let ctx = self
            .round_contexts
            .get_mut(&(task_id, round_number))
            .ok_or_else(|| anyhow::anyhow!("Round context not found"))?;
        ctx.phase = phase;
        Ok(())
    }

    /// Process an on-chain transaction deterministically.
    pub fn apply_transaction(&mut self, tx: Transaction, sender: &AccountId) -> Result<()> {
        // Banned/Slashed account verification: Reject any transactions from slashed accounts
        ensure!(
            !self.is_slashed(sender),
            "Transaction rejected: Sender {} has been slashed and permanently banned from the network",
            sender
        );

        // Anti-Replay Attack verification: verify transaction nonce matches expected sender nonce
        let expected_nonce = self.nonce_of(sender);
        ensure!(
            tx.nonce() == expected_nonce,
            "Invalid transaction nonce for {}: expected {}, got {}",
            sender,
            expected_nonce,
            tx.nonce()
        );

        match tx {
            Transaction::CreateTask {
                client,
                nonce: _,
                base_model_id,
                base_model_hash,
                dataset_cid,
                peft_method,
                max_rank,
                target_modules,
                bounty_pool,
                epoch_blocks,
                reward_distribution,
                merge_strategy,
            } => {
                ensure!(
                    &client == sender,
                    "Unauthorized: Transaction sender {} does not match client address {}",
                    sender,
                    client
                );
                ensure!(
                    !self.circuit_breaker_active,
                    "CreateTask rejected: Emergency Safe Mode is ACTIVE! Circuit breaker triggered at block #{:?}",
                    self.circuit_breaker_triggered_at_block
                );
                ensure!(
                    !self.slashed_validators.contains(&client),
                    "CreateTask rejected: Client {} is banned",
                    client
                );
                ensure!(bounty_pool > 0, "Bounty pool must be greater than 0 tokens");
                ensure!(
                    epoch_blocks >= 5,
                    "Epoch duration must be at least 5 blocks"
                );
                ensure!(
                    max_rank > 0 && max_rank <= 256,
                    "Max rank must be between 1 and 256"
                );
                ensure!(
                    !base_model_id.is_empty() && base_model_id.len() <= 128,
                    "Base model ID must be 1-128 bytes"
                );
                ensure!(
                    !dataset_cid.is_empty() && dataset_cid.len() <= 128,
                    "Dataset CID must be 1-128 bytes"
                );
                ensure!(
                    !target_modules.is_empty() && target_modules.len() <= 32,
                    "Target modules must contain 1-32 items"
                );
                ensure!(
                    target_modules
                        .iter()
                        .all(|m| !m.is_empty() && m.len() <= 64),
                    "Each target module name must be 1-64 bytes"
                );

                // Deduct from client balance and lock into escrow
                self.debit(&client, bounty_pool)?;
                let task_id = self.next_task_id;
                self.next_task_id += 1;

                self.escrows.insert(task_id, bounty_pool);

                let task_spec = TaskSpec {
                    task_id,
                    client_address: client,
                    base_model_id: base_model_id.clone(),
                    base_model_hash,
                    dataset_cid,
                    peft_method,
                    max_rank,
                    target_modules,
                    bounty_pool,
                    epoch_blocks,
                    epoch_end_block: self.block_height + epoch_blocks,
                    reward_distribution,
                    merge_strategy,
                };

                self.tasks.insert(task_id, task_spec);
            }

            Transaction::CommitAdapter {
                task_id,
                round,
                miner,
                nonce: _,
                commit_hash,
            } => {
                ensure!(
                    &miner == sender,
                    "Unauthorized: Transaction sender {} does not match miner address {}",
                    sender,
                    miner
                );
                ensure!(
                    !self.slashed_validators.contains(&miner),
                    "Commit rejected: Miner {} is banned",
                    miner
                );
                ensure!(
                    commit_hash != [0u8; 32],
                    "Commit rejected: commit hash cannot be all zeros"
                );
                ensure!(
                    self.tasks.contains_key(&task_id),
                    "Commit rejected: Task #{} does not exist",
                    task_id
                );

                // Auto initialize round context if not yet started
                if !self.round_contexts.contains_key(&(task_id, round)) {
                    if let Some(task) = self.tasks.get_mut(&task_id) {
                        let epoch_end_block = self.block_height + task.epoch_blocks;
                        task.epoch_end_block = epoch_end_block;
                        let base_model_cid_str = task.base_model_id_str();
                        let context =
                            RoundContext::new(task_id, round, base_model_cid_str.to_string(), epoch_end_block);
                        self.round_contexts.insert((task_id, round), context);
                    }
                }

                let ctx = self
                    .round_contexts
                    .get_mut(&(task_id, round))
                    .ok_or_else(|| anyhow::anyhow!("Round context not found"))?;

                ensure!(
                    ctx.phase == RoundPhase::CommitPhase,
                    "Commit rejected: Round is in phase {:?}",
                    ctx.phase
                );

                ensure!(
                    !ctx.commits.contains_key(&miner),
                    "Commit already submitted by miner {} for round {}",
                    miner,
                    round
                );

                ctx.commits.insert(
                    miner.clone(),
                    CommitRecord {
                        miner_address: miner,
                        commit_hash,
                        submitted_at_block: self.block_height,
                    },
                );
            }

            Transaction::RevealAdapter {
                task_id,
                round,
                miner,
                nonce: _,
                adapter_cid,
                salt,
                adapter_hash,
            } => {
                ensure!(
                    &miner == sender,
                    "Unauthorized: Transaction sender {} does not match miner address {}",
                    sender,
                    miner
                );
                ensure!(
                    !self.slashed_validators.contains(&miner),
                    "Reveal rejected: Miner {} is banned",
                    miner
                );
                ensure!(
                    adapter_hash != [0u8; 32],
                    "Reveal rejected: adapter hash cannot be all zeros"
                );
                ensure!(
                    self.tasks.contains_key(&task_id),
                    "Reveal rejected: Task #{} does not exist",
                    task_id
                );
                let ctx = self
                    .round_contexts
                    .get_mut(&(task_id, round))
                    .ok_or_else(|| anyhow::anyhow!("Round context not found"))?;

                ensure!(
                    ctx.phase == RoundPhase::RevealPhase,
                    "Reveal rejected: Round is in phase {:?}",
                    ctx.phase
                );

                ensure!(
                    Self::is_safe_cid(&adapter_cid),
                    "Reveal rejected: adapter CID is malformed or exceeds {} bytes",
                    Self::MAX_ADAPTER_CID_LEN
                );
                ensure!(
                    !salt.is_empty() && salt.len() <= Self::MAX_REVEAL_SALT_LEN,
                    "Reveal rejected: salt must contain 1-{} bytes",
                    Self::MAX_REVEAL_SALT_LEN
                );

                ensure!(
                    !ctx.reveals.contains_key(&miner),
                    "Reveal already submitted by miner {} for round {}",
                    miner,
                    round
                );

                // Anti-Plagiarism check: reject if another miner already revealed the identical adapter hash or CID
                ensure!(
                    !ctx.reveals.values().any(|r| r.adapter_hash == adapter_hash || r.adapter_cid == adapter_cid),
                    "Reveal rejected: duplicate adapter hash or CID detected from another miner in this round (plagiarism rejected)"
                );

                let commit = ctx
                    .commits
                    .get(&miner)
                    .ok_or_else(|| anyhow::anyhow!("No commit found for miner"))?;

                // Anti-Collusion verification: Check commit hash matches revealed adapter hash and salt
                let expected_commit = Self::compute_commit_hash(&adapter_hash, &salt);
                ensure!(
                    expected_commit == commit.commit_hash,
                    "Reveal failed: Commit hash mismatch for miner {}",
                    miner
                );

                ctx.reveals.insert(
                    miner.clone(),
                    RevealRecord {
                        miner_address: miner,
                        adapter_cid,
                        salt,
                        adapter_hash,
                    },
                );
            }

            Transaction::SubmitEvaluation {
                task_id,
                round,
                nonce: _,
                evaluation,
            } => {
                ensure!(
                    &evaluation.validator_address == sender,
                    "Unauthorized: Transaction sender {} does not match validator address {}",
                    sender,
                    evaluation.validator_address
                );
                ensure!(
                    !self.slashed_validators.contains(&evaluation.validator_address),
                    "SubmitEvaluation rejected: Validator {} has been slashed and permanently banned",
                    evaluation.validator_address
                );
                ensure!(
                    self.tasks.contains_key(&task_id),
                    "SubmitEvaluation rejected: Task #{} does not exist",
                    task_id
                );
                let ctx = self
                    .round_contexts
                    .get_mut(&(task_id, round))
                    .ok_or_else(|| anyhow::anyhow!("Round context not found"))?;

                ensure!(
                    ctx.phase == RoundPhase::EvaluationPhase,
                    "Evaluation rejected: Round is in phase {:?}",
                    ctx.phase
                );

                ensure!(
                    !ctx.evaluations.contains_key(&evaluation.validator_address),
                    "Evaluation already submitted by validator {} for round {}",
                    evaluation.validator_address,
                    round
                );

                Self::validate_evaluation(ctx, &evaluation)?;

                // Cryptographically verify Hardware TEE Attestation Quote if present
                if let Some(quote) = &evaluation.attestation_quote {
                    if let Err(e) = self.tee_verifier.verify_quote(quote, task_id, round, &evaluation.ranking) {
                        // Anti-Fraud: On production/mainnet (simulation disallowed),
                        // any fraudulent or spoofed quote immediately slashes the validator!
                        if !self.tee_verifier.allow_simulation {
                            self.slash_validator(
                                &evaluation.validator_address,
                                &format!("Fraudulent TEE attestation quote rejected: {}", e),
                            );
                        }
                        anyhow::bail!("On-Chain TEE Attestation verification rejected (validator slashed): {}", e);
                    }
                } else if self.tee_verifier.enforce_attestation {
                    anyhow::bail!(
                        "On-Chain TEE Attestation rejected: missing required hardware quote"
                    );
                }

                ctx.evaluations
                    .insert(evaluation.validator_address.clone(), evaluation);
            }

            Transaction::SlashValidator {
                reporter,
                nonce: _,
                evidence,
            } => {
                ensure!(
                    &reporter == sender,
                    "Unauthorized: Transaction sender {} does not match reporter address {}",
                    sender,
                    reporter
                );
                ensure!(
                    evidence.vote_a.validator == evidence.validator
                        && evidence.vote_b.validator == evidence.validator,
                    "Invalid equivocation evidence: vote validator mismatch"
                );
                ensure!(
                    evidence.vote_a.height == evidence.height
                        && evidence.vote_b.height == evidence.height
                        && evidence.vote_a.round == evidence.round
                        && evidence.vote_b.round == evidence.round
                        && evidence.vote_a.vote_type == evidence.vote_type
                        && evidence.vote_b.vote_type == evidence.vote_type,
                    "Invalid equivocation evidence: height, round, or vote_type mismatch"
                );
                ensure!(
                    evidence.vote_a.block_hash != evidence.vote_b.block_hash,
                    "Invalid equivocation evidence: votes have identical block hashes (not double voting)"
                );
                evidence
                    .vote_a
                    .verify_signature()
                    .map_err(|e| anyhow::anyhow!("Invalid signature on vote A in equivocation evidence: {}", e))?;
                evidence
                    .vote_b
                    .verify_signature()
                    .map_err(|e| anyhow::anyhow!("Invalid signature on vote B in equivocation evidence: {}", e))?;

                let target = evidence.validator.clone();
                ensure!(
                    reporter != target,
                    "SlashValidator rejected: validator cannot report themselves to claim whistleblower bounty"
                );
                ensure!(
                    !self.is_slashed(&reporter),
                    "SlashValidator rejected: reporter is slashed and permanently banned"
                );
                ensure!(
                    !self.is_slashed(&target),
                    "Validator {} is already slashed and banned",
                    target
                );

                let target_canon = Self::canonical_account(&target);
                let target_bal = self.balance_of(&target);
                self.slashed_validators.insert(target.clone());
                self.slashed_validators.insert(target_canon.clone());

                // Whistleblower reward: 80% burned, 20% whistleblower bounty to reporter
                if target_bal > 0 {
                    let bounty = (target_bal as f64 * 0.20).round() as u128;
                    let burned = target_bal.saturating_sub(bounty);
                    self.balances.remove(&target_canon);
                    self.total_burned += burned;
                    self.credit(&reporter, bounty);
                }
            }

            Transaction::Transfer {
                from,
                to,
                amount,
                nonce: _,
            } => {
                ensure!(
                    &from == sender,
                    "Unauthorized: Transaction sender {} does not match from address {}",
                    sender,
                    from
                );
                ensure!(
                    !self.circuit_breaker_active,
                    "Transfer rejected: Emergency Safe Mode is ACTIVE! Circuit breaker triggered at block #{:?}",
                    self.circuit_breaker_triggered_at_block
                );
                let from_canon = Self::canonical_account(&from);
                let to_canon = Self::canonical_account(&to);
                ensure!(
                    from_canon != to_canon,
                    "Transfer rejected: cannot transfer tokens to oneself"
                );
                ensure!(
                    !self.is_slashed(&from),
                    "Transfer rejected: Sender {} has been slashed and banned",
                    from
                );
                ensure!(
                    !self.is_slashed(&to),
                    "Transfer rejected: Recipient {} has been slashed and banned",
                    to
                );
                ensure!(amount > 0, "Transfer amount must be greater than 0");
                self.debit(&from, amount)?;
                self.credit(&to, amount);
            }
        }

        // Increment sender account nonce upon successful transaction application
        *self.nonces.entry(sender.clone()).or_insert(0) += 1;

        Ok(())
    }

    /// Execute Relative Consensus and finalize round:
    /// Select Top-1 winner, payout bounty reward from escrow, and record RoundSummary.
    pub fn finalize_round(
        &mut self,
        task_id: u64,
        round: usize,
        evolved_model_cid: String,
        pre_merge_loss: f64,
        post_merge_loss: f64,
        round_bounty: u128,
    ) -> Result<RoundSummary> {
        let (revealed_miners, eval_list, base_model_cid, winner, winning_adapter_cid, consensus) = {
            let ctx = self
                .round_contexts
                .get_mut(&(task_id, round))
                .ok_or_else(|| anyhow::anyhow!("Round context not found"))?;

            ensure!(
                ctx.phase == RoundPhase::MergePhase,
                "Cannot finalize round: round is in phase {:?}, expected MergePhase",
                ctx.phase
            );

            ensure!(
                !ctx.evaluations.is_empty(),
                "Cannot finalize round with zero validator evaluations"
            );

            let revealed_miners: Vec<AccountId> = ctx.reveals.keys().cloned().collect();
            let eval_list: Vec<ValidatorEvaluation> = ctx.evaluations.values().cloned().collect();

            // Run Relative Consensus (Borda Count rank aggregation)
            let consensus = RelativeConsensusEngine::aggregate(&eval_list, &revealed_miners)
                .ok_or_else(|| anyhow::anyhow!("Failed to compute relative consensus"))?;

            let winner = consensus.winner.clone();
            let winning_adapter_cid = ctx
                .reveals
                .get(&winner)
                .map(|r| r.adapter_cid.clone())
                .unwrap_or_default();

            let base_model_cid = ctx.base_model_cid.clone();
            ctx.consensus = Some(consensus.clone());
            ctx.phase = RoundPhase::Completed;

            (
                revealed_miners,
                eval_list,
                base_model_cid,
                winner,
                winning_adapter_cid,
                consensus,
            )
        };

        // Payout bounty reward according to the task's configured RewardDistribution strategy
        let task_reward_dist = self
            .tasks
            .get(&task_id)
            .map(|t| t.reward_distribution)
            .unwrap_or_default();

        let mut reward_distributions: Vec<(AccountId, u128)> = Vec::new();
        let mut validator_rewards: Vec<(AccountId, u128)> = Vec::new();
        let mut node_rewards: Vec<(AccountId, u128)> = Vec::new();
        let mut burned_bounty: u128 = 0;
        // Weapon 4: Emergency Circuit Breaker (Safe Mode)
        // If Safe Mode is active, halt round bounty payouts immediately
        if self.circuit_breaker_active {
            anyhow::bail!(
                "Emergency Safe Mode is ACTIVE! Circuit breaker triggered at block #{:?}. Round finalization payouts halted.",
                self.circuit_breaker_triggered_at_block
            );
        }

        let escrow_val = self.escrows.get(&task_id).copied().unwrap_or(0);
        let amount = if round_bounty > 0 {
            round_bounty.min(escrow_val)
        } else {
            escrow_val
        };

        let total_available_bounty = if amount > 0 {
            // Verify rolling window payout velocity cap before deducting from escrow
            self.check_and_record_payout(amount)?;
            if let Some(escrow) = self.escrows.get_mut(&task_id) {
                *escrow -= amount;
            }
            amount
        } else {
            0
        };

        if total_available_bounty > 0 {
            // Dynamic Deflationary Burn Mechanism:
            // Maximum 10% burn rate when client activity is abundant.
            // When client activity is scarce (e.g. only 1 active task/client on the network),
            // the burn rate drops down toward 1% or ~0% (0.005) to incentivize miners and validators.
            // Formula: burn_pct = clamp(0.005 + 0.015 * (num_tasks - 1), 0.005, 0.10)
            let active_tasks_count = self.tasks.len();
            let burn_pct =
                (0.005 + (active_tasks_count.saturating_sub(1) as f64) * 0.015).clamp(0.005, 0.10);
            burned_bounty = ((total_available_bounty as f64) * burn_pct).round() as u128;
            self.total_burned += burned_bounty;

            let distributable_bounty = total_available_bounty.saturating_sub(burned_bounty);

            // Three-Tier Dynamic Supply-Demand Elasticity Model:
            // 1. Tier 1 - Network & Storage Infrastructure Nodes (IPFS relay & consensus maintenance)
            //    Scales with network storage throughput / load (revealed candidate models to pin & gossip)
            //    Elastic range: 5% up to 15% max when model traffic is heavy.
            // 2. Tier 2 - TEE Hardware Validators (Intel SGX / AMD SEV)
            //    Elastic range: 15% up to 45% based on miner-to-validator supply scarcity ratio.
            // 3. Tier 3 - Miner Pool (Competitive Autograd Training)
            //    Receives remaining majority share distributed via winner/top-k policy.

            let num_miners = revealed_miners.len().max(1);
            let num_validators = eval_list.len().max(1);
            let supply_ratio = (num_miners as f64) / (num_validators as f64);

            // Tier 1: Storage/Relay Node Elasticity:
            // Base 5%, scales +1% per revealed model adapter being stored/relayed on IPFS, capped at 15%.
            let node_share_pct =
                (0.05 + (revealed_miners.len().saturating_sub(1) as f64) * 0.01).clamp(0.05, 0.15);
            let node_pool = ((distributable_bounty as f64) * node_share_pct).round() as u128;

            // Tier 2: TEE Validator Elasticity:
            // Base 15%, scales +5% per unit of miner-to-validator imbalance, capped at 45%.
            let validator_share_pct = (0.15 + (supply_ratio - 1.0) * 0.05).clamp(0.15, 0.45);
            let val_pool = if !eval_list.is_empty() {
                ((distributable_bounty as f64) * validator_share_pct).round() as u128
            } else {
                0
            };

            // Tier 3: Miner Pool receives remaining bounty
            let miner_pool = distributable_bounty
                .saturating_sub(node_pool)
                .saturating_sub(val_pool);

            // 1. Distribute Storage & Network Node Rewards
            if node_pool > 0 {
                let storage_node = AccountId::storage_gateway();
                self.credit(&storage_node, node_pool);
                node_rewards.push((storage_node, node_pool));
            }

            // 2. Distribute Validator Rewards evenly among authentic evaluating TEE Validators
            if val_pool > 0 && !eval_list.is_empty() {
                let per_val_reward = val_pool / (eval_list.len() as u128);
                let mut remaining_val_pool = val_pool;
                for (idx, val_eval) in eval_list.iter().enumerate() {
                    let amount = if idx == eval_list.len() - 1 {
                        remaining_val_pool
                    } else {
                        per_val_reward.min(remaining_val_pool)
                    };
                    remaining_val_pool = remaining_val_pool.saturating_sub(amount);
                    self.credit(&val_eval.validator_address, amount);
                    validator_rewards.push((val_eval.validator_address.clone(), amount));
                }
            }

            // 3. Distribute Miner Rewards according to TaskSpec RewardDistribution policy
            if miner_pool > 0 {
                match task_reward_dist {
                    crate::blockchain::types::RewardDistribution::WinnerTakesAll => {
                        self.credit(&winner, miner_pool);
                        reward_distributions.push((winner.clone(), miner_pool));
                    }
                    crate::blockchain::types::RewardDistribution::TopKDecay {
                        top_k,
                        decay_rate,
                    } => {
                        let k = top_k.min(consensus.consensus_ranking.len()).max(1);
                        let valid_decay = if decay_rate > 0.0 && decay_rate < 1.0 {
                            decay_rate
                        } else {
                            0.5
                        };

                        let mut weights: Vec<f64> =
                            (0..k).map(|i| (1.0 - valid_decay).powi(i as i32)).collect();
                        let sum_weights: f64 = weights.iter().sum();
                        if sum_weights > 0.0 {
                            for w in &mut weights {
                                *w /= sum_weights;
                            }
                        }

                        let mut remaining_to_distribute = miner_pool;
                        for (i, miner_id) in consensus.consensus_ranking.iter().take(k).enumerate()
                        {
                            let amount = if i == k - 1 {
                                remaining_to_distribute
                            } else {
                                let share = (miner_pool as f64 * weights[i]).round() as u128;
                                share.min(remaining_to_distribute)
                            };
                            remaining_to_distribute =
                                remaining_to_distribute.saturating_sub(amount);
                            self.credit(miner_id, amount);
                            reward_distributions.push((miner_id.clone(), amount));
                        }
                    }
                    crate::blockchain::types::RewardDistribution::TopKBordaWeighted { top_k } => {
                        let k = top_k.min(consensus.borda_scores.len()).max(1);
                        let top_borda: Vec<(AccountId, usize)> =
                            consensus.borda_scores.iter().take(k).cloned().collect();
                        let total_borda: usize = top_borda.iter().map(|(_, s)| *s).sum();

                        let mut remaining_to_distribute = miner_pool;
                        if total_borda > 0 {
                            for (i, (miner_id, score)) in top_borda.iter().enumerate() {
                                let amount = if i == k - 1 {
                                    remaining_to_distribute
                                } else {
                                    let share = (miner_pool as f64 * (*score as f64)
                                        / (total_borda as f64))
                                        .round()
                                        as u128;
                                    share.min(remaining_to_distribute)
                                };
                                remaining_to_distribute =
                                    remaining_to_distribute.saturating_sub(amount);
                                self.credit(miner_id, amount);
                                reward_distributions.push((miner_id.clone(), amount));
                            }
                        } else {
                            self.credit(&winner, miner_pool);
                            reward_distributions.push((winner.clone(), miner_pool));
                        }
                    }
                }
            }
        }

        let summary = RoundSummary {
            round_number: round,
            base_model_cid,
            evolved_model_cid,
            winning_miner: winner,
            winning_adapter_cid,
            consensus_ranking: consensus.consensus_ranking,
            borda_scores: consensus.borda_scores,
            pre_merge_loss,
            post_merge_loss,
            bounty_awarded: total_available_bounty,
            reward_distributions,
            validator_rewards,
            node_rewards,
            burned_bounty,
            heterogeneous_tee_quorum: consensus.heterogeneous_quorum_achieved,
            tee_diversity_count: consensus.tee_diversity_count,
        };

        const MAX_ROUND_HISTORY: usize = 1000;
        if self.round_history.len() >= MAX_ROUND_HISTORY {
            self.round_history.remove(0);
        }

        self.round_history.push(summary.clone());
        Ok(summary)
    }
}
