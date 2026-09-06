use crate::blockchain::relative_consensus::{ConsensusResult, RelativeConsensusEngine};
use crate::blockchain::transactions::Transaction;
use crate::blockchain::types::{
    AccountId, CommitRecord, RevealRecord, RoundPhase, RoundSummary, TaskSpec, ValidatorEvaluation,
};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Runtime state of a specific tournament round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundContext {
    pub task_id: u64,
    pub round_number: usize,
    pub phase: RoundPhase,
    pub base_model_cid: String,
    pub commits: HashMap<AccountId, CommitRecord>,
    pub reveals: HashMap<AccountId, RevealRecord>,
    pub evaluations: HashMap<AccountId, ValidatorEvaluation>,
    pub consensus: Option<ConsensusResult>,
}

impl RoundContext {
    pub fn new(task_id: u64, round_number: usize, base_model_cid: String) -> Self {
        Self {
            task_id,
            round_number,
            phase: RoundPhase::CommitPhase,
            base_model_cid,
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
    #[serde(skip)]
    pub tee_verifier: OnChainTeeVerifier,
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
            tee_verifier: OnChainTeeVerifier::default(),
        }
    }

    /// Deposit native tokens to an account balance.
    pub fn mint(&mut self, account: AccountId, amount: u128) {
        *self.balances.entry(account).or_insert(0) += amount;
    }

    /// Get current balance of an account.
    pub fn balance_of(&self, account: &AccountId) -> u128 {
        self.balances.get(account).copied().unwrap_or(0)
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
            if let Some(task) = self.tasks.get(task_id) {
                // Determine phase progression intervals
                // 1. In CommitPhase: move to RevealPhase only when epoch deadline is reached
                if ctx.phase == RoundPhase::CommitPhase && !ctx.commits.is_empty() {
                    if current_block >= task.epoch_end_block {
                        ctx.phase = RoundPhase::RevealPhase;
                    }
                }
                // 2. In RevealPhase: if all committed miners revealed, or time elapsed, move to EvaluationPhase
                else if ctx.phase == RoundPhase::RevealPhase && !ctx.reveals.is_empty() {
                    let all_revealed = ctx.commits.keys().all(|m| ctx.reveals.contains_key(m));
                    if all_revealed || current_block >= task.epoch_end_block + 2 {
                        ctx.phase = RoundPhase::EvaluationPhase;
                    }
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
            ensure!(ranked.insert(miner), "Evaluation ranking contains duplicate miner: {}", miner);
        }

        for (miner, loss) in &evaluation.loss_scores {
            ensure!(ranked.contains(miner), "Loss score contains unranked miner: {}", miner);
            ensure!(loss.is_finite() && *loss >= 0.0, "Loss score must be finite and non-negative");
        }
        for (miner, accuracy) in &evaluation.accuracy_scores {
            ensure!(ranked.contains(miner), "Accuracy score contains unranked miner: {}", miner);
            ensure!(
                accuracy.is_finite() && (0.0..=1.0).contains(accuracy),
                "Accuracy score must be finite and within [0, 1]"
            );
        }

        Ok(())
    }

    /// Initialize a new round context for a task.
    pub fn start_round(&mut self, task_id: u64, round_number: usize, base_model_cid: String) -> Result<()> {
        ensure!(self.tasks.contains_key(&task_id), "Task ID does not exist");
        ensure!(
            !self.round_contexts.contains_key(&(task_id, round_number)),
            "Round {} already initialized for task {}",
            round_number,
            task_id
        );
        let context = RoundContext::new(task_id, round_number, base_model_cid);
        self.round_contexts.insert((task_id, round_number), context);
        Ok(())
    }

    /// Transition round phase.
    pub fn set_round_phase(&mut self, task_id: u64, round_number: usize, phase: RoundPhase) -> Result<()> {
        let ctx = self
            .round_contexts
            .get_mut(&(task_id, round_number))
            .ok_or_else(|| anyhow::anyhow!("Round context not found"))?;
        ctx.phase = phase;
        Ok(())
    }

    /// Process an on-chain transaction deterministically.
    pub fn apply_transaction(&mut self, tx: Transaction, sender: &AccountId) -> Result<()> {
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
                ensure!(bounty_pool > 0, "Bounty pool must be greater than 0 tokens");
                ensure!(epoch_blocks >= 5, "Epoch duration must be at least 5 blocks");
                ensure!(max_rank > 0 && max_rank <= 256, "Max rank must be between 1 and 256");
                ensure!(!base_model_id.is_empty() && base_model_id.len() <= 128, "Base model ID must be 1-128 bytes");
                ensure!(!dataset_cid.is_empty() && dataset_cid.len() <= 128, "Dataset CID must be 1-128 bytes");
                ensure!(!target_modules.is_empty() && target_modules.len() <= 32, "Target modules must contain 1-32 items");
                ensure!(
                    target_modules.iter().all(|m| !m.is_empty() && m.len() <= 64),
                    "Each target module name must be 1-64 bytes"
                );

                let client_bal = self.balance_of(&client);
                ensure!(
                    client_bal >= bounty_pool,
                    "Client balance insufficient for bounty escrow"
                );

                // Deduct from client balance and lock into escrow
                if let Some(bal) = self.balances.get_mut(&client) {
                    *bal = bal.saturating_sub(bounty_pool);
                }
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

                // Auto initialize round context if not yet started
                if !self.round_contexts.contains_key(&(task_id, round)) {
                    if let Some(task) = self.tasks.get(&task_id) {
                        let base_model_cid_str = task.base_model_id_str();
                        let context = RoundContext::new(task_id, round, base_model_cid_str.to_string());
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
                    self.tee_verifier
                        .verify_quote(quote, task_id, round, &evaluation.ranking)
                        .map_err(|e| anyhow::anyhow!("On-Chain TEE Attestation verification rejected: {}", e))?;
                } else if self.tee_verifier.enforce_attestation {
                    anyhow::bail!("On-Chain TEE Attestation rejected: missing required hardware quote");
                }

                ctx.evaluations
                    .insert(evaluation.validator_address.clone(), evaluation);
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

        ctx.consensus = Some(consensus.clone());
        ctx.phase = RoundPhase::Completed;

        // Payout bounty reward according to the task's configured RewardDistribution strategy
        let task_reward_dist = self
            .tasks
            .get(&task_id)
            .map(|t| t.reward_distribution)
            .unwrap_or_default();

        let mut reward_distributions: Vec<(AccountId, u128)> = Vec::new();
        let mut validator_rewards: Vec<(AccountId, u128)> = Vec::new();
        let total_available_bounty = if let Some(escrow) = self.escrows.get_mut(&task_id) {
            let amount = if round_bounty > 0 {
                round_bounty.min(*escrow)
            } else {
                *escrow
            };
            *escrow -= amount;
            amount
        } else {
            0
        };

        if total_available_bounty > 0 {
            // Dynamic Supply-Demand Elasticity Model:
            // When TEE validators are scarce relative to miners, validator reward ratio increases
            // up to 50% to incentivize high-grade hardware provisioning.
            // Base validator ratio: 20%. Each miner-to-validator imbalance unit increases share.
            let num_miners = revealed_miners.len().max(1);
            let num_validators = eval_list.len().max(1);
            let supply_ratio = (num_miners as f64) / (num_validators as f64);
            // Elasticity formula: min 15%, scales up to 50% max when validators are scarce
            let validator_share_pct = (0.15 + (supply_ratio - 1.0) * 0.05).clamp(0.15, 0.50);

            let val_pool = if !eval_list.is_empty() {
                ((total_available_bounty as f64) * validator_share_pct).round() as u128
            } else {
                0
            };
            let miner_pool = total_available_bounty.saturating_sub(val_pool);

            // 1. Distribute Validator Rewards evenly among authentic evaluating TEE Validators
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
                    *self.balances.entry(val_eval.validator_address.clone()).or_insert(0) += amount;
                    validator_rewards.push((val_eval.validator_address.clone(), amount));
                }
            }

            // 2. Distribute Miner Rewards according to TaskSpec RewardDistribution policy
            if miner_pool > 0 {
                match task_reward_dist {
                    crate::blockchain::types::RewardDistribution::WinnerTakesAll => {
                        *self.balances.entry(winner.clone()).or_insert(0) += miner_pool;
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

                        let mut weights: Vec<f64> = (0..k)
                            .map(|i| (1.0 - valid_decay).powi(i as i32))
                            .collect();
                        let sum_weights: f64 = weights.iter().sum();
                        if sum_weights > 0.0 {
                            for w in &mut weights {
                                *w /= sum_weights;
                            }
                        }

                        let mut remaining_to_distribute = miner_pool;
                        for (i, miner_id) in consensus.consensus_ranking.iter().take(k).enumerate() {
                            let amount = if i == k - 1 {
                                remaining_to_distribute
                            } else {
                                let share = (miner_pool as f64 * weights[i]).round() as u128;
                                share.min(remaining_to_distribute)
                            };
                            remaining_to_distribute = remaining_to_distribute.saturating_sub(amount);
                            *self.balances.entry(miner_id.clone()).or_insert(0) += amount;
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
                                        .round() as u128;
                                    share.min(remaining_to_distribute)
                                };
                                remaining_to_distribute =
                                    remaining_to_distribute.saturating_sub(amount);
                                *self.balances.entry(miner_id.clone()).or_insert(0) += amount;
                                reward_distributions.push((miner_id.clone(), amount));
                            }
                        } else {
                            *self.balances.entry(winner.clone()).or_insert(0) += miner_pool;
                            reward_distributions.push((winner.clone(), miner_pool));
                        }
                    }
                }
            }
        }

        let summary = RoundSummary {
            round_number: round,
            base_model_cid: ctx.base_model_cid.clone(),
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
        };

        const MAX_ROUND_HISTORY: usize = 1000;
        if self.round_history.len() >= MAX_ROUND_HISTORY {
            self.round_history.remove(0);
        }

        self.round_history.push(summary.clone());
        Ok(summary)
    }
}
