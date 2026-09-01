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

    /// Advance block height by 1.
    pub fn advance_block(&mut self) -> u32 {
        self.block_height += 1;
        self.block_height
    }

    /// Helper to compute commit hash: SHA256(adapter_hash || salt).
    pub fn compute_commit_hash(adapter_hash: &[u8; 32], salt: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(adapter_hash);
        hasher.update(salt);
        hasher.finalize().into()
    }

    /// Initialize a new round context for a task.
    pub fn start_round(&mut self, task_id: u64, round_number: usize, base_model_cid: String) -> Result<()> {
        ensure!(self.tasks.contains_key(&task_id), "Task ID does not exist");
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
            } => {
                ensure!(
                    &client == sender,
                    "Unauthorized: Transaction sender {} does not match client address {}",
                    sender,
                    client
                );
                let client_bal = self.balance_of(&client);
                ensure!(
                    client_bal >= bounty_pool,
                    "Client balance insufficient for bounty escrow"
                );

                // Deduct from client balance and lock into escrow
                *self.balances.get_mut(&client).unwrap() -= bounty_pool;
                let task_id = self.next_task_id;
                self.next_task_id += 1;

                self.escrows.insert(task_id, bounty_pool);

                let task_spec = TaskSpec {
                    task_id,
                    client_address: client,
                    base_model_id,
                    base_model_hash,
                    dataset_cid,
                    peft_method,
                    max_rank,
                    target_modules,
                    bounty_pool,
                    epoch_end_block: self.block_height + epoch_blocks,
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
                let ctx = self
                    .round_contexts
                    .get_mut(&(task_id, round))
                    .ok_or_else(|| anyhow::anyhow!("Round context not found"))?;

                ensure!(
                    ctx.phase == RoundPhase::CommitPhase,
                    "Commit rejected: Round is in phase {:?}",
                    ctx.phase
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

        // Payout bounty reward to winning miner from escrow
        let bounty_payout = if let Some(escrow) = self.escrows.get_mut(&task_id) {
            let amount = if round_bounty > 0 {
                round_bounty.min(*escrow)
            } else {
                *escrow
            };
            *escrow -= amount;
            *self.balances.entry(winner.clone()).or_insert(0) += amount;
            amount
        } else {
            0
        };

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
            bounty_awarded: bounty_payout,
        };

        self.round_history.push(summary.clone());
        Ok(summary)
    }
}
