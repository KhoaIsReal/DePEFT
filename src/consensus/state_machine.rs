use crate::blockchain::types::AccountId;
use crate::consensus::types::{Block, BlockCommit, ConsensusValidator, Vote, VoteType};
use crate::consensus::validator_set::ValidatorSet;
use crate::crypto::{AccountKeypair, SignedTransaction};
use anyhow::{ensure, Result};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// BFT 2-Phase Commit state transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BftStep {
    NewHeight,
    Propose,
    Prevote,
    Precommit,
    Commit,
}

/// Active consensus state for a specific height and round.
#[derive(Debug, Clone)]
pub struct BftRoundState {
    pub height: u64,
    pub round: u32,
    pub step: BftStep,
    pub proposal: Option<Block>,
    pub prevotes: HashMap<AccountId, Vote>,
    pub precommits: HashMap<AccountId, Vote>,
    pub locked_block: Option<Block>,
    pub locked_round: Option<u32>,
}

impl BftRoundState {
    pub fn new(height: u64, round: u32) -> Self {
        Self {
            height,
            round,
            step: BftStep::NewHeight,
            proposal: None,
            prevotes: HashMap::new(),
            precommits: HashMap::new(),
            locked_block: None,
            locked_round: None,
        }
    }
}

/// Byzantine Fault Tolerant (BFT) Consensus State Machine.
#[derive(Debug, Clone)]
pub struct BftEngine {
    pub validator_set: ValidatorSet,
    pub blockchain: Vec<Block>,
    pub current_height: u64,
    pub current_round: u32,
    pub state_root: [u8; 32],
    pub round_state: BftRoundState,
}

impl BftEngine {
    /// Initialize BFT consensus engine with a validator set and genesis state root.
    pub fn new(validators: Vec<ConsensusValidator>, genesis_state_root: [u8; 32]) -> Self {
        let validator_set = ValidatorSet::new(validators);
        let height = 1;
        let round = 0;
        let round_state = BftRoundState::new(height, round);

        Self {
            validator_set,
            blockchain: Vec::new(),
            current_height: height,
            current_round: round,
            state_root: genesis_state_root,
            round_state,
        }
    }

    /// Retrieve the previous block hash on the finalized chain (or zero hash for genesis).
    pub fn last_block_hash(&self) -> [u8; 32] {
        self.blockchain
            .last()
            .map(|b| b.block_hash())
            .unwrap_or([0u8; 32])
    }

    /// Propose a new candidate block for the current height & round.
    pub fn create_proposal(
        &mut self,
        proposer_keypair: &AccountKeypair,
        transactions: Vec<SignedTransaction>,
    ) -> Result<Block> {
        let expected_proposer = self
            .validator_set
            .get_proposer(self.current_height, self.current_round);
        ensure!(
            proposer_keypair.account_id() == expected_proposer,
            "Unauthorized proposer for height {} round {}: expected {}",
            self.current_height,
            self.current_round,
            expected_proposer
        );

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let block = Block::new(
            self.current_height,
            self.current_round,
            self.last_block_hash(),
            self.state_root,
            proposer_keypair.account_id(),
            transactions,
            timestamp,
        );

        self.round_state.proposal = Some(block.clone());
        self.round_state.step = BftStep::Prevote;

        Ok(block)
    }

    /// Cast a cryptographic PREVOTE for a candidate block hash.
    pub fn cast_prevote(
        &self,
        keypair: &AccountKeypair,
        block_hash: Option<[u8; 32]>,
    ) -> Result<Vote> {
        ensure!(
            self.validator_set.contains(&keypair.account_id()),
            "Account {} is not in active validator set",
            keypair.account_id()
        );

        let sign_bytes = Vote::sign_bytes(
            VoteType::Prevote,
            self.current_height,
            self.current_round,
            block_hash,
        );
        let signature = keypair.sign_message(&sign_bytes);

        Ok(Vote {
            vote_type: VoteType::Prevote,
            height: self.current_height,
            round: self.current_round,
            block_hash,
            validator: keypair.account_id(),
            signature,
        })
    }

    /// Cast a cryptographic PRECOMMIT for a candidate block hash.
    pub fn cast_precommit(
        &self,
        keypair: &AccountKeypair,
        block_hash: Option<[u8; 32]>,
    ) -> Result<Vote> {
        ensure!(
            self.validator_set.contains(&keypair.account_id()),
            "Account {} is not in active validator set",
            keypair.account_id()
        );

        let sign_bytes = Vote::sign_bytes(
            VoteType::Precommit,
            self.current_height,
            self.current_round,
            block_hash,
        );
        let signature = keypair.sign_message(&sign_bytes);

        Ok(Vote {
            vote_type: VoteType::Precommit,
            height: self.current_height,
            round: self.current_round,
            block_hash,
            validator: keypair.account_id(),
            signature,
        })
    }

    /// Record a validator vote and process BFT quorum transitions.
    /// Returns `Some(finalized_block)` when 2/3+ Precommits commit the block.
    pub fn add_vote(&mut self, vote: Vote) -> Result<Option<Block>> {
        ensure!(
            vote.height == self.current_height && vote.round == self.current_round,
            "Vote height/round mismatch: got ({}, {}), expected ({}, {})",
            vote.height,
            vote.round,
            self.current_height,
            self.current_round
        );

        ensure!(
            self.validator_set.contains(&vote.validator),
            "Validator {} is not recognized",
            vote.validator
        );

        match vote.vote_type {
            VoteType::Prevote => {
                self.round_state.prevotes.insert(vote.validator.clone(), vote);
                self.check_prevote_quorum()
            }
            VoteType::Precommit => {
                self.round_state.precommits.insert(vote.validator.clone(), vote);
                self.check_precommit_quorum()
            }
        }
    }

    /// Check if 2/3+ Prevotes have been gathered (Proof-of-Lock).
    fn check_prevote_quorum(&mut self) -> Result<Option<Block>> {
        let threshold = self.validator_set.two_thirds_threshold();

        // Tally votes per block hash
        let mut tally: HashMap<Option<[u8; 32]>, u64> = HashMap::new();
        for vote in self.round_state.prevotes.values() {
            let power = self.validator_set.get_voting_power(&vote.validator);
            *tally.entry(vote.block_hash).or_insert(0) += power;
        }

        for (block_hash_opt, total_power) in tally {
            if total_power >= threshold {
                if let Some(hash) = block_hash_opt {
                    if let Some(proposal) = &self.round_state.proposal {
                        if proposal.block_hash() == hash {
                            // Lock on candidate block
                            self.round_state.locked_block = Some(proposal.clone());
                            self.round_state.locked_round = Some(self.current_round);
                            self.round_state.step = BftStep::Precommit;
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    /// Check if 2/3+ Precommits have been gathered to finalize the block.
    fn check_precommit_quorum(&mut self) -> Result<Option<Block>> {
        let threshold = self.validator_set.two_thirds_threshold();

        let mut tally: HashMap<Option<[u8; 32]>, u64> = HashMap::new();
        for vote in self.round_state.precommits.values() {
            let power = self.validator_set.get_voting_power(&vote.validator);
            *tally.entry(vote.block_hash).or_insert(0) += power;
        }

        for (block_hash_opt, total_power) in tally {
            if total_power >= threshold {
                if let Some(hash) = block_hash_opt {
                    if let Some(mut proposal) = self.round_state.proposal.clone() {
                        if proposal.block_hash() == hash {
                            // Assemble 2/3+ Precommit cryptographic signatures
                            let signatures: Vec<(AccountId, Vec<u8>)> = self
                                .round_state
                                .precommits
                                .values()
                                .filter(|v| v.block_hash == Some(hash))
                                .map(|v| (v.validator.clone(), v.signature.clone()))
                                .collect();

                            let commit = BlockCommit {
                                height: self.current_height,
                                round: self.current_round,
                                block_hash: hash,
                                signatures,
                            };

                            proposal.commit = Some(commit);

                            // Append block to finalized immutable blockchain
                            self.blockchain.push(proposal.clone());

                            // Advance consensus to next height
                            self.current_height += 1;
                            self.current_round = 0;
                            self.round_state = BftRoundState::new(self.current_height, self.current_round);

                            return Ok(Some(proposal));
                        }
                    }
                }
            }
        }

        Ok(None)
    }
}
