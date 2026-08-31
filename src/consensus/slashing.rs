use crate::blockchain::types::AccountId;
use crate::consensus::types::{Vote, VoteType};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Proof of Byzantine equivocation (double-voting).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EquivocationEvidence {
    pub validator: AccountId,
    pub height: u64,
    pub round: u32,
    pub vote_type: VoteType,
    pub vote_a: Vote,
    pub vote_b: Vote,
}

/// Slashing engine detecting and penalizing Byzantine behaviors.
#[derive(Debug, Clone, Default)]
pub struct SlashingEngine {
    /// History of votes by (validator, height, round, vote_type)
    vote_records: HashMap<(AccountId, u64, u32, u8), Vote>,
    pub slashed_validators: HashMap<AccountId, u64>,
}

impl SlashingEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a new vote and check for equivocation.
    /// If double-voting is detected, returns `Some(EquivocationEvidence)`.
    pub fn check_vote(&mut self, vote: &Vote) -> Result<Option<EquivocationEvidence>> {
        let key = (
            vote.validator.clone(),
            vote.height,
            vote.round,
            vote.vote_type as u8,
        );

        if let Some(existing_vote) = self.vote_records.get(&key) {
            if existing_vote.block_hash != vote.block_hash {
                // Byzantine double-voting detected!
                let evidence = EquivocationEvidence {
                    validator: vote.validator.clone(),
                    height: vote.height,
                    round: vote.round,
                    vote_type: vote.vote_type,
                    vote_a: existing_vote.clone(),
                    vote_b: vote.clone(),
                };

                *self.slashed_validators.entry(vote.validator.clone()).or_insert(0) += 1;
                return Ok(Some(evidence));
            }
        } else {
            self.vote_records.insert(key, vote.clone());
        }

        Ok(None)
    }

    /// Check if a validator has been slashed.
    pub fn is_slashed(&self, validator: &AccountId) -> bool {
        self.slashed_validators.contains_key(validator)
    }
}
