use crate::blockchain::types::AccountId;
use crate::consensus::types::ConsensusValidator;
use serde::{Deserialize, Serialize};

/// Active validator set participating in BFT consensus.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidatorSet {
    pub validators: Vec<ConsensusValidator>,
    pub total_voting_power: u64,
}

impl ValidatorSet {
    pub fn new(validators: Vec<ConsensusValidator>) -> Self {
        let total_voting_power: u64 = validators.iter().map(|v| v.voting_power).sum();
        Self {
            validators,
            total_voting_power,
        }
    }

    /// Calculate strict 2/3+ threshold required for BFT quorum ($> 2/3$).
    pub fn two_thirds_threshold(&self) -> u64 {
        (self.total_voting_power * 2) / 3 + 1
    }

    /// Deterministic weighted round-robin proposer selection based on height & round.
    pub fn get_proposer(&self, height: u64, round: u32) -> AccountId {
        if self.validators.is_empty() {
            return AccountId::new("0x0000000000000000000000000000000000000000");
        }
        let index = ((height + round as u64) as usize) % self.validators.len();
        self.validators[index].address.clone()
    }

    /// Lookup voting power for a specific validator.
    pub fn get_voting_power(&self, address: &AccountId) -> u64 {
        self.validators
            .iter()
            .find(|v| &v.address == address)
            .map(|v| v.voting_power)
            .unwrap_or(0)
    }

    /// Check if an account is an active validator.
    pub fn contains(&self, address: &AccountId) -> bool {
        self.validators.iter().any(|v| &v.address == address)
    }
}
