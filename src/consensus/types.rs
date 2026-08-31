use crate::blockchain::types::AccountId;
use crate::crypto::SignedTransaction;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

/// Block Header containing cryptographic commitments to state and transactions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockHeader {
    pub height: u64,
    pub round: u32,
    pub prev_block_hash: [u8; 32],
    pub state_root: [u8; 32],
    pub tx_root: [u8; 32],
    pub proposer: AccountId,
    pub timestamp: u64,
}

impl BlockHeader {
    pub fn compute_hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(&self.height.to_be_bytes());
        hasher.update(&self.round.to_be_bytes());
        hasher.update(&self.prev_block_hash);
        hasher.update(&self.state_root);
        hasher.update(&self.tx_root);
        hasher.update(self.proposer.0.as_bytes());
        hasher.update(&self.timestamp.to_be_bytes());
        hasher.finalize().into()
    }
}

/// Cryptographic 2/3+ Commit proof signed by the validator set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockCommit {
    pub height: u64,
    pub round: u32,
    pub block_hash: [u8; 32],
    pub signatures: Vec<(AccountId, Vec<u8>)>,
}

/// Finalized Block on the DePEFT App-Chain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Block {
    pub header: BlockHeader,
    pub transactions: Vec<SignedTransaction>,
    pub commit: Option<BlockCommit>,
}

impl Block {
    pub fn new(
        height: u64,
        round: u32,
        prev_block_hash: [u8; 32],
        state_root: [u8; 32],
        proposer: AccountId,
        transactions: Vec<SignedTransaction>,
        timestamp: u64,
    ) -> Self {
        let tx_root = Self::compute_tx_merkle_root(&transactions);
        let header = BlockHeader {
            height,
            round,
            prev_block_hash,
            state_root,
            tx_root,
            proposer,
            timestamp,
        };
        Self {
            header,
            transactions,
            commit: None,
        }
    }

    pub fn block_hash(&self) -> [u8; 32] {
        self.header.compute_hash()
    }

    pub fn compute_tx_merkle_root(txs: &[SignedTransaction]) -> [u8; 32] {
        if txs.is_empty() {
            return [0u8; 32];
        }
        let mut hasher = Sha256::new();
        for tx in txs {
            let tx_bytes = serde_json::to_vec(tx).unwrap_or_default();
            let mut h = Sha256::new();
            h.update(&tx_bytes);
            hasher.update(&h.finalize());
        }
        hasher.finalize().into()
    }
}

/// BFT Vote Phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoteType {
    Prevote,
    Precommit,
}

impl fmt::Display for VoteType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VoteType::Prevote => write!(f, "PREVOTE"),
            VoteType::Precommit => write!(f, "PRECOMMIT"),
        }
    }
}

/// Cryptographic BFT consensus vote signed by a validator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Vote {
    pub vote_type: VoteType,
    pub height: u64,
    pub round: u32,
    pub block_hash: Option<[u8; 32]>,
    pub validator: AccountId,
    pub signature: Vec<u8>,
}

impl Vote {
    pub fn sign_bytes(
        vote_type: VoteType,
        height: u64,
        round: u32,
        block_hash: Option<[u8; 32]>,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(match vote_type {
            VoteType::Prevote => 1,
            VoteType::Precommit => 2,
        });
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&round.to_be_bytes());
        if let Some(h) = block_hash {
            bytes.push(1);
            bytes.extend_from_slice(&h);
        } else {
            bytes.push(0);
        }
        bytes
    }
}

/// Validator with voting power in the active consensus set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsensusValidator {
    pub address: AccountId,
    pub voting_power: u64,
}
