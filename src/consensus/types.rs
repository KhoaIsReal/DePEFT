use crate::blockchain::types::AccountId;
use crate::crypto::SignedTransaction;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

/// Block Header containing state root, tx root, and consensus metadata.
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
        hasher.update(self.proposer.as_str().as_bytes());
        hasher.update(&self.timestamp.to_be_bytes());
        hasher.finalize().into()
    }
}

/// Finalized Block Commit with gathered 2/3+ Precommit cryptographic signatures.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockCommit {
    pub height: u64,
    pub round: u32,
    pub block_hash: [u8; 32],
    pub signatures: Vec<(AccountId, Vec<u8>)>,
}

/// Full Candidate Block proposed by the consensus leader.
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

    /// Cryptographically verify the Ed25519 signature of the vote.
    pub fn verify_signature(&self) -> anyhow::Result<()> {
        let addr_hex = self.validator.as_str().trim_start_matches("0x");
        let pubkey_bytes = hex::decode(addr_hex)
            .map_err(|e| anyhow::anyhow!("Invalid validator account hex: {}", e))?;

        if pubkey_bytes.len() != 32 {
            anyhow::bail!("Validator account id is not a 32-byte Ed25519 public key");
        }

        let mut key_arr = [0u8; 32];
        key_arr.copy_from_slice(&pubkey_bytes);

        let verifying_key = VerifyingKey::from_bytes(&key_arr)
            .map_err(|e| anyhow::anyhow!("Invalid verifying key bytes: {}", e))?;

        if self.signature.len() != 64 {
            anyhow::bail!("Invalid signature length for vote: expected 64 bytes");
        }

        let sig_bytes: [u8; 64] = self.signature.as_slice().try_into()?;
        let signature = Signature::from_bytes(&sig_bytes);

        let sign_bytes = Self::sign_bytes(self.vote_type, self.height, self.round, self.block_hash);

        verifying_key
            .verify(&sign_bytes, &signature)
            .map_err(|e| anyhow::anyhow!("Cryptographic signature verification failed for vote from {}: {}", self.validator, e))?;

        Ok(())
    }
}

/// Validator with voting power in the active consensus set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsensusValidator {
    pub address: AccountId,
    pub voting_power: u64,
}
