use crate::blockchain::transactions::Transaction;
use crate::blockchain::types::AccountId;
use anyhow::{Result, bail};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};

/// Cryptographic Ed25519 Keypair for DePEFT accounts.
#[derive(Clone)]
pub struct AccountKeypair {
    pub signing_key: SigningKey,
    pub verifying_key: VerifyingKey,
}

impl AccountKeypair {
    /// Generate a fresh random keypair using OS cryptographic randomness.
    pub fn generate() -> Self {
        let mut csprng = OsRng;
        let signing_key = SigningKey::generate(&mut csprng);
        let verifying_key = signing_key.verifying_key();
        Self {
            signing_key,
            verifying_key,
        }
    }

    /// Reconstruct keypair from 32-byte secret seed.
    pub fn from_secret_bytes(bytes: &[u8; 32]) -> Self {
        let signing_key = SigningKey::from_bytes(bytes);
        let verifying_key = signing_key.verifying_key();
        Self {
            signing_key,
            verifying_key,
        }
    }

    /// Get the public AccountId (`0x...` hex).
    pub fn account_id(&self) -> AccountId {
        AccountId::new(format!("0x{}", hex::encode(self.verifying_key.to_bytes())))
    }

    /// Export public key 32 bytes.
    pub fn public_key_bytes(&self) -> [u8; 32] {
        self.verifying_key.to_bytes()
    }

    /// Export secret key 32 bytes.
    pub fn secret_key_bytes(&self) -> [u8; 32] {
        self.signing_key.to_bytes()
    }

    /// Sign an arbitrary message.
    pub fn sign_message(&self, message: &[u8]) -> Vec<u8> {
        let signature = self.signing_key.sign(message);
        signature.to_bytes().to_vec()
    }

    /// Sign an on-chain transaction.
    pub fn sign_transaction(&self, tx: Transaction) -> Result<SignedTransaction> {
        let tx_bytes = serde_json::to_vec(&tx)?;
        let signature_bytes = self.sign_message(&tx_bytes);
        Ok(SignedTransaction {
            tx,
            sender_public_key: self.public_key_bytes(),
            signature: signature_bytes,
        })
    }
}

/// A cryptographically signed on-chain transaction envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SignedTransaction {
    pub tx: Transaction,
    pub sender_public_key: [u8; 32],
    pub signature: Vec<u8>,
}

impl SignedTransaction {
    /// Verify cryptographic signature against transaction payload.
    pub fn verify_signature(&self) -> Result<AccountId> {
        let verifying_key = VerifyingKey::from_bytes(&self.sender_public_key)
            .map_err(|e| anyhow::anyhow!("Invalid verifying key: {}", e))?;

        if self.signature.len() != 64 {
            bail!("Invalid signature length: expected 64 bytes");
        }

        let sig_bytes: [u8; 64] = self.signature.as_slice().try_into()?;
        let signature = Signature::from_bytes(&sig_bytes);

        let tx_bytes = serde_json::to_vec(&self.tx)?;
        verifying_key
            .verify(&tx_bytes, &signature)
            .map_err(|e| anyhow::anyhow!("Cryptographic signature verification failed: {}", e))?;

        let sender_account = AccountId::new(format!("0x{}", hex::encode(self.sender_public_key)));
        anyhow::ensure!(
            self.tx.sender() == &sender_account,
            "Transaction inner sender {} does not match signer public key account {}",
            self.tx.sender(),
            sender_account
        );

        Ok(sender_account)
    }
}
