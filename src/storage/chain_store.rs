use crate::blockchain::state::AppChainState;
use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, OptionalExtension, params};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Durable, crash-safe storage for the canonical application state.
/// SQLite WAL makes each replacement snapshot atomic; block execution will use
/// this same transaction boundary in the subsequent BFT milestone.
pub struct ChainStore {
    conn: Connection,
}

impl ChainStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path).context("opening chain SQLite database")?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=FULL;
             CREATE TABLE IF NOT EXISTS chain_state (
                 id INTEGER PRIMARY KEY CHECK (id = 1),
                 state_json BLOB NOT NULL,
                 state_hash BLOB NOT NULL
             );",
        )?;
        Ok(Self { conn })
    }

    /// Serialize through `serde_json::Value` so object keys are canonicalized by
    /// serde_json's ordered map implementation before hashing/persisting.
    pub fn canonical_bytes(state: &AppChainState) -> Result<Vec<u8>> {
        let value = serde_json::to_value(state)?;
        Ok(serde_json::to_vec(&value)?)
    }

    pub fn state_hash(state: &AppChainState) -> Result<[u8; 32]> {
        Ok(Sha256::digest(Self::canonical_bytes(state)?).into())
    }

    pub fn load(&self) -> Result<Option<AppChainState>> {
        let row: Option<(Vec<u8>, Vec<u8>)> = self
            .conn
            .query_row(
                "SELECT state_json, state_hash FROM chain_state WHERE id = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        let Some((bytes, stored_hash)) = row else {
            return Ok(None);
        };
        let state: AppChainState =
            serde_json::from_slice(&bytes).context("decoding persisted chain state")?;
        let computed = Self::state_hash(&state)?;
        ensure!(
            stored_hash.as_slice() == computed,
            "persisted chain state hash mismatch"
        );
        Ok(Some(state))
    }

    pub fn save(&mut self, state: &AppChainState) -> Result<[u8; 32]> {
        let bytes = Self::canonical_bytes(state)?;
        let hash: [u8; 32] = Sha256::digest(&bytes).into();
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO chain_state (id, state_json, state_hash) VALUES (1, ?1, ?2)
             ON CONFLICT(id) DO UPDATE SET state_json = excluded.state_json, state_hash = excluded.state_hash",
            params![bytes, hash.to_vec()],
        )?;
        tx.commit()?;
        Ok(hash)
    }
}
