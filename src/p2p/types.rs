use crate::blockchain::types::{AccountId, RoundPhase};
use crate::crypto::SignedTransaction;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

/// Unique Peer Identifier in the P2P overlay network.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PeerId(pub String);

impl PeerId {
    pub fn from_account(account: &AccountId) -> Self {
        Self(account.0.clone())
    }

    pub fn to_account(&self) -> AccountId {
        AccountId::new(self.0.clone())
    }
}

impl fmt::Display for PeerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// P2P Network protocol message framing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum P2pMessage {
    /// Initial handshake sent upon TCP connection.
    Handshake {
        protocol_version: String,
        peer_id: PeerId,
        listen_addr: Option<String>,
    },
    /// Handshake acknowledgment.
    HandshakeAck {
        protocol_version: String,
        peer_id: PeerId,
    },
    /// Gossip: Broadcast a new signed transaction across the network.
    BroadcastTx(SignedTransaction),
    /// Gossip: Announce available storage CID (Base Model or .safetensors adapter).
    AnnounceCid {
        cid: String,
        size_bytes: usize,
        provider: PeerId,
    },
    /// Gossip: Announce tournament round phase change.
    RoundPhaseChange {
        task_id: u64,
        round: usize,
        phase: RoundPhase,
    },
    /// Peer discovery: Exchange known active peer network addresses.
    PeerExchange(Vec<String>),
    /// Circuit Relay: Forward a routed message across NAT/CGNAT boundaries to a target peer.
    RelayForward {
        target_peer: PeerId,
        source_peer: PeerId,
        payload: Vec<u8>,
    },
    /// Circuit Relay: Direct payload delivered from a relay intermediary.
    RelayPayload {
        source_peer: PeerId,
        payload: Vec<u8>,
    },
    /// Keep-alive ping.
    Ping(u64),
    /// Keep-alive pong.
    Pong(u64),
}

impl P2pMessage {
    /// Compute a unique deterministic hash for this message to prevent duplicate gossip loops.
    pub fn message_id(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        if let Ok(bytes) = serde_json::to_vec(self) {
            hasher.update(&bytes);
        }
        hasher.finalize().into()
    }
}
