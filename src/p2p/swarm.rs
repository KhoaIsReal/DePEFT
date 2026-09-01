use crate::crypto::SignedTransaction;
use crate::p2p::codec::{read_message, write_message};
use crate::p2p::types::{P2pMessage, PeerId};
use anyhow::{bail, Context, Result};
use std::collections::{HashMap, HashSet, VecDeque};
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;

const PROTOCOL_VERSION: &str = "depeft/1.0.0";
const MAX_SEEN_CACHE: usize = 10_000;
const MAX_CONNECTED_PEERS: usize = 64;

type MessageDeduplicationCache = (HashSet<[u8; 32]>, VecDeque<[u8; 32]>);

/// Network Swarm managing P2P overlay connections, peer discovery, and gossip routing.
#[derive(Clone)]
pub struct P2pSwarm {
    pub local_peer_id: PeerId,
    pub listen_addr: SocketAddr,
    connected_peers: Arc<RwLock<HashMap<PeerId, mpsc::UnboundedSender<P2pMessage>>>>,
    seen_messages: Arc<RwLock<MessageDeduplicationCache>>,
    known_addresses: Arc<RwLock<HashSet<String>>>,
    incoming_tx_sender: mpsc::UnboundedSender<SignedTransaction>,
    incoming_msg_sender: mpsc::UnboundedSender<(PeerId, P2pMessage)>,
}

impl P2pSwarm {
    pub fn new(
        local_peer_id: PeerId,
        listen_addr: SocketAddr,
    ) -> (
        Self,
        mpsc::UnboundedReceiver<SignedTransaction>,
        mpsc::UnboundedReceiver<(PeerId, P2pMessage)>,
    ) {
        let (tx_sender, tx_receiver) = mpsc::unbounded_channel();
        let (msg_sender, msg_receiver) = mpsc::unbounded_channel();

        let swarm = Self {
            local_peer_id,
            listen_addr,
            connected_peers: Arc::new(RwLock::new(HashMap::new())),
            seen_messages: Arc::new(RwLock::new((HashSet::new(), VecDeque::new()))),
            known_addresses: Arc::new(RwLock::new(HashSet::new())),
            incoming_tx_sender: tx_sender,
            incoming_msg_sender: msg_sender,
        };

        (swarm, tx_receiver, msg_receiver)
    }

    /// Check if a message has already been processed/gossiped (deduplication).
    pub fn is_seen(&self, msg_id: &[u8; 32]) -> bool {
        let (seen, _) = &*self.seen_messages.read().unwrap();
        seen.contains(msg_id)
    }

    /// Mark a message ID as seen with FIFO eviction to prevent cache wipe attack.
    pub fn mark_seen(&self, msg_id: [u8; 32]) {
        let (seen, queue) = &mut *self.seen_messages.write().unwrap();
        if seen.insert(msg_id) {
            queue.push_back(msg_id);
            if queue.len() > MAX_SEEN_CACHE {
                if let Some(oldest) = queue.pop_front() {
                    seen.remove(&oldest);
                }
            }
        }
    }

    /// Number of active connected peers.
    pub fn peer_count(&self) -> usize {
        let peers = self.connected_peers.read().unwrap();
        peers.len()
    }

    /// List of connected peer IDs.
    pub fn get_connected_peers(&self) -> Vec<PeerId> {
        let peers = self.connected_peers.read().unwrap();
        peers.keys().cloned().collect()
    }

    /// Broadcast a P2pMessage to all connected peers with gossip deduplication.
    pub fn broadcast(&self, msg: P2pMessage) {
        let msg_id = msg.message_id();
        self.mark_seen(msg_id);

        let peers = self.connected_peers.read().unwrap();
        for sender in peers.values() {
            let _ = sender.send(msg.clone());
        }
    }

    /// Broadcast a signed transaction across the P2P network.
    pub fn broadcast_transaction(&self, signed_tx: SignedTransaction) {
        self.broadcast(P2pMessage::BroadcastTx(signed_tx));
    }

    /// Start listening for incoming TCP connections.
    pub async fn start_listener(self: Arc<Self>) -> Result<()> {
        let listener = TcpListener::bind(self.listen_addr).await?;
        println!("[*] P2P Swarm listening on tcp://{}", self.listen_addr);

        let swarm = self.clone();
        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, remote_addr)) => {
                        let s = swarm.clone();
                        tokio::spawn(async move {
                            if let Err(e) = s.handle_incoming_connection(stream, remote_addr).await {
                                eprintln!("[!] Inbound P2P connection from {} error: {}", remote_addr, e);
                            }
                        });
                    }
                    Err(e) => {
                        eprintln!("[!] P2P accept error: {}", e);
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    /// Connect to a remote peer address (e.g. "127.0.0.1:9001").
    pub async fn connect_peer(&self, addr_str: &str) -> Result<PeerId> {
        let stream = TcpStream::connect(addr_str)
            .await
            .with_context(|| format!("Failed to connect to peer at {}", addr_str))?;

        self.handle_outbound_connection(stream, addr_str).await
    }

    /// Handle outbound connection handshake.
    async fn handle_outbound_connection(&self, mut stream: TcpStream, remote_addr: &str) -> Result<PeerId> {
        // 1. Send Handshake
        let handshake = P2pMessage::Handshake {
            protocol_version: PROTOCOL_VERSION.to_string(),
            peer_id: self.local_peer_id.clone(),
            listen_addr: Some(self.listen_addr.to_string()),
        };
        write_message(&mut stream, &handshake).await?;

        // 2. Read HandshakeAck or Handshake response
        let resp = read_message(&mut stream).await?;
        let remote_peer_id = match resp {
            P2pMessage::HandshakeAck { protocol_version, peer_id } => {
                if protocol_version != PROTOCOL_VERSION {
                    bail!("Protocol version mismatch: {} vs {}", protocol_version, PROTOCOL_VERSION);
                }
                peer_id
            }
            P2pMessage::Handshake { protocol_version, peer_id, .. } => {
                if protocol_version != PROTOCOL_VERSION {
                    bail!("Protocol version mismatch: {} vs {}", protocol_version, PROTOCOL_VERSION);
                }
                let ack = P2pMessage::HandshakeAck {
                    protocol_version: PROTOCOL_VERSION.to_string(),
                    peer_id: self.local_peer_id.clone(),
                };
                write_message(&mut stream, &ack).await?;
                peer_id
            }
            _ => bail!("Expected Handshake response from remote peer"),
        };

        // Record known address
        self.known_addresses.write().unwrap().insert(remote_addr.to_string());

        // Setup bidirectional message channels
        self.spawn_peer_handler(remote_peer_id.clone(), stream);
        Ok(remote_peer_id)
    }

    /// Handle inbound connection handshake.
    async fn handle_incoming_connection(&self, mut stream: TcpStream, remote_addr: SocketAddr) -> Result<()> {
        if self.peer_count() >= MAX_CONNECTED_PEERS {
            bail!("Max peer limit reached ({}/{})", self.peer_count(), MAX_CONNECTED_PEERS);
        }

        // 1. Read inbound Handshake
        let msg = read_message(&mut stream).await?;
        let (remote_peer_id, remote_listen) = match msg {
            P2pMessage::Handshake {
                protocol_version,
                peer_id,
                listen_addr,
            } => {
                if protocol_version != PROTOCOL_VERSION {
                    bail!("Protocol version mismatch: {} vs {}", protocol_version, PROTOCOL_VERSION);
                }
                (peer_id, listen_addr)
            }
            _ => bail!("First message must be Handshake"),
        };

        // 2. Send HandshakeAck
        let ack = P2pMessage::HandshakeAck {
            protocol_version: PROTOCOL_VERSION.to_string(),
            peer_id: self.local_peer_id.clone(),
        };
        write_message(&mut stream, &ack).await?;

        if let Some(l) = remote_listen {
            self.known_addresses.write().unwrap().insert(l);
        } else {
            self.known_addresses.write().unwrap().insert(remote_addr.to_string());
        }

        self.spawn_peer_handler(remote_peer_id, stream);
        Ok(())
    }

    /// Spawn peer reader and writer tasks.
    fn spawn_peer_handler(&self, peer_id: PeerId, stream: TcpStream) {
        let (mut reader, mut writer) = stream.into_split();
        let (outbound_tx, mut outbound_rx) = mpsc::unbounded_channel::<P2pMessage>();

        // Register in connected peers map
        self.connected_peers.write().unwrap().insert(peer_id.clone(), outbound_tx);

        // Writer task: sends queued outbound messages over TCP
        tokio::spawn(async move {
            while let Some(msg) = outbound_rx.recv().await {
                if let Err(_) = write_message(&mut writer, &msg).await {
                    break;
                }
            }
        });

        // Reader task: receives incoming messages from peer
        let swarm_clone = self.clone();
        let pid_clone = peer_id.clone();

        tokio::spawn(async move {
            loop {
                match read_message(&mut reader).await {
                    Ok(msg) => {
                        let msg_id = msg.message_id();
                        if swarm_clone.is_seen(&msg_id) {
                            continue; // Deduplicate already-seen messages
                        }
                        swarm_clone.mark_seen(msg_id);

                        // Handle incoming gossip messages
                        match &msg {
                            P2pMessage::BroadcastTx(signed_tx) => {
                                let _ = swarm_clone.incoming_tx_sender.send(signed_tx.clone());
                                // Re-gossip to other peers (except sender)
                                swarm_clone.regossip_except(&pid_clone, msg.clone());
                            }
                            P2pMessage::Ping(nonce) => {
                                let pong = P2pMessage::Pong(*nonce);
                                if let Some(sender) = swarm_clone.connected_peers.read().unwrap().get(&pid_clone) {
                                    let _ = sender.send(pong);
                                }
                            }
                            _ => {
                                let _ = swarm_clone.incoming_msg_sender.send((pid_clone.clone(), msg.clone()));
                                swarm_clone.regossip_except(&pid_clone, msg);
                            }
                        }
                    }
                    Err(_) => {
                        // Connection closed or broken
                        break;
                    }
                }
            }

            // Unregister disconnected peer
            swarm_clone.connected_peers.write().unwrap().remove(&pid_clone);
        });
    }

    /// Re-gossip a message to all connected peers except the originating peer.
    fn regossip_except(&self, exclude_peer: &PeerId, msg: P2pMessage) {
        let peers = self.connected_peers.read().unwrap();
        for (pid, sender) in peers.iter() {
            if pid != exclude_peer {
                let _ = sender.send(msg.clone());
            }
        }
    }
}
