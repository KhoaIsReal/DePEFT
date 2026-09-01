use crate::blockchain::state::AppChainState;
use crate::blockchain::types::{AccountId, TaskSpec};
use crate::crypto::SignedTransaction;
use crate::storage::disk_ipfs::DiskIpfsStorage;
use crate::storage::vector_db::EmbeddedVectorDb;
use anyhow::Result;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};
use crate::p2p::P2pSwarm;
use tower_http::cors::CorsLayer;

/// Shared runtime state of the DePEFT App-Chain Node.
#[derive(Clone)]
pub struct NodeContext {
    pub chain: Arc<RwLock<AppChainState>>,
    pub storage: Arc<DiskIpfsStorage>,
    pub vector_db: Arc<EmbeddedVectorDb>,
    pub swarm: Option<Arc<P2pSwarm>>,
}

#[derive(Serialize)]
struct NodeStatusResponse {
    block_height: u32,
    tasks_count: usize,
    storage_objects_count: usize,
    connected_peers_count: usize,
    version: &'static str,
}

#[derive(Serialize)]
struct GenericResponse {
    status: &'static str,
    message: String,
}

#[derive(Deserialize)]
struct ConnectPeerRequest {
    addr: String,
}

#[derive(Serialize)]
struct PeerListResponse {
    connected_peers: Vec<String>,
}

#[derive(Serialize)]
struct StorageUploadResponse {
    cid: String,
    size_bytes: usize,
}

#[derive(Deserialize)]
struct StorageUploadRequest {
    /// Base64 or Hex encoded data bytes
    data_hex: String,
}

#[derive(Serialize)]
struct BalanceResponse {
    account: String,
    balance: u128,
}

/// HTTP API Handlers
async fn get_status(State(ctx): State<NodeContext>) -> Json<NodeStatusResponse> {
    let chain = ctx.chain.read().unwrap();
    let peers_count = ctx.swarm.as_ref().map(|s| s.peer_count()).unwrap_or(0);
    Json(NodeStatusResponse {
        block_height: chain.block_height,
        tasks_count: chain.tasks.len(),
        storage_objects_count: ctx.storage.count(),
        connected_peers_count: peers_count,
        version: "0.1.0-depeft",
    })
}

async fn get_tasks(State(ctx): State<NodeContext>) -> Json<Vec<TaskSpec>> {
    let chain = ctx.chain.read().unwrap();
    let tasks: Vec<TaskSpec> = chain.tasks.values().cloned().collect();
    Json(tasks)
}

async fn get_task_by_id(
    Path(id): Path<u64>,
    State(ctx): State<NodeContext>,
) -> Result<Json<TaskSpec>, (StatusCode, Json<GenericResponse>)> {
    let chain = ctx.chain.read().unwrap();
    if let Some(task) = chain.tasks.get(&id) {
        Ok(Json(task.clone()))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(GenericResponse {
                status: "error",
                message: format!("Task ID {} not found", id),
            }),
        ))
    }
}

async fn submit_signed_tx(
    State(ctx): State<NodeContext>,
    Json(signed_tx): Json<SignedTransaction>,
) -> Result<Json<GenericResponse>, (StatusCode, Json<GenericResponse>)> {
    // 1. Cryptographically verify Ed25519 signature
    let sender = match signed_tx.verify_signature() {
        Ok(acc) => acc,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(GenericResponse {
                    status: "signature_error",
                    message: format!("Signature verification failed: {}", e),
                }),
            ));
        }
    };

    // 2. Deterministically apply transaction to AppChainState
    let mut chain = ctx.chain.write().unwrap();
    match chain.apply_transaction(signed_tx.tx.clone()) {
        Ok(_) => {
            // 3. Broadcast to P2P network overlay
            if let Some(swarm) = &ctx.swarm {
                swarm.broadcast_transaction(signed_tx);
            }

            Ok(Json(GenericResponse {
                status: "ok",
                message: format!("Transaction successfully committed by {}", sender),
            }))
        }
        Err(e) => Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(GenericResponse {
                status: "state_error",
                message: format!("Transaction execution rejected: {}", e),
            }),
        )),
    }
}

async fn get_connected_peers(State(ctx): State<NodeContext>) -> Json<PeerListResponse> {
    let peers = ctx
        .swarm
        .as_ref()
        .map(|s| s.get_connected_peers().into_iter().map(|p| p.0).collect())
        .unwrap_or_default();
    Json(PeerListResponse {
        connected_peers: peers,
    })
}

async fn connect_to_p2p_peer(
    State(ctx): State<NodeContext>,
    Json(payload): Json<ConnectPeerRequest>,
) -> Result<Json<GenericResponse>, (StatusCode, Json<GenericResponse>)> {
    if let Some(swarm) = &ctx.swarm {
        match swarm.connect_peer(&payload.addr).await {
            Ok(peer_id) => Ok(Json(GenericResponse {
                status: "ok",
                message: format!("Successfully connected to peer {}", peer_id),
            })),
            Err(e) => Err((
                StatusCode::BAD_GATEWAY,
                Json(GenericResponse {
                    status: "p2p_error",
                    message: format!("Failed to connect to peer {}: {}", payload.addr, e),
                }),
            )),
        }
    } else {
        Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(GenericResponse {
                status: "error",
                message: "P2P swarm is not enabled on this node".to_string(),
            }),
        ))
    }
}

async fn upload_storage(
    State(ctx): State<NodeContext>,
    Json(payload): Json<StorageUploadRequest>,
) -> Result<Json<StorageUploadResponse>, (StatusCode, Json<GenericResponse>)> {
    let data = match hex::decode(&payload.data_hex) {
        Ok(d) => d,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(GenericResponse {
                    status: "error",
                    message: format!("Invalid hex data: {}", e),
                }),
            ));
        }
    };

    let size_bytes = data.len();
    match ctx.storage.put(&data) {
        Ok(cid) => Ok(Json(StorageUploadResponse { cid, size_bytes })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(GenericResponse {
                status: "storage_error",
                message: format!("Failed to persist artifact: {}", e),
            }),
        )),
    }
}

async fn download_storage(
    Path(cid): Path<String>,
    State(ctx): State<NodeContext>,
) -> Result<Response, (StatusCode, Json<GenericResponse>)> {
    if let Some(bytes) = ctx.storage.get(&cid) {
        Ok(bytes.into_response())
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(GenericResponse {
                status: "not_found",
                message: format!("CID {} not found in CAS storage", cid),
            }),
        ))
    }
}

async fn get_account_balance(
    Path(account_str): Path<String>,
    State(ctx): State<NodeContext>,
) -> Json<BalanceResponse> {
    let chain = ctx.chain.read().unwrap();
    let account = AccountId::new(account_str.clone());
    let balance = chain.balance_of(&account);
    Json(BalanceResponse {
        account: account_str,
        balance,
    })
}

#[derive(Deserialize)]
struct FaucetRequest {
    account: String,
    amount: Option<u128>,
}

async fn request_faucet(
    State(ctx): State<NodeContext>,
    Json(payload): Json<FaucetRequest>,
) -> Result<Json<GenericResponse>, (StatusCode, Json<GenericResponse>)> {
    let account = AccountId::new(payload.account);
    let amount = payload.amount.unwrap_or(10_000).min(100_000); // default 10k, max 100k per request

    let mut chain = ctx.chain.write().unwrap();
    chain.mint(account.clone(), amount);

    Ok(Json(GenericResponse {
        status: "ok",
        message: format!("Successfully minted {} tokens to {}", amount, account),
    }))
}

async fn get_dashboard() -> axum::response::Html<&'static str> {
    axum::response::Html(crate::node::dashboard::DASHBOARD_HTML)
}

/// Create the Axum Router for the Node API.
pub fn create_app(ctx: NodeContext) -> Router {
    Router::new()
        .route("/", get(get_dashboard))
        .route("/dashboard", get(get_dashboard))
        .route("/api/v1/status", get(get_status))
        .route("/api/v1/tasks", get(get_tasks))
        .route("/api/v1/tasks/:id", get(get_task_by_id))
        .route("/api/v1/tx", post(submit_signed_tx))
        .route("/api/v1/p2p/peers", get(get_connected_peers))
        .route("/api/v1/p2p/connect", post(connect_to_p2p_peer))
        .route("/api/v1/storage", post(upload_storage))
        .route("/api/v1/storage/:cid", get(download_storage))
        .route("/api/v1/accounts/:account/balance", get(get_account_balance))
        .route("/api/v1/faucet", post(request_faucet))
        .layer(CorsLayer::permissive())
        .with_state(ctx)
}

/// Start the real DePEFT App-Chain Node HTTP/JSON-RPC Server.
pub async fn start_node_server(
    chain: Arc<RwLock<AppChainState>>,
    storage: Arc<DiskIpfsStorage>,
    vector_db: Arc<EmbeddedVectorDb>,
    swarm: Option<Arc<P2pSwarm>>,
    addr: SocketAddr,
) -> Result<()> {
    let ctx = NodeContext {
        chain,
        storage,
        vector_db,
        swarm,
    };

    let app = create_app(ctx);

    println!("[*] DePEFT App-Chain Node Server listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
