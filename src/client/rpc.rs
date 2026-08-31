use crate::blockchain::types::TaskSpec;
use crate::crypto::SignedTransaction;
use anyhow::{bail, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct DePeftClient {
    base_url: String,
    http: Client,
}

#[derive(Serialize)]
struct StorageUploadRequest {
    data_hex: String,
}

#[derive(Deserialize)]
struct StorageUploadResponse {
    cid: String,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct GenericResponse {
    status: String,
    message: String,
}

#[derive(Deserialize)]
pub struct NodeStatus {
    pub block_height: u32,
    pub tasks_count: usize,
    pub storage_objects_count: usize,
    pub version: String,
}

#[derive(Deserialize)]
pub struct BalanceInfo {
    pub account: String,
    pub balance: u128,
}

impl DePeftClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            http: Client::new(),
        }
    }

    /// Query node status.
    pub async fn get_status(&self) -> Result<NodeStatus> {
        let url = format!("{}/api/v1/status", self.base_url);
        let resp = self.http.get(&url).send().await?.json::<NodeStatus>().await?;
        Ok(resp)
    }

    /// Query all active tasks.
    pub async fn get_tasks(&self) -> Result<Vec<TaskSpec>> {
        let url = format!("{}/api/v1/tasks", self.base_url);
        let resp = self.http.get(&url).send().await?.json::<Vec<TaskSpec>>().await?;
        Ok(resp)
    }

    /// Query specific task by ID.
    pub async fn get_task(&self, task_id: u64) -> Result<TaskSpec> {
        let url = format!("{}/api/v1/tasks/{}", self.base_url, task_id);
        let resp = self.http.get(&url).send().await?.json::<TaskSpec>().await?;
        Ok(resp)
    }

    /// Submit a signed transaction.
    pub async fn submit_transaction(&self, signed_tx: &SignedTransaction) -> Result<String> {
        let url = format!("{}/api/v1/tx", self.base_url);
        let resp = self.http.post(&url).json(signed_tx).send().await?;

        if !resp.status().is_success() {
            let err_resp: GenericResponse = resp.json().await.unwrap_or(GenericResponse {
                status: "error".to_string(),
                message: "Unknown server error".to_string(),
            });
            bail!("Transaction failed: {}", err_resp.message);
        }

        let ok_resp: GenericResponse = resp.json().await?;
        Ok(ok_resp.message)
    }

    /// Upload binary artifact to CAS storage via HTTP.
    pub async fn upload_storage(&self, data: &[u8]) -> Result<String> {
        let url = format!("{}/api/v1/storage", self.base_url);
        let req = StorageUploadRequest {
            data_hex: hex::encode(data),
        };
        let resp = self.http.post(&url).json(&req).send().await?;

        if !resp.status().is_success() {
            bail!("Storage upload failed with status: {}", resp.status());
        }

        let res: StorageUploadResponse = resp.json().await?;
        Ok(res.cid)
    }

    /// Download binary artifact from CAS storage via HTTP.
    pub async fn download_storage(&self, cid: &str) -> Result<Vec<u8>> {
        let url = format!("{}/api/v1/storage/{}", self.base_url, cid);
        let resp = self.http.get(&url).send().await?;

        if !resp.status().is_success() {
            bail!("Storage download failed for CID {}: status {}", cid, resp.status());
        }

        let bytes = resp.bytes().await?.to_vec();
        Ok(bytes)
    }

    /// Query account token balance.
    pub async fn get_balance(&self, account_hex: &str) -> Result<u128> {
        let url = format!("{}/api/v1/accounts/{}/balance", self.base_url, account_hex);
        let info = self.http.get(&url).send().await?.json::<BalanceInfo>().await?;
        Ok(info.balance)
    }

    /// Query connected P2P peers.
    pub async fn get_peers(&self) -> Result<Vec<String>> {
        #[derive(Deserialize)]
        struct PeerListResp {
            connected_peers: Vec<String>,
        }
        let url = format!("{}/api/v1/p2p/peers", self.base_url);
        let resp = self.http.get(&url).send().await?.json::<PeerListResp>().await?;
        Ok(resp.connected_peers)
    }

    /// Instruct node to connect to a remote P2P peer.
    pub async fn connect_peer(&self, peer_addr: &str) -> Result<String> {
        #[derive(Serialize)]
        struct ConnReq {
            addr: String,
        }
        let url = format!("{}/api/v1/p2p/connect", self.base_url);
        let req = ConnReq {
            addr: peer_addr.to_string(),
        };
        let resp = self.http.post(&url).json(&req).send().await?;

        if !resp.status().is_success() {
            bail!("P2P connect failed with status: {}", resp.status());
        }

        let res: GenericResponse = resp.json().await?;
        Ok(res.message)
    }
}
