use anyhow::{bail, Context, Result};
use reqwest::multipart::{Form, Part};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Metadata returned by an IPFS Kubo daemon on file addition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpfsAddResponse {
    #[serde(rename = "Name")]
    pub name: String,
    #[serde(rename = "Hash")]
    pub hash: String, // CID string (e.g. Qm... or bafy...)
    #[serde(rename = "Size")]
    pub size: String,
}

/// Node identity and network information from IPFS Kubo `/api/v0/id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpfsNodeInfo {
    #[serde(rename = "ID")]
    pub id: String,
    #[serde(rename = "Addresses", default)]
    pub addresses: Vec<String>,
    #[serde(rename = "AgentVersion", default)]
    pub agent_version: String,
    #[serde(rename = "ProtocolVersion", default)]
    pub protocol_version: String,
}

/// Client for live IPFS Kubo RPC API (`/api/v0/`).
#[derive(Debug, Clone)]
pub struct IpfsKuboClient {
    api_url: String,
    gateway_url: String,
    http: Client,
}

impl Default for IpfsKuboClient {
    fn default() -> Self {
        Self::new("http://127.0.0.1:5001", "http://127.0.0.1:8080")
    }
}

impl IpfsKuboClient {
    /// Initialize with custom RPC API and gateway URLs.
    pub fn new(api_url: impl Into<String>, gateway_url: impl Into<String>) -> Self {
        let http = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_default();

        Self {
            api_url: api_url.into().trim_end_matches('/').to_string(),
            gateway_url: gateway_url.into().trim_end_matches('/').to_string(),
            http,
        }
    }

    pub fn api_url(&self) -> &str {
        &self.api_url
    }

    pub fn gateway_url(&self) -> &str {
        &self.gateway_url
    }

    /// Check if the local or remote IPFS Kubo daemon is reachable.
    pub async fn is_online(&self) -> bool {
        self.node_info().await.is_ok()
    }

    /// Query the IPFS node identity and connected addresses via `/api/v0/id`.
    pub async fn node_info(&self) -> Result<IpfsNodeInfo> {
        let url = format!("{}/api/v0/id", self.api_url);
        let resp = self
            .http
            .post(&url)
            .send()
            .await
            .context("Failed to connect to IPFS Kubo RPC daemon")?;

        if !resp.status().is_success() {
            bail!("IPFS id request returned status: {}", resp.status());
        }

        let info = resp.json::<IpfsNodeInfo>().await?;
        Ok(info)
    }

    /// Upload binary bytes (e.g. .safetensors weights, datasets) to IPFS via `/api/v0/add`.
    pub async fn add_bytes(&self, data: &[u8], filename: &str) -> Result<String> {
        let url = format!("{}/api/v0/add?pin=true", self.api_url);

        let part = Part::bytes(data.to_vec()).file_name(filename.to_string());
        let form = Form::new().part("file", part);

        let resp = self
            .http
            .post(&url)
            .multipart(form)
            .send()
            .await
            .context("Failed to send /api/v0/add request to IPFS daemon")?;

        if !resp.status().is_success() {
            bail!("IPFS /api/v0/add failed with status: {}", resp.status());
        }

        let add_resp = resp.json::<IpfsAddResponse>().await?;
        Ok(add_resp.hash)
    }

    /// Validate that a CID string is safe and conforms to standard alphanumeric multihash format.
    fn validate_cid(cid: &str) -> Result<()> {
        if cid.is_empty() || cid.len() > 128 || !cid.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
            bail!("Invalid or unsafe IPFS CID format: '{}'", cid);
        }
        Ok(())
    }

    /// Download binary bytes from IPFS by CID via `/api/v0/cat`.
    pub async fn cat_bytes(&self, cid: &str) -> Result<Vec<u8>> {
        Self::validate_cid(cid)?;
        let url = format!("{}/api/v0/cat?arg={}", self.api_url, cid);
        let resp = self
            .http
            .post(&url)
            .send()
            .await
            .context(format!("Failed to retrieve CID {} from IPFS daemon", cid))?;

        if !resp.status().is_success() {
            bail!("IPFS /api/v0/cat failed for CID {}: status {}", cid, resp.status());
        }

        let bytes = resp.bytes().await?.to_vec();
        Ok(bytes)
    }

    /// Pin a CID on the IPFS daemon so it is protected against garbage collection.
    pub async fn pin_add(&self, cid: &str) -> Result<()> {
        Self::validate_cid(cid)?;
        let url = format!("{}/api/v0/pin/add?arg={}", self.api_url, cid);
        let resp = self
            .http
            .post(&url)
            .send()
            .await
            .context(format!("Failed to pin CID {} on IPFS", cid))?;

        if !resp.status().is_success() {
            bail!("IPFS /api/v0/pin/add failed for CID {}: status {}", cid, resp.status());
        }
        Ok(())
    }
}
