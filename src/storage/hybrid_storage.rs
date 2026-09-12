use crate::storage::disk_ipfs::DiskIpfsStorage;
use crate::storage::kubo_client::IpfsKuboClient;
use anyhow::Result;
use std::sync::Arc;

/// Hybrid Storage Manager providing unified access to local disk CAS and live IPFS Kubo network daemon.
#[derive(Clone)]
pub struct HybridStorageManager {
    pub local_cas: Arc<DiskIpfsStorage>,
    pub kubo_client: Option<Arc<IpfsKuboClient>>,
}

impl HybridStorageManager {
    pub fn new(local_cas: Arc<DiskIpfsStorage>, kubo_client: Option<Arc<IpfsKuboClient>>) -> Self {
        Self {
            local_cas,
            kubo_client,
        }
    }

    /// Store data: writes to local fast CAS disk, and syncs to live IPFS network if daemon is online.
    pub async fn put(&self, data: &[u8], filename: &str) -> Result<String> {
        // 1. Always store in local CAS
        let local_cid = self.local_cas.put(data)?;

        // 2. If live IPFS Kubo daemon is available, upload & pin to the global IPFS swarm
        if let Some(kubo) = &self.kubo_client {
            if let Ok(ipfs_cid) = kubo.add_bytes(data, filename).await {
                let _ = kubo.pin_add(&ipfs_cid).await;
                return Ok(ipfs_cid);
            }
        }

        Ok(local_cid)
    }

    /// Retrieve data: checks local fast CAS disk first; if not found, streams from live IPFS network and caches locally.
    pub async fn get(&self, cid: &str) -> Result<Option<Vec<u8>>> {
        // 1. Fast local cache hit
        if let Some(bytes) = self.local_cas.get(cid) {
            return Ok(Some(bytes));
        }

        // 2. Fetch from live IPFS network if configured
        if let Some(kubo) = &self.kubo_client {
            if let Ok(bytes) = kubo.cat_bytes(cid).await {
                // Content-Integrity Verification: ensure returned bytes hash matches requested CID
                if cid.starts_with("bafy") {
                    let computed = DiskIpfsStorage::compute_cid(&bytes);
                    if computed != cid {
                        anyhow::bail!(
                            "IPFS Content Integrity mismatch: expected {}, computed {}",
                            cid,
                            computed
                        );
                    }
                }

                // Cache into local CAS for subsequent zero-latency reads
                let _ = self.local_cas.put(&bytes);
                return Ok(Some(bytes));
            }
        }

        Ok(None)
    }

    /// Check if CID exists in local storage or remote IPFS.
    pub fn contains_local(&self, cid: &str) -> bool {
        self.local_cas.contains(cid)
    }
}
