use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Simulated Content Addressed Storage (CAS) representing IPFS / Filecoin.
#[derive(Debug, Clone)]
pub struct IpfsStorage {
    storage: Arc<RwLock<HashMap<String, Vec<u8>>>>,
}

impl Default for IpfsStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl IpfsStorage {
    pub fn new() -> Self {
        Self {
            storage: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Compute CID (Content Identifier) from bytes: "bafy" + sha256 hex.
    pub fn compute_cid(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let hash = hasher.finalize();
        format!("bafy{}", hex::encode(hash))
    }

    /// Upload data to IPFS and return its CID.
    pub fn put(&self, data: &[u8]) -> String {
        let cid = Self::compute_cid(data);
        const MAX_STORE_OBJECTS: usize = 10_000;
        let mut store = self.storage.write().unwrap();
        if store.len() >= MAX_STORE_OBJECTS && !store.contains_key(&cid) {
            if let Some(first_key) = store.keys().next().cloned() {
                store.remove(&first_key);
            }
        }
        store.insert(cid.clone(), data.to_vec());
        cid
    }

    /// Retrieve data from IPFS by CID.
    pub fn get(&self, cid: &str) -> Option<Vec<u8>> {
        let store = self.storage.read().unwrap();
        store.get(cid).cloned()
    }

    /// Check if a CID exists in IPFS storage.
    pub fn contains(&self, cid: &str) -> bool {
        let store = self.storage.read().unwrap();
        store.contains_key(cid)
    }

    /// Get total number of pinned objects.
    pub fn count(&self) -> usize {
        let store = self.storage.read().unwrap();
        store.len()
    }
}
