use anyhow::Result;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

/// Persistent Disk-backed Content Addressable Storage (CAS) for IPFS artifacts.
#[derive(Debug, Clone)]
pub struct DiskIpfsStorage {
    root_dir: PathBuf,
}

impl DiskIpfsStorage {
    /// Initialize disk storage in the specified directory (creates it if missing).
    pub fn new(root_dir: impl AsRef<Path>) -> Result<Self> {
        let path = root_dir.as_ref().to_path_buf();
        fs::create_dir_all(&path)?;
        Ok(Self { root_dir: path })
    }

    /// Default storage directory under user's home: `~/.depeft/storage`.
    pub fn default_location() -> Result<Self> {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let default_path = home.join(".depeft").join("storage");
        Self::new(default_path)
    }

    /// Compute CID (Content Identifier) from bytes.
    pub fn compute_cid(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let hash = hasher.finalize();
        format!("bafy{}", hex::encode(hash))
    }

    /// Store binary data on disk and return its CID.
    pub fn put(&self, data: &[u8]) -> Result<String> {
        let cid = Self::compute_cid(data);
        let file_path = self.root_dir.join(&cid);
        fs::write(file_path, data)?;
        Ok(cid)
    }

    /// Retrieve binary data by CID.
    pub fn get(&self, cid: &str) -> Option<Vec<u8>> {
        let file_path = self.root_dir.join(cid);
        fs::read(file_path).ok()
    }

    /// Check if a CID exists on disk.
    pub fn contains(&self, cid: &str) -> bool {
        let file_path = self.root_dir.join(cid);
        file_path.exists()
    }

    /// Count total stored objects.
    pub fn count(&self) -> usize {
        if let Ok(entries) = fs::read_dir(&self.root_dir) {
            entries.filter_map(Result::ok).count()
        } else {
            0
        }
    }

    /// Get root directory path.
    pub fn path(&self) -> &Path {
        &self.root_dir
    }
}
