use crate::blockchain::types::AccountId;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256, Sha512};
use std::fmt;

/// Supported Trusted Execution Environment (TEE) hardware platforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TeeType {
    IntelSgxDcap = 0,
    IntelTdx = 1,
    AmdSevSnp = 2,
    AwsNitroEnclave = 3,
}

impl fmt::Display for TeeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TeeType::IntelSgxDcap => write!(f, "Intel SGX (DCAP)"),
            TeeType::IntelTdx => write!(f, "Intel TDX"),
            TeeType::AmdSevSnp => write!(f, "AMD SEV-SNP"),
            TeeType::AwsNitroEnclave => write!(f, "AWS Nitro Enclave"),
        }
    }
}

impl std::str::FromStr for TeeType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().replace('-', "_").as_str() {
            "sgx" | "intelsgx" | "intelsgxdcap" | "intel_sgx" | "intel_sgx_dcap" => {
                Ok(TeeType::IntelSgxDcap)
            }
            "tdx" | "inteltdx" | "intel_tdx" => Ok(TeeType::IntelTdx),
            "sev" | "amdsev" | "amdsevsnp" | "amd_sev" | "amd_sev_snp" => {
                Ok(TeeType::AmdSevSnp)
            }
            "nitro" | "awsnitro" | "awsnitroenclave" | "aws_nitro" | "aws_nitro_enclave" => {
                Ok(TeeType::AwsNitroEnclave)
            }
            _ => Err(format!(
                "Unknown TEE type: '{}'. Supported types: sgx, tdx, sev, nitro",
                s
            )),
        }
    }
}

/// Hardware enclave code and author measurements (e.g. MRENCLAVE / MRSIGNER, or MRTD / RTMR for Intel TDX).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnclaveMeasurement {
    /// SHA-256 hash of the enclave memory layout and code at initialization (MRTD for TDX)
    pub mrenclave: [u8; 32],
    /// SHA-256 hash of the enclave author release signing key (RTMR0 for TDX)
    pub mrsigner: [u8; 32],
    pub isv_prod_id: u16,
    pub isv_svn: u16,
}

impl EnclaveMeasurement {
    pub fn mrenclave_hex(&self) -> String {
        format!("0x{}", hex::encode(self.mrenclave))
    }

    pub fn mrsigner_hex(&self) -> String {
        format!("0x{}", hex::encode(self.mrsigner))
    }

    /// For Intel TDX: Alias for MRTD (Measurement of Initial Trust Domain).
    pub fn mrtd_hex(&self) -> String {
        self.mrenclave_hex()
    }

    /// For Intel TDX: Alias for RTMR0 (Runtime Measurement Register 0).
    pub fn rtmr0_hex(&self) -> String {
        self.mrsigner_hex()
    }
}

/// Hardware TEE Security Flags and Attestation Mode.
/// Differentiates genuine production hardware from testnet software simulations,
/// and detects insecure debug-mode enclaves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeeSecurityFlags {
    /// True if quote was produced by a software simulator (valid on testnet with --testnet-tee-sim only)
    pub is_simulation: bool,
    /// Debug mode: if true, memory was inspectable via debugger (insecure for mainnet production)
    pub debug_mode: bool,
    /// Hardware assurance level (0 = pure software mock, 1 = simulated, 2 = genuine production hardware)
    pub hardware_level: u8,
}

impl Default for TeeSecurityFlags {
    fn default() -> Self {
        Self {
            is_simulation: false,
            debug_mode: false,
            hardware_level: 2,
        }
    }
}

impl TeeSecurityFlags {
    /// Security flags for testnet software simulation mode
    pub fn simulation() -> Self {
        Self {
            is_simulation: true,
            debug_mode: true,
            hardware_level: 0,
        }
    }

    /// Security flags for genuine, production hardware TEE execution
    pub fn genuine_production() -> Self {
        Self {
            is_simulation: false,
            debug_mode: false,
            hardware_level: 2,
        }
    }
}

/// Cryptographic Hardware Remote Attestation Quote.
/// Proves to the App-Chain that private evaluation executed inside an untampered hardware TEE.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttestationQuote {
    pub tee_type: TeeType,
    pub measurement: EnclaveMeasurement,
    /// 64-byte user report data cryptographically committed inside the hardware quote
    pub report_data: Vec<u8>,
    /// Hardware platform identity public key (e.g. Intel Quoting Enclave / AMD Platform Key)
    pub platform_public_key: [u8; 32],
    /// Hardware cryptographic signature over measurement, report_data, and security flags
    pub quote_signature: Vec<u8>,
    pub timestamp: u64,
    /// Hardware assurance & simulation detection flags
    pub security_flags: TeeSecurityFlags,
}

impl AttestationQuote {
    /// Construct the canonical byte payload signed by the hardware quoting key.
    /// Binds: tee_type || mrenclave || mrsigner || report_data || timestamp || is_simulation || debug_mode || hardware_level
    pub fn construct_quote_payload(
        tee_type: TeeType,
        measurement: &EnclaveMeasurement,
        report_data: &[u8],
        timestamp: u64,
        security_flags: &TeeSecurityFlags,
    ) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.push(tee_type as u8);
        payload.extend_from_slice(&measurement.mrenclave);
        payload.extend_from_slice(&measurement.mrsigner);
        payload.extend_from_slice(report_data);
        payload.extend_from_slice(&timestamp.to_be_bytes());
        payload.push(if security_flags.is_simulation { 1 } else { 0 });
        payload.push(if security_flags.debug_mode { 1 } else { 0 });
        payload.push(security_flags.hardware_level);
        payload
    }

    /// Get the signed canonical payload for this quote.
    pub fn payload(&self) -> Vec<u8> {
        Self::construct_quote_payload(
            self.tee_type,
            &self.measurement,
            &self.report_data,
            self.timestamp,
            &self.security_flags,
        )
    }

    /// Compute the deterministic 64-byte report data binding the quote to a specific round & ranking.
    /// report_data = SHA512(task_id || round || SHA256(ranking))
    pub fn compute_report_data(task_id: u64, round: usize, ranking: &[AccountId]) -> Vec<u8> {
        let mut ranking_hasher = Sha256::new();
        for m in ranking {
            ranking_hasher.update(m.0.as_bytes());
        }
        let ranking_hash = ranking_hasher.finalize();

        let mut report_hasher = Sha512::new();
        report_hasher.update(&task_id.to_be_bytes());
        report_hasher.update(&(round as u64).to_be_bytes());
        report_hasher.update(&ranking_hash);
        report_hasher.finalize().to_vec()
    }

    /// Check if the quote's report_data matches the claimed evaluation ranking.
    pub fn verify_report_data(&self, task_id: u64, round: usize, ranking: &[AccountId]) -> bool {
        let expected = Self::compute_report_data(task_id, round, ranking);
        self.report_data == expected
    }
}
