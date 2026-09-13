pub mod detector;
pub mod enclave;
pub mod types;
pub mod verifier;

pub use detector::{detect_host_tee, is_hardware_tee_available, HostTeeStatus};
pub use enclave::HardwareTeeEnclave;
pub use types::{AttestationQuote, EnclaveMeasurement, TeeSecurityFlags, TeeType};
pub use verifier::OnChainTeeVerifier;

