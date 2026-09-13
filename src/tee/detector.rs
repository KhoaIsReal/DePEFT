use crate::tee::types::TeeType;
use std::path::Path;

/// Status of hardware TEE detection on the current host machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostTeeStatus {
    /// Native hardware TEE driver and character device detected
    HardwareAvailable {
        tee_type: TeeType,
        device_path: String,
    },
    /// No supported hardware TEE device found; must fall back to simulation
    SimulationOnly {
        reason: String,
    },
}

/// Detect whether the host machine is running inside a genuine hardware TEE
/// (Intel SGX, Intel TDX, AMD SEV-SNP, or AWS Nitro) or a software environment.
pub fn detect_host_tee() -> HostTeeStatus {
    // 1. Check for Intel TDX (Trust Domain Extensions) guest devices (QEMU / KVM)
    for dev in &[
        "/dev/tdx_guest",
        "/dev/tdx-guest",
        "/dev/tdx_attest",
        "/sys/devices/virtual/misc/tdx_guest",
    ] {
        if Path::new(dev).exists() {
            return HostTeeStatus::HardwareAvailable {
                tee_type: TeeType::IntelTdx,
                device_path: dev.to_string(),
            };
        }
    }
    if Path::new("/sys/firmware/tdx").exists() {
        return HostTeeStatus::HardwareAvailable {
            tee_type: TeeType::IntelTdx,
            device_path: "/sys/firmware/tdx".to_string(),
        };
    }

    // 2. Check for Intel SGX (Software Guard Extensions) device nodes
    for dev in &["/dev/sgx_enclave", "/dev/sgx/enclave", "/dev/isgx"] {
        if Path::new(dev).exists() {
            return HostTeeStatus::HardwareAvailable {
                tee_type: TeeType::IntelSgxDcap,
                device_path: dev.to_string(),
            };
        }
    }

    // 3. Check for AMD SEV-SNP (Secure Encrypted Virtualization) guest devices (QEMU / KVM)
    for dev in &[
        "/dev/sev-guest",
        "/dev/sev",
        "/sys/devices/virtual/misc/sev-guest",
    ] {
        if Path::new(dev).exists() {
            return HostTeeStatus::HardwareAvailable {
                tee_type: TeeType::AmdSevSnp,
                device_path: dev.to_string(),
            };
        }
    }

    // 4. Check for AWS Nitro Enclaves device node
    if Path::new("/dev/nitro_enclaves").exists() {
        return HostTeeStatus::HardwareAvailable {
            tee_type: TeeType::AwsNitroEnclave,
            device_path: "/dev/nitro_enclaves".to_string(),
        };
    }

    HostTeeStatus::SimulationOnly {
        reason: "No physical TEE character devices (/dev/tdx*, /dev/sgx*, /dev/sev*, /dev/nitro*) detected on host OS".to_string(),
    }
}

/// Helper to check if genuine hardware is available for a requested TEE type.
pub fn is_hardware_tee_available(requested_type: TeeType) -> bool {
    match detect_host_tee() {
        HostTeeStatus::HardwareAvailable { tee_type, .. } => tee_type == requested_type,
        HostTeeStatus::SimulationOnly { .. } => false,
    }
}
