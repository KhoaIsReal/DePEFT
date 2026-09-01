use crate::ml::dataset::Dataset;
use crate::ml::model::{AdapterPackage, DePEFTModel};

/// Trusted Execution Environment (TEE) Sandbox Enclave for protecting Private Test Sets.
/// Prevents data leakage and overfitting by miners as specified in Section 4.1.
#[derive(Debug, Clone)]
pub struct TeeSandbox {
    private_test_set: Dataset,
    enclave_id: String,
}

impl TeeSandbox {
    pub fn new(private_test_set: Dataset, enclave_id: impl Into<String>) -> Self {
        Self {
            private_test_set,
            enclave_id: enclave_id.into(),
        }
    }

    /// Run isolated inference inside TEE and export only final metrics (loss, accuracy).
    /// Private test samples are never leaked outside the enclave.
    pub fn evaluate_adapter(
        &self,
        base_model: &DePEFTModel,
        adapter: &AdapterPackage,
        hardware_drift: f32,
    ) -> (f64, f64) {
        let mut model = base_model.clone();
        if model.load_adapters(adapter).is_err() {
            return (9999.0, 0.0);
        }
        model.evaluate(&self.private_test_set, hardware_drift)
    }

    pub fn enclave_id(&self) -> &str {
        &self.enclave_id
    }

    pub fn test_set_size(&self) -> usize {
        self.private_test_set.len()
    }
}
