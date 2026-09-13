use crate::candle_peft::transformer::CandleTransformerLM;
use anyhow::{Context, Result};
use candle_core::safetensors::load_buffer;

/// ReLoRA Permanent Weight Merger.
pub struct CandleWeightMerger;

impl CandleWeightMerger {
    /// Fuse winning LoRA adapter weights into base model weights: W_{N+1} = W_N + (alpha / r) * (B @ A).
    pub fn merge_winning_adapter(
        base_model: &mut CandleTransformerLM,
        safetensors_bytes: &[u8],
    ) -> Result<()> {
        let tensor_map = load_buffer(safetensors_bytes, &base_model.device)
            .context("Failed to load winning safetensors buffer")?;

        // 1. Load winning adapter matrices across all adapted layers
        base_model.load_adapter_tensors(&tensor_map)?;

        // 2. Perform permanent weight fusion
        base_model.merge_all_adapters()?;
        Ok(())
    }
}
