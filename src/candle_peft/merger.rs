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

        // 1. Load winning adapter matrices into layers
        for (i, layer) in base_model.layers.iter_mut().enumerate() {
            let q_a = tensor_map.get(&format!("model.layers.{}.self_attn.q_proj.lora_a.weight", i));
            let q_b = tensor_map.get(&format!("model.layers.{}.self_attn.q_proj.lora_b.weight", i));
            if let (Some(a), Some(b)) = (q_a, q_b) {
                layer.self_attn.q_proj.load_adapter(a, b)?;
            }

            let v_a = tensor_map.get(&format!("model.layers.{}.self_attn.v_proj.lora_a.weight", i));
            let v_b = tensor_map.get(&format!("model.layers.{}.self_attn.v_proj.lora_b.weight", i));
            if let (Some(a), Some(b)) = (v_a, v_b) {
                layer.self_attn.v_proj.load_adapter(a, b)?;
            }
        }

        // 2. Perform permanent weight fusion
        base_model.merge_all_adapters()?;
        Ok(())
    }
}
