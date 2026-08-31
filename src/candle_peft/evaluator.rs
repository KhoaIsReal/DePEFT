use crate::blockchain::types::{AccountId, ValidatorEvaluation};
use crate::candle_peft::tokenizer::SimpleByteTokenizer;
use crate::candle_peft::transformer::CandleTransformerLM;
use anyhow::{Context, Result};
use candle_core::safetensors::load_buffer;

/// Evaluator worker running inside TEE Sandbox to evaluate Candle model adapters.
pub struct CandleValidatorEvaluator;

impl CandleValidatorEvaluator {
    /// Evaluate a single model on a private test set of text strings.
    pub fn evaluate_dataset(
        model: &CandleTransformerLM,
        test_samples: &[String],
    ) -> Result<f32> {
        let tokenizer = SimpleByteTokenizer::new();
        let encoded_samples: Vec<Vec<u32>> = test_samples
            .iter()
            .map(|text| tokenizer.encode(text))
            .filter(|v| v.len() >= 2)
            .collect();

        if encoded_samples.is_empty() {
            anyhow::bail!("No valid test samples");
        }

        let batch_tensor = tokenizer.batch_to_tensor(&encoded_samples, &model.device)?;
        let loss = model.forward_loss(&batch_tensor)?;
        let loss_val = loss.to_scalar::<f32>()?;
        Ok(loss_val)
    }

    /// Evaluate candidate miners on the private test set and return relative ordinal ranking.
    pub fn evaluate_miners(
        base_model: &CandleTransformerLM,
        test_samples: &[String],
        candidate_adapters: &[(AccountId, Vec<u8>)],
        validator_address: AccountId,
        hardware_info: &str,
        float_drift: f32,
    ) -> Result<ValidatorEvaluation> {
        let mut scores = Vec::new();

        for (miner_id, safetensors_bytes) in candidate_adapters {
            let mut model_clone = base_model.clone();

            // Load adapter tensors from safetensors buffer
            let tensor_map = load_buffer(safetensors_bytes, &model_clone.device)
                .context("Failed to parse safetensors buffer")?;

            // Attach adapter weights into model layers
            for (i, layer) in model_clone.layers.iter_mut().enumerate() {
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

            let mut loss = Self::evaluate_dataset(&model_clone, test_samples)?;
            // Apply micro-drift simulating heterogeneous CUDA/ROCm floating point tolerances
            loss += float_drift;

            scores.push((miner_id.clone(), loss));
        }

        // Sort ascending by loss (lowest loss is top rank)
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let ranking: Vec<AccountId> = scores.iter().map(|(m, _)| m.clone()).collect();
        let loss_scores: Vec<(AccountId, f64)> = scores.iter().map(|(m, l)| (m.clone(), *l as f64)).collect();
        let accuracy_scores: Vec<(AccountId, f64)> = scores
            .iter()
            .map(|(m, l)| (m.clone(), 1.0 / (1.0 + *l as f64)))
            .collect();

        let enclave = crate::tee::HardwareTeeEnclave::official(crate::tee::TeeType::IntelSgxDcap);
        let quote = enclave.generate_quote(1, 1, &ranking).ok();

        Ok(ValidatorEvaluation {
            validator_address,
            ranking,
            loss_scores,
            accuracy_scores,
            hardware_info: hardware_info.to_string(),
            attestation_quote: quote,
        })
    }
}
