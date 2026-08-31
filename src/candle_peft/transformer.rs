use crate::candle_peft::lora::CandleLoraLinear;
use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Tensor, Var};
use candle_nn::loss::cross_entropy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for a Candle-based Decoder Transformer LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleTransformerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub lora_rank: usize,
    pub lora_alpha: f64,
}

impl Default for CandleTransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256, // Byte-level or subword vocab for fast demonstration
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            max_position_embeddings: 128,
            lora_rank: 8,
            lora_alpha: 16.0,
        }
    }
}

/// Root Mean Square Layer Normalization (RMSNorm) standard in LLaMA/Qwen.
#[derive(Clone)]
pub struct CandleRMSNorm {
    pub weight: Tensor, // Shape: [hidden_size]
    pub eps: f64,
}

impl CandleRMSNorm {
    pub fn new(hidden_size: usize, eps: f64, device: &Device) -> Result<Self> {
        let weight = Tensor::ones((hidden_size,), DType::F32, device)?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x shape: [batch, seq, hidden]
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let variance_eps = (variance + self.eps)?;
        let rsqrt = variance_eps.sqrt()?.recip()?;
        let norm_x = x.broadcast_mul(&rsqrt)?;
        let out = norm_x.broadcast_mul(&self.weight)?;
        Ok(out)
    }
}

/// Transformer Multi-Head Self-Attention block with QLoRA on projections.
#[derive(Clone)]
pub struct CandleAttentionBlock {
    pub q_proj: CandleLoraLinear,
    pub k_proj: CandleLoraLinear,
    pub v_proj: CandleLoraLinear,
    pub o_proj: CandleLoraLinear,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl CandleAttentionBlock {
    pub fn new(config: &CandleTransformerConfig, device: &Device) -> Result<Self> {
        let h = config.hidden_size;
        let rank = config.lora_rank;
        let alpha = config.lora_alpha;
        let head_dim = h / config.num_attention_heads;

        let q_w = Tensor::randn(0f32, 1.0 / (h as f32).sqrt(), (h, h), device)?;
        let k_w = Tensor::randn(0f32, 1.0 / (h as f32).sqrt(), (h, h), device)?;
        let v_w = Tensor::randn(0f32, 1.0 / (h as f32).sqrt(), (h, h), device)?;
        let o_w = Tensor::randn(0f32, 1.0 / (h as f32).sqrt(), (h, h), device)?;

        let q_proj = CandleLoraLinear::new(q_w, rank, alpha, device)?;
        let k_proj = CandleLoraLinear::new(k_w, rank, alpha, device)?;
        let v_proj = CandleLoraLinear::new(v_w, rank, alpha, device)?;
        let o_proj = CandleLoraLinear::new(o_w, rank, alpha, device)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.num_attention_heads,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [batch, num_heads, seq_len, head_dim]
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = (q.matmul(&k_t)? * scale)?;

        // Causal attention softmax
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let context = attn_weights.matmul(&v)?;

        // Transpose and flatten heads: [batch, seq_len, hidden]
        let context = context
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, hidden))?;

        let out = self.o_proj.forward(&context)?;
        Ok(out)
    }
}

/// Transformer SwiGLU Feed-Forward Network with QLoRA.
#[derive(Clone)]
pub struct CandleMlpBlock {
    pub gate_proj: CandleLoraLinear,
    pub up_proj: CandleLoraLinear,
    pub down_proj: CandleLoraLinear,
}

impl CandleMlpBlock {
    pub fn new(config: &CandleTransformerConfig, device: &Device) -> Result<Self> {
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let rank = config.lora_rank;
        let alpha = config.lora_alpha;

        let g_w = Tensor::randn(0f32, 1.0 / (h as f32).sqrt(), (inter, h), device)?;
        let u_w = Tensor::randn(0f32, 1.0 / (h as f32).sqrt(), (inter, h), device)?;
        let d_w = Tensor::randn(0f32, 1.0 / (inter as f32).sqrt(), (h, inter), device)?;

        Ok(Self {
            gate_proj: CandleLoraLinear::new(g_w, rank, alpha, device)?,
            up_proj: CandleLoraLinear::new(u_w, rank, alpha, device)?,
            down_proj: CandleLoraLinear::new(d_w, rank, alpha, device)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let silu_gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        let inter = silu_gate.mul(&up)?;
        let out = self.down_proj.forward(&inter)?;
        Ok(out)
    }
}

/// Single Transformer Decoder Layer (Pre-Norm + Self-Attn + SwiGLU MLP).
#[derive(Clone)]
pub struct CandleTransformerLayer {
    pub input_layernorm: CandleRMSNorm,
    pub self_attn: CandleAttentionBlock,
    pub post_attention_layernorm: CandleRMSNorm,
    pub mlp: CandleMlpBlock,
}

impl CandleTransformerLayer {
    pub fn new(config: &CandleTransformerConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            input_layernorm: CandleRMSNorm::new(config.hidden_size, 1e-5, device)?,
            self_attn: CandleAttentionBlock::new(config, device)?,
            post_attention_layernorm: CandleRMSNorm::new(config.hidden_size, 1e-5, device)?,
            mlp: CandleMlpBlock::new(config, device)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Residual Attention
        let norm_x = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&norm_x)?;
        let x = (x + &attn_out)?;

        // Residual MLP
        let norm_x2 = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&norm_x2)?;
        let out = (x + &mlp_out)?;
        Ok(out)
    }
}

/// Full Decoder-only Language Model with QLoRA adapters across all layers.
#[derive(Clone)]
pub struct CandleTransformerLM {
    pub config: CandleTransformerConfig,
    pub token_embedding: Tensor, // [vocab_size, hidden_size]
    pub layers: Vec<CandleTransformerLayer>,
    pub final_norm: CandleRMSNorm,
    pub lm_head: CandleLoraLinear,
    pub device: Device,
}

impl CandleTransformerLM {
    pub fn new(config: CandleTransformerConfig, device: Device) -> Result<Self> {
        let h = config.hidden_size;
        let v = config.vocab_size;

        let token_embedding = Tensor::randn(0f32, 0.02, (v, h), &device)?;
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(CandleTransformerLayer::new(&config, &device)?);
        }

        let final_norm = CandleRMSNorm::new(h, 1e-5, &device)?;
        let lm_head_w = Tensor::randn(0f32, 0.02, (v, h), &device)?;
        let lm_head = CandleLoraLinear::new(lm_head_w, config.lora_rank, config.lora_alpha, &device)?;

        Ok(Self {
            config,
            token_embedding,
            layers,
            final_norm,
            lm_head,
            device,
        })
    }

    /// Forward pass: returns logits `[batch, seq_len, vocab_size]`
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (batch, seq_len) = input_ids.dims2()?;
        let flat_ids = input_ids.flatten_all()?;
        let embeds = self.token_embedding.index_select(&flat_ids, 0)?;
        let mut hidden_states = embeds.reshape((batch, seq_len, self.config.hidden_size))?;

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        let normed = self.final_norm.forward(&hidden_states)?;
        let logits = self.lm_head.forward(&normed)?;
        Ok(logits)
    }

    /// Compute autoregressive cross entropy loss over a batch of token sequences.
    pub fn forward_loss(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_batch, seq_len) = input_ids.dims2()?;
        if seq_len < 2 {
            bail!("Sequence length must be at least 2 for autoregressive language modeling");
        }

        let logits = self.forward(input_ids)?;

        // Shift for next-token prediction
        // inputs: tokens 0 .. seq_len-2
        // targets: tokens 1 .. seq_len-1
        let shifted_logits = logits.i((.., ..seq_len - 1, ..))?;
        let targets = input_ids.i((.., 1..))?;

        let (b, s, v) = shifted_logits.dims3()?;
        let flat_logits = shifted_logits.reshape((b * s, v))?;
        let flat_targets = targets.reshape((b * s,))?;

        let loss = cross_entropy(&flat_logits, &flat_targets)?;
        Ok(loss)
    }

    /// ReLoRA Permanent Weight Merge: fuses all LoRA adapters into base weights.
    pub fn merge_all_adapters(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            layer.self_attn.q_proj.merge_and_reset()?;
            layer.self_attn.k_proj.merge_and_reset()?;
            layer.self_attn.v_proj.merge_and_reset()?;
            layer.self_attn.o_proj.merge_and_reset()?;
            layer.mlp.gate_proj.merge_and_reset()?;
            layer.mlp.up_proj.merge_and_reset()?;
            layer.mlp.down_proj.merge_and_reset()?;
        }
        self.lm_head.merge_and_reset()?;
        Ok(())
    }

    /// Collect all trainable LoRA Var parameters across all layers.
    pub fn get_trainable_vars(&self) -> Vec<Var> {
        let mut vars = Vec::new();
        for layer in &self.layers {
            vars.push(layer.self_attn.q_proj.lora_a.clone());
            vars.push(layer.self_attn.q_proj.lora_b.clone());
            vars.push(layer.self_attn.k_proj.lora_a.clone());
            vars.push(layer.self_attn.k_proj.lora_b.clone());
            vars.push(layer.self_attn.v_proj.lora_a.clone());
            vars.push(layer.self_attn.v_proj.lora_b.clone());
            vars.push(layer.self_attn.o_proj.lora_a.clone());
            vars.push(layer.self_attn.o_proj.lora_b.clone());
            vars.push(layer.mlp.gate_proj.lora_a.clone());
            vars.push(layer.mlp.gate_proj.lora_b.clone());
            vars.push(layer.mlp.up_proj.lora_a.clone());
            vars.push(layer.mlp.up_proj.lora_b.clone());
            vars.push(layer.mlp.down_proj.lora_a.clone());
            vars.push(layer.mlp.down_proj.lora_b.clone());
        }
        vars.push(self.lm_head.lora_a.clone());
        vars.push(self.lm_head.lora_b.clone());
        vars
    }

    /// Export all adapter tensors in named map.
    pub fn export_adapter_tensors(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (k, v) in layer.self_attn.q_proj.export_tensors(&format!("model.layers.{}.self_attn.q_proj", i)) {
                map.insert(k, v);
            }
            for (k, v) in layer.self_attn.k_proj.export_tensors(&format!("model.layers.{}.self_attn.k_proj", i)) {
                map.insert(k, v);
            }
            for (k, v) in layer.self_attn.v_proj.export_tensors(&format!("model.layers.{}.self_attn.v_proj", i)) {
                map.insert(k, v);
            }
            for (k, v) in layer.self_attn.o_proj.export_tensors(&format!("model.layers.{}.self_attn.o_proj", i)) {
                map.insert(k, v);
            }
            for (k, v) in layer.mlp.gate_proj.export_tensors(&format!("model.layers.{}.mlp.gate_proj", i)) {
                map.insert(k, v);
            }
            for (k, v) in layer.mlp.up_proj.export_tensors(&format!("model.layers.{}.mlp.up_proj", i)) {
                map.insert(k, v);
            }
            for (k, v) in layer.mlp.down_proj.export_tensors(&format!("model.layers.{}.mlp.down_proj", i)) {
                map.insert(k, v);
            }
        }
        for (k, v) in self.lm_head.export_tensors("lm_head") {
            map.insert(k, v);
        }
        map
    }

    /// Total trainable LoRA parameter count.
    pub fn total_trainable_parameters(&self) -> usize {
        self.get_trainable_vars().iter().map(|v| v.as_tensor().elem_count()).sum()
    }
}
