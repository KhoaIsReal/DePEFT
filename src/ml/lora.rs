use crate::blockchain::types::PeftType;
use crate::ml::tensor::{Matrix, QuantizedWeight};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Serialized LoRA adapter tensors for a single module.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModuleAdapter {
    pub module_name: String,
    pub rank: usize,
    pub alpha: f32,
    pub lora_a: Matrix, // (rank, in_features)
    pub lora_b: Matrix, // (out_features, rank)
}

impl ModuleAdapter {
    /// Compute the full weight delta matrix: $\Delta W = \frac{\alpha}{r} (B \times A)$
    pub fn compute_delta_w(&self) -> Matrix {
        let scaling = self.alpha / self.rank as f32;
        let ba = self.lora_b.matmul(&self.lora_a);
        ba.scale(scaling)
    }
}

/// A QLoRA-enabled Linear Layer supporting NF4, INT4, or FP32 base weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QLoRALinear {
    pub in_features: usize,
    pub out_features: usize,
    pub rank: usize,
    pub alpha: f32,
    pub peft_type: PeftType,
    pub base_weight: QuantizedWeight,
    pub lora_a: Matrix, // Shape: (rank, in_features)
    pub lora_b: Matrix, // Shape: (out_features, rank)
}

impl QLoRALinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        peft_type: PeftType,
        rng: &mut impl Rng,
    ) -> Self {
        // Initialize base weight matrix with Xavier uniform
        let base_fp32 = Matrix::xavier_uniform(out_features, in_features, rng);
        let base_weight = match peft_type {
            PeftType::LoRA => QuantizedWeight::FP32(base_fp32),
            PeftType::QLoRA_NF4 => QuantizedWeight::quantize_nf4(&base_fp32, 16),
            PeftType::QLoRA_INT4 => QuantizedWeight::quantize_int4(&base_fp32, 16),
        };

        // Initialize LoRA A with Gaussian N(0, 0.02) and LoRA B with 0s
        let lora_a = Matrix::random_normal(rank, in_features, 0.0, 0.02, rng);
        let lora_b = Matrix::zeros(out_features, rank);

        Self {
            in_features,
            out_features,
            rank,
            alpha,
            peft_type,
            base_weight,
            lora_a,
            lora_b,
        }
    }

    /// Forward pass: $Y = W \cdot X + \frac{\alpha}{r} (B \cdot A \cdot X)$
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        if input.len() != self.in_features {
            return vec![0.0; self.out_features];
        }

        let w = self.base_weight.dequantize();
        let input_mat = Matrix::new(self.in_features, 1, input.to_vec());

        // Base output: W * X -> (out_features, 1)
        let base_out = w.matmul(&input_mat);

        // LoRA output: scaling * B * (A * X)
        let scaling = self.alpha / self.rank as f32;
        let ax = self.lora_a.matmul(&input_mat); // (rank, 1)
        let bax = self.lora_b.matmul(&ax); // (out_features, 1)

        let mut out = base_out.data;
        for (i, val) in bax.data.iter().enumerate() {
            out[i] += scaling * val;
        }

        out
    }

    /// Compute gradients for LoRA A and LoRA B given input and output gradient:
    /// $\nabla B = \gamma (\text{grad\_out}) (A X)^T$
    /// $\nabla A = \gamma (B^T \text{grad\_out}) X^T$
    pub fn compute_gradients(&self, input: &[f32], grad_out: &[f32]) -> (Matrix, Matrix) {
        if input.len() != self.in_features || grad_out.len() != self.out_features {
            return (
                Matrix::zeros(self.rank, self.in_features),
                Matrix::zeros(self.out_features, self.rank),
            );
        }

        let scaling = self.alpha / self.rank as f32;
        let input_mat = Matrix::new(self.in_features, 1, input.to_vec());
        let grad_out_mat = Matrix::new(self.out_features, 1, grad_out.to_vec());

        // A * X -> (rank, 1)
        let ax = self.lora_a.matmul(&input_mat);
        // (A * X)^T -> (1, rank)
        let ax_t = ax.transpose();

        // grad_B = scaling * grad_out * (A * X)^T -> (out_features, rank)
        let grad_b = grad_out_mat.matmul(&ax_t).scale(scaling);

        // B^T * grad_out -> (rank, 1)
        let b_t = self.lora_b.transpose();
        let b_t_grad = b_t.matmul(&grad_out_mat);

        // X^T -> (1, in_features)
        let x_t = input_mat.transpose();

        // grad_A = scaling * (B^T * grad_out) * X^T -> (rank, in_features)
        let grad_a = b_t_grad.matmul(&x_t).scale(scaling);

        (grad_a, grad_b)
    }

    /// Apply gradient update step to LoRA matrices with gradient clipping and finite checks.
    pub fn update_lora(&mut self, grad_a: &Matrix, grad_b: &Matrix, lr: f32) {
        if !lr.is_finite() || lr <= 0.0 {
            return;
        }

        const GRAD_CLIP: f32 = 5.0;

        for (a_val, g_val) in self.lora_a.data.iter_mut().zip(&grad_a.data) {
            if g_val.is_finite() {
                let clamped_grad = g_val.clamp(-GRAD_CLIP, GRAD_CLIP);
                *a_val -= lr * clamped_grad;
            }
        }
        for (b_val, g_val) in self.lora_b.data.iter_mut().zip(&grad_b.data) {
            if g_val.is_finite() {
                let clamped_grad = g_val.clamp(-GRAD_CLIP, GRAD_CLIP);
                *b_val -= lr * clamped_grad;
            }
        }
    }

    /// Extract adapter matrices into ModuleAdapter.
    pub fn export_adapter(&self, module_name: &str) -> ModuleAdapter {
        ModuleAdapter {
            module_name: module_name.to_string(),
            rank: self.rank,
            alpha: self.alpha,
            lora_a: self.lora_a.clone(),
            lora_b: self.lora_b.clone(),
        }
    }

    /// Load an external adapter into this layer with shape validation.
    pub fn load_adapter(&mut self, adapter: &ModuleAdapter) -> anyhow::Result<()> {
        if adapter.lora_a.rows != self.rank
            || adapter.lora_a.cols != self.in_features
            || adapter.lora_b.rows != self.out_features
            || adapter.lora_b.cols != self.rank
        {
            anyhow::bail!(
                "Adapter dimension mismatch: expected ({}, {}), ({}, {}), got ({}, {}), ({}, {})",
                self.rank,
                self.in_features,
                self.out_features,
                self.rank,
                adapter.lora_a.rows,
                adapter.lora_a.cols,
                adapter.lora_b.rows,
                adapter.lora_b.cols
            );
        }
        self.lora_a = adapter.lora_a.clone();
        self.lora_b = adapter.lora_b.clone();
        Ok(())
    }

    /// Merge current LoRA weights permanently into Base Weights:
    /// $W_{N+1} = W_N + \frac{\alpha}{r} (B \times A)$
    /// and reset LoRA A and B for next ReLoRA round.
    pub fn merge_and_reset(&mut self, rng: &mut impl Rng) {
        let delta_w = {
            let scaling = self.alpha / self.rank as f32;
            let ba = self.lora_b.matmul(&self.lora_a);
            ba.scale(scaling)
        };

        let current_w = self.base_weight.dequantize();
        let evolved_w = current_w.add(&delta_w);

        // Re-quantize to the configured PEFT type
        self.base_weight = match self.peft_type {
            PeftType::LoRA => QuantizedWeight::FP32(evolved_w),
            PeftType::QLoRA_NF4 => QuantizedWeight::quantize_nf4(&evolved_w, 16),
            PeftType::QLoRA_INT4 => QuantizedWeight::quantize_int4(&evolved_w, 16),
        };

        // Reset LoRA matrices for next round
        self.lora_a = Matrix::random_normal(self.rank, self.in_features, 0.0, 0.02, rng);
        self.lora_b = Matrix::zeros(self.out_features, self.rank);
    }

    /// Set newly evolved base weights directly and re-initialize LoRA matrices.
    pub fn apply_evolved_weight(&mut self, evolved_w: Matrix, rng: &mut impl Rng) {
        self.base_weight = match self.peft_type {
            PeftType::LoRA => QuantizedWeight::FP32(evolved_w),
            PeftType::QLoRA_NF4 => QuantizedWeight::quantize_nf4(&evolved_w, 16),
            PeftType::QLoRA_INT4 => QuantizedWeight::quantize_int4(&evolved_w, 16),
        };
        self.lora_a = Matrix::random_normal(self.rank, self.in_features, 0.0, 0.02, rng);
        self.lora_b = Matrix::zeros(self.out_features, self.rank);
    }
}
