use crate::blockchain::types::PeftType;
use crate::ml::dataset::Dataset;
use crate::ml::lora::{ModuleAdapter, QLoRALinear};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete adapter package for a multi-module model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterPackage {
    pub model_id: String,
    pub round: usize,
    pub peft_type: PeftType,
    pub modules: HashMap<String, ModuleAdapter>,
}

impl AdapterPackage {
    pub fn new(model_id: impl Into<String>, round: usize, peft_type: PeftType) -> Self {
        Self {
            model_id: model_id.into(),
            round,
            peft_type,
            modules: HashMap::new(),
        }
    }

    /// Compute a compact signature / embedding vector of the adapter for the Vector DB.
    pub fn generate_signature_vector(&self, dim: usize) -> Vec<f32> {
        let mut signature = vec![0.0f32; dim];
        let mut idx = 0;

        for adapter in self.modules.values() {
            let delta = adapter.compute_delta_w();
            for &val in &delta.data {
                if val.is_finite() {
                    signature[idx % dim] += val;
                }
                idx += 1;
            }
        }

        // Normalize vector to unit length
        let norm = signature.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for v in signature.iter_mut() {
                *v /= norm;
            }
        } else {
            // Fallback default unit vector
            signature[0] = 1.0;
        }

        signature
    }
}

/// A multi-layer PEFT neural model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DePEFTModel {
    pub model_id: String,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub rank: usize,
    pub peft_type: PeftType,
    pub q_proj: QLoRALinear,
    pub v_proj: QLoRALinear,
    pub out_proj: QLoRALinear,
}

impl DePEFTModel {
    pub fn new(
        model_id: impl Into<String>,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        rank: usize,
        peft_type: PeftType,
        rng: &mut impl Rng,
    ) -> Self {
        let alpha = 16.0;
        let q_proj = QLoRALinear::new(input_dim, hidden_dim, rank, alpha, peft_type, rng);
        let v_proj = QLoRALinear::new(input_dim, hidden_dim, rank, alpha, peft_type, rng);
        let out_proj = QLoRALinear::new(hidden_dim, output_dim, rank, alpha, peft_type, rng);

        Self {
            model_id: model_id.into(),
            input_dim,
            hidden_dim,
            output_dim,
            rank,
            peft_type,
            q_proj,
            v_proj,
            out_proj,
        }
    }

    /// Forward pass through network: Input -> [q_proj + v_proj activation] -> out_proj -> Output
    pub fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q = self.q_proj.forward(input);
        let v = self.v_proj.forward(input);

        // Combined hidden layer with tanh non-linearity
        let mut hidden = vec![0.0f32; self.hidden_dim];
        for i in 0..self.hidden_dim {
            hidden[i] = (q[i] + v[i]).tanh();
        }

        let output = self.out_proj.forward(&hidden);
        (q, hidden, output)
    }

    /// Compute Mean Squared Error Loss and Accuracy on a dataset.
    /// Simulates hardware floating point drift parameter when specified.
    pub fn evaluate(&self, dataset: &Dataset, drift_epsilon: f32) -> (f64, f64) {
        if dataset.is_empty() {
            return (0.0, 1.0);
        }

        let mut total_loss = 0.0;
        let mut correct = 0;

        for sample in &dataset.samples {
            let (_, _, pred) = self.forward(&sample.input);
            let mut sample_loss = 0.0;
            let mut sample_close = true;

            for (p, t) in pred.iter().zip(&sample.target) {
                let diff = p - t;
                sample_loss += (diff * diff) as f64;
                if (p - t).abs() > 0.4 {
                    sample_close = false;
                }
            }

            total_loss += sample_loss / sample.target.len() as f64;
            if sample_close {
                correct += 1;
            }
        }

        let mut avg_loss = (total_loss / dataset.len() as f64) + (drift_epsilon as f64);
        if !avg_loss.is_finite() {
            avg_loss = 9999.0;
        }
        let accuracy = correct as f64 / dataset.len() as f64;
        (avg_loss, accuracy)
    }

    /// Fine-tune LoRA adapter layers on a training dataset using gradient descent.
    pub fn train_epoch(&mut self, dataset: &Dataset, learning_rate: f32, target_modules: &[String]) {
        for sample in &dataset.samples {
            let (_, hidden, pred) = self.forward(&sample.input);

            // Compute output loss gradient with clipping: dL/dPred = 2 * (pred - target) / N
            let mut grad_out = vec![0.0f32; self.output_dim];
            for (i, (p, t)) in pred.iter().zip(&sample.target).enumerate() {
                let g = 2.0 * (p - t) / self.output_dim as f32;
                grad_out[i] = g.clamp(-1.0, 1.0);
            }

            // Train out_proj if targeted
            if target_modules.is_empty() || target_modules.iter().any(|m| m == "out_proj") {
                let (grad_a, grad_b) = self.out_proj.compute_gradients(&hidden, &grad_out);
                self.out_proj.update_lora(&grad_a, &grad_b, learning_rate);
            }

            // Backprop gradient into hidden layer: dL/dHidden = (out_proj_W + delta_W)^T * grad_out
            let out_w = self.out_proj.base_weight.dequantize();
            let out_delta = self.out_proj.export_adapter("out_proj").compute_delta_w();
            let effective_out_w = out_w.add(&out_delta);

            let mut grad_hidden = vec![0.0f32; self.hidden_dim];
            for r in 0..self.out_proj.out_features {
                for c in 0..self.out_proj.in_features {
                    grad_hidden[c] += effective_out_w.get(r, c) * grad_out[r];
                }
            }

            // Non-linear derivative: d(tanh(x))/dx = 1 - tanh(x)^2 = 1 - hidden[i]^2
            let mut grad_act = vec![0.0f32; self.hidden_dim];
            for i in 0..self.hidden_dim {
                let dtanh = 1.0 - hidden[i] * hidden[i];
                grad_act[i] = grad_hidden[i] * dtanh;
            }

            // Train q_proj and v_proj if targeted
            if target_modules.is_empty() || target_modules.iter().any(|m| m == "q_proj") {
                let (g_a, g_b) = self.q_proj.compute_gradients(&sample.input, &grad_act);
                self.q_proj.update_lora(&g_a, &g_b, learning_rate);
            }

            if target_modules.is_empty() || target_modules.iter().any(|m| m == "v_proj") {
                let (g_a, g_b) = self.v_proj.compute_gradients(&sample.input, &grad_act);
                self.v_proj.update_lora(&g_a, &g_b, learning_rate);
            }
        }
    }

    /// Export current adapter weights.
    pub fn export_adapters(&self, round: usize) -> AdapterPackage {
        let mut pkg = AdapterPackage::new(&self.model_id, round, self.peft_type);
        pkg.modules.insert("q_proj".into(), self.q_proj.export_adapter("q_proj"));
        pkg.modules.insert("v_proj".into(), self.v_proj.export_adapter("v_proj"));
        pkg.modules.insert("out_proj".into(), self.out_proj.export_adapter("out_proj"));
        pkg
    }

    /// Load an external adapter package into the model for evaluation.
    pub fn load_adapters(&mut self, package: &AdapterPackage) -> anyhow::Result<()> {
        if let Some(q) = package.modules.get("q_proj") {
            self.q_proj.load_adapter(q)?;
        }
        if let Some(v) = package.modules.get("v_proj") {
            self.v_proj.load_adapter(v)?;
        }
        if let Some(out) = package.modules.get("out_proj") {
            self.out_proj.load_adapter(out)?;
        }
        Ok(())
    }

    /// Merge the loaded adapter into base weights and reset adapters for next round (ReLoRA merge).
    pub fn merge_and_evolve(&mut self, rng: &mut impl Rng) {
        self.q_proj.merge_and_reset(rng);
        self.v_proj.merge_and_reset(rng);
        self.out_proj.merge_and_reset(rng);
    }
}
