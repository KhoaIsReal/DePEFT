use crate::blockchain::types::PeftType;
use crate::ml::dataset::Dataset;
use crate::ml::lora::{ModuleAdapter, QLoRALinear};
use crate::ml::tensor::{Matrix, QuantizedWeight};
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

    /// Compute the maximum Frobenius norm of delta weights across all adapted modules.
    pub fn max_delta_norm(&self) -> f32 {
        self.modules
            .values()
            .map(|m| m.compute_delta_w().frobenius_norm())
            .fold(0.0f32, f32::max)
    }

    /// Check if the adapter exhibits weight anomalies (e.g. non-finite values or extreme delta norm explosion).
    pub fn is_weight_anomalous(&self, max_allowed_norm: f32) -> bool {
        for adapter in self.modules.values() {
            let delta = adapter.compute_delta_w();
            for &val in &delta.data {
                if !val.is_finite() {
                    return true;
                }
            }
            if delta.frobenius_norm() > max_allowed_norm {
                return true;
            }
        }
        false
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
    pub fn train_epoch(
        &mut self,
        dataset: &Dataset,
        learning_rate: f32,
        target_modules: &[String],
    ) {
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
        pkg.modules
            .insert("q_proj".into(), self.q_proj.export_adapter("q_proj"));
        pkg.modules
            .insert("v_proj".into(), self.v_proj.export_adapter("v_proj"));
        pkg.modules
            .insert("out_proj".into(), self.out_proj.export_adapter("out_proj"));
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

    /// Merge multiple candidate adapters into base weights using weighted ensemble fusion.
    pub fn merge_and_evolve_ensemble(
        &mut self,
        weighted_packages: &[(&AdapterPackage, f32)],
        rng: &mut impl Rng,
    ) -> anyhow::Result<()> {
        if weighted_packages.is_empty() {
            return Ok(());
        }

        // Calculate combined delta for each layer
        let mut q_delta_sum = None;
        let mut v_delta_sum = None;
        let mut out_delta_sum = None;

        for &(pkg, weight) in weighted_packages {
            if let Some(q) = pkg.modules.get("q_proj") {
                let delta = q.compute_delta_w().scale(weight);
                q_delta_sum = Some(match q_delta_sum {
                    None => delta,
                    Some(sum) => Matrix::add(&sum, &delta),
                });
            }
            if let Some(v) = pkg.modules.get("v_proj") {
                let delta = v.compute_delta_w().scale(weight);
                v_delta_sum = Some(match v_delta_sum {
                    None => delta,
                    Some(sum) => Matrix::add(&sum, &delta),
                });
            }
            if let Some(out) = pkg.modules.get("out_proj") {
                let delta = out.compute_delta_w().scale(weight);
                out_delta_sum = Some(match out_delta_sum {
                    None => delta,
                    Some(sum) => Matrix::add(&sum, &delta),
                });
            }
        }

        if let Some(delta) = q_delta_sum {
            let cur = self.q_proj.base_weight.dequantize();
            let evolved = cur.add(&delta);
            self.q_proj.base_weight = match self.peft_type {
                PeftType::LoRA => QuantizedWeight::FP32(evolved),
                PeftType::QLoRA_NF4 => QuantizedWeight::quantize_nf4(&evolved, 16),
                PeftType::QLoRA_INT4 => QuantizedWeight::quantize_int4(&evolved, 16),
            };
            self.q_proj.lora_a =
                Matrix::random_normal(self.q_proj.rank, self.q_proj.in_features, 0.0, 0.02, rng);
            self.q_proj.lora_b = Matrix::zeros(self.q_proj.out_features, self.q_proj.rank);
        }

        if let Some(delta) = v_delta_sum {
            let cur = self.v_proj.base_weight.dequantize();
            let evolved = cur.add(&delta);
            self.v_proj.base_weight = match self.peft_type {
                PeftType::LoRA => QuantizedWeight::FP32(evolved),
                PeftType::QLoRA_NF4 => QuantizedWeight::quantize_nf4(&evolved, 16),
                PeftType::QLoRA_INT4 => QuantizedWeight::quantize_int4(&evolved, 16),
            };
            self.v_proj.lora_a =
                Matrix::random_normal(self.v_proj.rank, self.v_proj.in_features, 0.0, 0.02, rng);
            self.v_proj.lora_b = Matrix::zeros(self.v_proj.out_features, self.v_proj.rank);
        }

        if let Some(delta) = out_delta_sum {
            let cur = self.out_proj.base_weight.dequantize();
            let evolved = cur.add(&delta);
            self.out_proj.base_weight = match self.peft_type {
                PeftType::LoRA => QuantizedWeight::FP32(evolved),
                PeftType::QLoRA_NF4 => QuantizedWeight::quantize_nf4(&evolved, 16),
                PeftType::QLoRA_INT4 => QuantizedWeight::quantize_int4(&evolved, 16),
            };
            self.out_proj.lora_a = Matrix::random_normal(
                self.out_proj.rank,
                self.out_proj.in_features,
                0.0,
                0.02,
                rng,
            );
            self.out_proj.lora_b = Matrix::zeros(self.out_proj.out_features, self.out_proj.rank);
        }

        Ok(())
    }

    /// DiLoCo / FedAdam: Merge candidate adapters using an outer optimizer with momentum and second moment dampening.
    /// Dampens conflicting or high-variance dimensions across divergent local updates while accumulating consensus directions.
    pub fn merge_and_evolve_outer_optimizer(
        &mut self,
        weighted_packages: &[(&AdapterPackage, f32)],
        outer_state: &mut OuterOptimizerState,
        outer_lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        rng: &mut impl Rng,
    ) -> anyhow::Result<()> {
        if weighted_packages.is_empty() {
            return Ok(());
        }

        let mut q_delta_sum = None;
        let mut v_delta_sum = None;
        let mut out_delta_sum = None;

        for &(pkg, weight) in weighted_packages {
            if let Some(q) = pkg.modules.get("q_proj") {
                let delta = q.compute_delta_w().scale(weight);
                q_delta_sum = Some(match q_delta_sum {
                    None => delta,
                    Some(sum) => Matrix::add(&sum, &delta),
                });
            }
            if let Some(v) = pkg.modules.get("v_proj") {
                let delta = v.compute_delta_w().scale(weight);
                v_delta_sum = Some(match v_delta_sum {
                    None => delta,
                    Some(sum) => Matrix::add(&sum, &delta),
                });
            }
            if let Some(out) = pkg.modules.get("out_proj") {
                let delta = out.compute_delta_w().scale(weight);
                out_delta_sum = Some(match out_delta_sum {
                    None => delta,
                    Some(sum) => Matrix::add(&sum, &delta),
                });
            }
        }

        outer_state.step_count += 1;
        let t = outer_state.step_count as f32;
        let bias_c1 = 1.0 - beta1.powf(t);
        let bias_c2 = 1.0 - beta2.powf(t);

        // Helper to compute outer Adam/DiLoCo step for a module
        let compute_effective_step = |pseudo_grad: &Matrix,
                                      m_opt: &mut Option<Matrix>,
                                      v_opt: &mut Option<Matrix>|
         -> Matrix {
            let m = m_opt.get_or_insert_with(|| Matrix::zeros(pseudo_grad.rows, pseudo_grad.cols));
            let v = v_opt.get_or_insert_with(|| Matrix::zeros(pseudo_grad.rows, pseudo_grad.cols));

            let mut step_data = Vec::with_capacity(pseudo_grad.data.len());
            for i in 0..pseudo_grad.data.len() {
                let g = pseudo_grad.data[i];
                m.data[i] = beta1 * m.data[i] + (1.0 - beta1) * g;
                v.data[i] = beta2 * v.data[i] + (1.0 - beta2) * (g * g);

                let m_hat = m.data[i] / bias_c1;
                let v_hat = v.data[i] / bias_c2;

                // Conflicting coordinates have high variance v_hat, dampening the effective update.
                let update = outer_lr * m_hat / (v_hat.sqrt() + eps);
                step_data.push(update);
            }
            Matrix::new(pseudo_grad.rows, pseudo_grad.cols, step_data)
        };

        if let Some(delta_q) = q_delta_sum {
            let eff_step = compute_effective_step(
                &delta_q,
                &mut outer_state.m_q_proj,
                &mut outer_state.v_q_proj,
            );
            let cur = self.q_proj.base_weight.dequantize();
            let evolved = cur.add(&eff_step);
            self.q_proj.apply_evolved_weight(evolved, rng);
        }

        if let Some(delta_v) = v_delta_sum {
            let eff_step = compute_effective_step(
                &delta_v,
                &mut outer_state.m_v_proj,
                &mut outer_state.v_v_proj,
            );
            let cur = self.v_proj.base_weight.dequantize();
            let evolved = cur.add(&eff_step);
            self.v_proj.apply_evolved_weight(evolved, rng);
        }

        if let Some(delta_out) = out_delta_sum {
            let eff_step = compute_effective_step(
                &delta_out,
                &mut outer_state.m_out_proj,
                &mut outer_state.v_out_proj,
            );
            let cur = self.out_proj.base_weight.dequantize();
            let evolved = cur.add(&eff_step);
            self.out_proj.apply_evolved_weight(evolved, rng);
        }

        Ok(())
    }
}

/// Global / Outer Optimizer state (tracking momentum and second-moment variance across rounds).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OuterOptimizerState {
    pub step_count: usize,
    pub m_q_proj: Option<Matrix>,
    pub v_q_proj: Option<Matrix>,
    pub m_v_proj: Option<Matrix>,
    pub v_v_proj: Option<Matrix>,
    pub m_out_proj: Option<Matrix>,
    pub v_out_proj: Option<Matrix>,
}
