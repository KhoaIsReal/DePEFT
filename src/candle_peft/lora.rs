use anyhow::Result;
use candle_core::{DType, Device, Tensor, Var};

/// A LoRA-augmented Linear projection layer using Candle Tensors.
#[derive(Clone)]
pub struct CandleLoraLinear {
    pub base_weight: Tensor, // Shape: [d_out, d_in] (Frozen base model weight)
    pub lora_a: Var,         // Shape: [rank, d_in] (Down-projection, trainable)
    pub lora_b: Var,         // Shape: [d_out, rank] (Up-projection, trainable)
    pub rank: usize,
    pub alpha: f64,
    pub scale: f64,
}

impl CandleLoraLinear {
    /// Initialize a new CandleLoraLinear attached to a base weight tensor.
    pub fn new(base_weight: Tensor, rank: usize, alpha: f64, device: &Device) -> Result<Self> {
        let (d_out, d_in) = base_weight.dims2()?;
        let scale = alpha / (rank as f64);

        // Kaiming uniform / Gaussian init for LoRA A
        let std_dev = 1.0 / (rank as f64).sqrt();
        let a_init = (Tensor::randn(0f32, std_dev as f32, (rank, d_in), device)? * 0.1)?;
        let lora_a = Var::from_tensor(&a_init)?;

        // Zero init for LoRA B so ΔW = 0 at initialization
        let b_init = Tensor::zeros((d_out, rank), DType::F32, device)?;
        let lora_b = Var::from_tensor(&b_init)?;

        Ok(Self {
            base_weight,
            lora_a,
            lora_b,
            rank,
            alpha,
            scale,
        })
    }

    /// Forward pass: y = x W^T + (alpha / r) * (x A^T) B^T
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dims = x.dims();
        let hidden = *orig_dims.last().unwrap();
        let flat_x = x.reshape(((), hidden))?;

        // Base forward: flat_x @ W^T
        let w_t = self.base_weight.t()?;
        let base_out = flat_x.matmul(&w_t)?;

        // LoRA forward: (flat_x @ A^T) @ B^T * scale
        let a_t = self.lora_a.as_tensor().t()?;
        let b_t = self.lora_b.as_tensor().t()?;

        let lora_down = flat_x.matmul(&a_t)?;
        let lora_up = lora_down.matmul(&b_t)?;
        let lora_scaled = (lora_up * self.scale)?;

        let flat_out = (&base_out + &lora_scaled)?;

        let (d_out, _) = self.base_weight.dims2()?;
        let mut out_dims = orig_dims.to_vec();
        let last_idx = out_dims.len() - 1;
        out_dims[last_idx] = d_out;
        let out = flat_out.reshape(out_dims)?;
        Ok(out)
    }

    /// ReLoRA Permanent Weight Fusion: W = W + (alpha / r) * (B @ A), then reset adapters to 0.
    pub fn merge_and_reset(&mut self) -> Result<()> {
        let delta_w = self.lora_b.as_tensor().matmul(self.lora_a.as_tensor())?;
        let delta_w_scaled = (delta_w * self.scale)?;
        let new_base = (&self.base_weight + &delta_w_scaled)?;
        self.base_weight = new_base;

        // Reset adapters
        let device = self.base_weight.device();
        let (d_out, d_in) = self.base_weight.dims2()?;
        let std_dev = 1.0 / (self.rank as f64).sqrt();
        let a_init = (Tensor::randn(0f32, std_dev as f32, (self.rank, d_in), device)? * 0.1)?;
        let b_init = Tensor::zeros((d_out, self.rank), DType::F32, device)?;

        self.lora_a = Var::from_tensor(&a_init)?;
        self.lora_b = Var::from_tensor(&b_init)?;
        Ok(())
    }

    /// Compute delta weight matrix: ΔW = (alpha / r) * (B @ A).
    pub fn delta_weight(&self) -> Result<Tensor> {
        let delta_w = self.lora_b.as_tensor().matmul(self.lora_a.as_tensor())?;
        let scaled = (delta_w * self.scale)?;
        Ok(scaled)
    }

    /// Export adapter tensors into named map for standard HuggingFace SafeTensors persistence.
    pub fn export_tensors(&self, prefix: &str) -> Vec<(String, Tensor)> {
        vec![
            (format!("{}.lora_a.weight", prefix), self.lora_a.as_tensor().clone()),
            (format!("{}.lora_b.weight", prefix), self.lora_b.as_tensor().clone()),
        ]
    }

    /// Load adapter weights into the layer.
    pub fn load_adapter(&mut self, lora_a: &Tensor, lora_b: &Tensor) -> Result<()> {
        self.lora_a = Var::from_tensor(lora_a)?;
        self.lora_b = Var::from_tensor(lora_b)?;
        Ok(())
    }

    /// Total trainable parameters in this LoRA layer.
    pub fn trainable_params(&self) -> usize {
        self.lora_a.as_tensor().elem_count() + self.lora_b.as_tensor().elem_count()
    }
}
