use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// 2D Matrix for neural network and PEFT weight computations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(rows * cols, data.len(), "Matrix dimension mismatch");
        Self { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn random_normal(
        rows: usize,
        cols: usize,
        mean: f32,
        std_dev: f32,
        rng: &mut impl Rng,
    ) -> Self {
        let normal = Normal::new(mean, std_dev).unwrap();
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..(rows * cols) {
            data.push(normal.sample(rng));
        }
        Self { rows, cols, data }
    }

    pub fn xavier_uniform(rows: usize, cols: usize, rng: &mut impl Rng) -> Self {
        let limit = (6.0 / (rows + cols) as f32).sqrt();
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..(rows * cols) {
            data.push(rng.gen_range(-limit..limit));
        }
        Self { rows, cols, data }
    }

    #[inline(always)]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    #[inline(always)]
    pub fn set(&mut self, r: usize, c: usize, val: f32) {
        self.data[r * self.cols + c] = val;
    }

    /// Matrix multiplication: C = A * B.
    /// A: (M, K), B: (K, N) -> C: (M, N)
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.cols, other.rows,
            "Matmul dimension mismatch: ({}, {}) x ({}, {})",
            self.rows, self.cols, other.rows, other.cols
        );
        let m = self.rows;
        let k = self.cols;
        let n = other.cols;
        let mut result = vec![0.0; m * n];

        for i in 0..m {
            let row_offset = i * k;
            let out_offset = i * n;
            for p in 0..k {
                let a_val = self.data[row_offset + p];
                let b_offset = p * n;
                for j in 0..n {
                    result[out_offset + j] += a_val * other.data[b_offset + j];
                }
            }
        }

        Matrix::new(m, n, result)
    }

    /// Transpose of matrix: (rows, cols) -> (cols, rows).
    pub fn transpose(&self) -> Matrix {
        let mut result = vec![0.0; self.rows * self.cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                result[c * self.rows + r] = self.get(r, c);
            }
        }
        Matrix::new(self.cols, self.rows, result)
    }

    /// Element-wise addition: C = self + other.
    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        Matrix::new(self.rows, self.cols, data)
    }

    /// Element-wise addition in place: self += other.
    pub fn add_assign(&mut self, other: &Matrix) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += *b;
        }
    }

    /// Scalar multiplication: C = self * scalar.
    pub fn scale(&self, scalar: f32) -> Matrix {
        let data: Vec<f32> = self.data.iter().map(|&x| x * scalar).collect();
        Matrix::new(self.rows, self.cols, data)
    }

    /// Frobenius norm of the matrix.
    pub fn frobenius_norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

/// NormalFloat4 (NF4) codebook: 16 optimal quantiles for standard normal distribution $\mathcal{N}(0, 1)$.
/// As defined in the QLoRA paper (Dettmers et al., 2023).
pub const NF4_CODEBOOK: [f32; 16] = [
    -1.0000000, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, -0.0910500, 0.0000000,
    0.0795803, 0.1609302, 0.2461123, 0.3379152, 0.4407098, 0.5626170, 0.7229568, 1.0000000,
];

/// Quantized representation of a base model weight matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizedWeight {
    FP32(Matrix),
    NF4 {
        rows: usize,
        cols: usize,
        /// 4-bit nibbles packed (2 elements per u8 byte).
        packed_data: Vec<u8>,
        /// Blockwise absolute maximum scaling factors.
        absmax_scales: Vec<f32>,
        block_size: usize,
    },
    INT4 {
        rows: usize,
        cols: usize,
        packed_data: Vec<u8>,
        absmax_scales: Vec<f32>,
        block_size: usize,
    },
}

impl QuantizedWeight {
    /// Quantize a full precision FP32 matrix to NF4.
    pub fn quantize_nf4(matrix: &Matrix, block_size: usize) -> Self {
        let block_size = block_size.max(1);
        let rows = matrix.rows;
        let cols = matrix.cols;
        let total = rows * cols;
        let num_blocks = total.div_ceil(block_size);
        let mut absmax_scales = Vec::with_capacity(num_blocks);
        let mut quantized_indices = Vec::with_capacity(total);

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(total);
            let slice = &matrix.data[start..end];

            // Compute block absolute max
            let mut absmax = 1e-8f32;
            for &val in slice {
                if val.abs() > absmax {
                    absmax = val.abs();
                }
            }
            absmax_scales.push(absmax);

            // Quantize each value to nearest NF4 codebook index
            for &val in slice {
                let normalized = val / absmax;
                let mut best_idx = 0;
                let mut min_diff = f32::MAX;
                for (idx, &code) in NF4_CODEBOOK.iter().enumerate() {
                    let diff = (normalized - code).abs();
                    if diff < min_diff {
                        min_diff = diff;
                        best_idx = idx as u8;
                    }
                }
                quantized_indices.push(best_idx);
            }
        }

        // Pack two 4-bit nibbles into one u8
        let mut packed_data = Vec::with_capacity(total.div_ceil(2));
        for chunk in quantized_indices.chunks(2) {
            let high = chunk[0] & 0x0F;
            let low = if chunk.len() > 1 { chunk[1] & 0x0F } else { 0 };
            packed_data.push((high << 4) | low);
        }

        Self::NF4 {
            rows,
            cols,
            packed_data,
            absmax_scales,
            block_size,
        }
    }

    /// Quantize a full precision FP32 matrix to uniform INT4.
    pub fn quantize_int4(matrix: &Matrix, block_size: usize) -> Self {
        let block_size = block_size.max(1);
        let rows = matrix.rows;
        let cols = matrix.cols;
        let total = rows * cols;
        let num_blocks = total.div_ceil(block_size);
        let mut absmax_scales = Vec::with_capacity(num_blocks);
        let mut quantized_indices = Vec::with_capacity(total);

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(total);
            let slice = &matrix.data[start..end];

            let mut absmax = 1e-8f32;
            for &val in slice {
                if val.abs() > absmax {
                    absmax = val.abs();
                }
            }
            absmax_scales.push(absmax);

            for &val in slice {
                // Map [-absmax, absmax] to [-7, 7] + 8 -> [1, 15], 0 is reserved / clamping
                let norm = (val / absmax).clamp(-1.0, 1.0);
                let int_val = (norm * 7.0).round() as i8;
                let u4_val = (int_val + 8) as u8;
                quantized_indices.push(u4_val);
            }
        }

        let mut packed_data = Vec::with_capacity(total.div_ceil(2));
        for chunk in quantized_indices.chunks(2) {
            let high = chunk[0] & 0x0F;
            let low = if chunk.len() > 1 { chunk[1] & 0x0F } else { 0 };
            packed_data.push((high << 4) | low);
        }

        Self::INT4 {
            rows,
            cols,
            packed_data,
            absmax_scales,
            block_size,
        }
    }

    /// Dequantize back to full precision FP32 matrix for forward evaluation / merge.
    pub fn dequantize(&self) -> Matrix {
        match self {
            Self::FP32(m) => m.clone(),
            Self::NF4 {
                rows,
                cols,
                packed_data,
                absmax_scales,
                block_size,
            } => {
                let block_size = (*block_size).max(1);
                let total = rows * cols;
                let mut data = Vec::with_capacity(total);

                for (byte_idx, &byte) in packed_data.iter().enumerate() {
                    let high = (byte >> 4) & 0x0F;
                    let low = byte & 0x0F;

                    let idx0 = byte_idx * 2;
                    if idx0 < total {
                        let block = idx0 / block_size;
                        let scale = absmax_scales.get(block).copied().unwrap_or(1.0);
                        data.push(NF4_CODEBOOK[high as usize] * scale);
                    }

                    let idx1 = idx0 + 1;
                    if idx1 < total {
                        let block = idx1 / block_size;
                        let scale = absmax_scales.get(block).copied().unwrap_or(1.0);
                        data.push(NF4_CODEBOOK[low as usize] * scale);
                    }
                }

                Matrix::new(*rows, *cols, data)
            }
            Self::INT4 {
                rows,
                cols,
                packed_data,
                absmax_scales,
                block_size,
            } => {
                let block_size = (*block_size).max(1);
                let total = rows * cols;
                let mut data = Vec::with_capacity(total);

                for (byte_idx, &byte) in packed_data.iter().enumerate() {
                    let high = (byte >> 4) & 0x0F;
                    let low = byte & 0x0F;

                    let idx0 = byte_idx * 2;
                    if idx0 < total {
                        let block = idx0 / block_size;
                        let scale = absmax_scales.get(block).copied().unwrap_or(1.0);
                        let int_val = (high as i8) - 8;
                        data.push((int_val as f32 / 7.0) * scale);
                    }

                    let idx1 = idx0 + 1;
                    if idx1 < total {
                        let block = idx1 / block_size;
                        let scale = absmax_scales.get(block).copied().unwrap_or(1.0);
                        let int_val = (low as i8) - 8;
                        data.push((int_val as f32 / 7.0) * scale);
                    }
                }

                Matrix::new(*rows, *cols, data)
            }
        }
    }
}
