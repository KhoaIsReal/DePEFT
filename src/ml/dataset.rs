use rand::Rng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

/// A single machine learning sample consisting of input features and target outputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    pub input: Vec<f32>,
    pub target: Vec<f32>,
}

/// Dataset container for fine-tuning and evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dataset {
    pub samples: Vec<Sample>,
}

impl Dataset {
    pub fn new(samples: Vec<Sample>) -> Self {
        Self { samples }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Split dataset into Training Set and Private Test Set (for Validator TEE).
    pub fn train_test_split(&self, train_ratio: f32, rng: &mut impl Rng) -> (Dataset, Dataset) {
        let valid_ratio = if train_ratio.is_finite() {
            train_ratio.clamp(0.0, 1.0)
        } else {
            0.8
        };

        let mut indices: Vec<usize> = (0..self.samples.len()).collect();
        indices.shuffle(rng);

        let train_size = ((self.samples.len() as f32) * valid_ratio).round() as usize;
        let train_size = train_size.min(self.samples.len());
        let mut train_samples = Vec::with_capacity(train_size);
        let mut test_samples = Vec::with_capacity(self.samples.len().saturating_sub(train_size));

        for (i, &idx) in indices.iter().enumerate() {
            if i < train_size {
                train_samples.push(self.samples[idx].clone());
            } else {
                test_samples.push(self.samples[idx].clone());
            }
        }

        (Dataset::new(train_samples), Dataset::new(test_samples))
    }

    /// Generate a synthetic target adaptation dataset for PEFT benchmarking.
    /// Simulates a domain shift where a base model needs adaptation to fit complex non-linear relations.
    pub fn generate_synthetic_task(
        num_samples: usize,
        input_dim: usize,
        output_dim: usize,
        domain_complexity: f32,
        rng: &mut impl Rng,
    ) -> Self {
        let domain_complexity = if domain_complexity.is_finite() {
            domain_complexity
        } else {
            1.0
        };
        let mut samples = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            let mut input = Vec::with_capacity(input_dim);
            for _ in 0..input_dim {
                input.push(rng.gen_range(-1.5..1.5));
            }

            let mut target = vec![0.0f32; output_dim];
            for j in 0..output_dim {
                let mut val = 0.0f32;
                for i in 0..input_dim {
                    let weight = ((i + j) as f32 * 0.35).sin();
                    let non_linear = (input[i] * domain_complexity).tanh();
                    val += weight * input[i] + 0.25 * non_linear;
                }
                // Add soft non-linearity
                target[j] = val + 0.1 * ((j as f32 * 1.2).cos());
            }

            samples.push(Sample { input, target });
        }

        Self { samples }
    }
}
