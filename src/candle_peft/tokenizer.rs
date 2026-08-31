use anyhow::Result;
use candle_core::{Device, Tensor};

/// Byte-level UTF-8 Tokenizer for zero-dependency universal text processing.
#[derive(Clone, Debug)]
pub struct SimpleByteTokenizer {
    pub vocab_size: usize,
}

impl SimpleByteTokenizer {
    pub fn new() -> Self {
        Self { vocab_size: 256 }
    }

    /// Encode UTF-8 string into token IDs (0..255).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.as_bytes().iter().map(|&b| b as u32).collect()
    }

    /// Decode token IDs back to UTF-8 String (lossy).
    pub fn decode(&self, tokens: &[u32]) -> String {
        let bytes: Vec<u8> = tokens.iter().map(|&t| (t % 256) as u8).collect();
        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Convert a batch of token vectors into a 2D Tensor [batch_size, max_seq_len] padded to max_len.
    pub fn batch_to_tensor(&self, batch: &[Vec<u32>], device: &Device) -> Result<Tensor> {
        let max_len = batch.iter().map(|v| v.len()).max().unwrap_or(0).max(2);
        let mut flat = Vec::with_capacity(batch.len() * max_len);

        for seq in batch {
            for &t in seq {
                flat.push(t);
            }
            // Pad sequence with 0 (NULL / PAD byte)
            for _ in seq.len()..max_len {
                flat.push(0u32);
            }
        }

        let tensor = Tensor::from_vec(flat, (batch.len(), max_len), device)?;
        Ok(tensor)
    }
}
