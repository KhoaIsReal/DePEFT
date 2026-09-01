use crate::blockchain::types::PeftType;
use crate::ml::lora::ModuleAdapter;
use crate::ml::model::AdapterPackage;
use crate::ml::tensor::Matrix;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
struct TensorMetadata {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

#[derive(Debug, Serialize, Deserialize)]
struct SafeTensorsHeader {
    #[serde(rename = "__metadata__")]
    metadata: HashMap<String, String>,
    #[serde(flatten)]
    tensors: HashMap<String, TensorMetadata>,
}

/// Encode an AdapterPackage into .safetensors binary format.
pub fn serialize_safetensors(package: &AdapterPackage) -> Result<Vec<u8>> {
    let mut header_metadata = HashMap::new();
    header_metadata.insert("model_id".to_string(), package.model_id.clone());
    header_metadata.insert("round".to_string(), package.round.to_string());
    header_metadata.insert("peft_type".to_string(), format!("{:?}", package.peft_type));

    let mut tensors_meta = HashMap::new();
    let mut binary_buffer = Vec::new();

    for (module_name, adapter) in &package.modules {
        // Save LoRA A
        let start_a = binary_buffer.len();
        for &val in &adapter.lora_a.data {
            binary_buffer.extend_from_slice(&val.to_le_bytes());
        }
        let end_a = binary_buffer.len();
        tensors_meta.insert(
            format!("{}.lora_a", module_name),
            TensorMetadata {
                dtype: "F32".to_string(),
                shape: vec![adapter.lora_a.rows, adapter.lora_a.cols],
                data_offsets: [start_a, end_a],
            },
        );

        // Save LoRA B
        let start_b = binary_buffer.len();
        for &val in &adapter.lora_b.data {
            binary_buffer.extend_from_slice(&val.to_le_bytes());
        }
        let end_b = binary_buffer.len();
        tensors_meta.insert(
            format!("{}.lora_b", module_name),
            TensorMetadata {
                dtype: "F32".to_string(),
                shape: vec![adapter.lora_b.rows, adapter.lora_b.cols],
                data_offsets: [start_b, end_b],
            },
        );

        // Store hyperparams in metadata
        header_metadata.insert(format!("{}.rank", module_name), adapter.rank.to_string());
        header_metadata.insert(format!("{}.alpha", module_name), adapter.alpha.to_string());
    }

    let header_struct = SafeTensorsHeader {
        metadata: header_metadata,
        tensors: tensors_meta,
    };

    let header_json = serde_json::to_string(&header_struct)?;
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut output = Vec::with_capacity(8 + header_bytes.len() + binary_buffer.len());
    output.extend_from_slice(&header_len.to_le_bytes());
    output.extend_from_slice(header_bytes);
    output.extend_from_slice(&binary_buffer);

    Ok(output)
}

/// Decode .safetensors binary format into an AdapterPackage safely.
pub fn deserialize_safetensors(bytes: &[u8]) -> Result<AdapterPackage> {
    if bytes.len() < 8 {
        bail!("File too small to be a valid safetensors file");
    }

    const MAX_HEADER_LEN: usize = 16 * 1024 * 1024; // 16 MB maximum header JSON
    let header_len = u64::from_le_bytes(bytes[0..8].try_into()?) as usize;
    if header_len > MAX_HEADER_LEN || bytes.len() < 8 + header_len {
        bail!("Corrupted safetensors: invalid header length {}", header_len);
    }

    let header_json_bytes = &bytes[8..8 + header_len];
    let header: SafeTensorsHeader = serde_json::from_slice(header_json_bytes)
        .context("Failed to parse safetensors JSON header")?;

    let binary_data = &bytes[8 + header_len..];

    let model_id = header
        .metadata
        .get("model_id")
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());
    let round = header
        .metadata
        .get("round")
        .and_then(|r| r.parse().ok())
        .unwrap_or(0);
    let peft_type = match header.metadata.get("peft_type").map(|s| s.as_str()) {
        Some("QLoRA_NF4") => PeftType::QLoRA_NF4,
        Some("QLoRA_INT4") => PeftType::QLoRA_INT4,
        _ => PeftType::LoRA,
    };

    let mut modules = HashMap::new();

    // Group tensors by module prefix
    for key in header.tensors.keys() {
        if let Some(module_name) = key.strip_suffix(".lora_a") {
            let key_a = format!("{}.lora_a", module_name);
            let key_b = format!("{}.lora_b", module_name);

            let meta_a = match header.tensors.get(&key_a) {
                Some(m) if m.shape.len() >= 2 => m,
                _ => continue,
            };
            let meta_b = match header.tensors.get(&key_b) {
                Some(m) if m.shape.len() >= 2 => m,
                _ => continue,
            };

            // Bounds check for tensor A
            let start_a = meta_a.data_offsets[0];
            let end_a = meta_a.data_offsets[1];
            if start_a > end_a || end_a > binary_data.len() {
                bail!("Corrupted safetensors: data offsets out of bounds for {}", key_a);
            }
            let slice_a = &binary_data[start_a..end_a];
            let mut data_a = Vec::with_capacity(slice_a.len() / 4);
            for chunk in slice_a.chunks_exact(4) {
                data_a.push(f32::from_le_bytes(chunk.try_into()?));
            }
            if data_a.len() != meta_a.shape[0] * meta_a.shape[1] {
                bail!("Tensor A data length does not match specified shape");
            }
            let lora_a = Matrix::new(meta_a.shape[0], meta_a.shape[1], data_a);

            // Bounds check for tensor B
            let start_b = meta_b.data_offsets[0];
            let end_b = meta_b.data_offsets[1];
            if start_b > end_b || end_b > binary_data.len() {
                bail!("Corrupted safetensors: data offsets out of bounds for {}", key_b);
            }
            let slice_b = &binary_data[start_b..end_b];
            let mut data_b = Vec::with_capacity(slice_b.len() / 4);
            for chunk in slice_b.chunks_exact(4) {
                data_b.push(f32::from_le_bytes(chunk.try_into()?));
            }
            if data_b.len() != meta_b.shape[0] * meta_b.shape[1] {
                bail!("Tensor B data length does not match specified shape");
            }
            let lora_b = Matrix::new(meta_b.shape[0], meta_b.shape[1], data_b);

            let rank = header
                .metadata
                .get(&format!("{}.rank", module_name))
                .and_then(|r| r.parse().ok())
                .unwrap_or(meta_a.shape[0]);
            let alpha = header
                .metadata
                .get(&format!("{}.alpha", module_name))
                .and_then(|a| a.parse().ok())
                .unwrap_or(16.0);

            modules.insert(
                module_name.to_string(),
                ModuleAdapter {
                    module_name: module_name.to_string(),
                    rank,
                    alpha,
                    lora_a,
                    lora_b,
                },
            );
        }
    }

    Ok(AdapterPackage {
        model_id,
        round,
        peft_type,
        modules,
    })
}
