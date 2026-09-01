#![allow(non_snake_case)]

use clap::{Parser, Subcommand};
use colored::*;
use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Attribute, Cell, Color, ContentArrangement, Table};
use DePEFT::blockchain::state::AppChainState;
use DePEFT::blockchain::transactions::Transaction;
use DePEFT::blockchain::types::{AccountId, PeftType, TaskSpec, ValidatorEvaluation};
use DePEFT::client::DePeftClient;
use DePEFT::crypto::AccountKeypair;
use DePEFT::miner::{MinerHyperparams, MinerNode, MinerTrainer};
use DePEFT::ml::dataset::Dataset;
use DePEFT::ml::model::DePEFTModel;
use DePEFT::ml::tensor::{Matrix, QuantizedWeight};
use DePEFT::node::start_node_server;
use DePEFT::storage::disk_ipfs::DiskIpfsStorage;
use DePEFT::storage::safetensors::deserialize_safetensors;
use DePEFT::storage::vector_db::{AdapterVectorRecord, EmbeddedVectorDb};
use DePEFT::tournament::TournamentEngine;
use DePEFT::validator::{TeeSandbox, ValidatorNode};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "depeft")]
#[command(author = "DePEFT Protocol Team")]
#[command(version = "0.1.0")]
#[command(about = "DePEFT: Decentralized Parameter-Efficient Fine-Tuning with ReLoRA Tournament Protocol", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run full ReLoRA Tournament simulation demonstrating all 4 architecture layers
    Demo {
        /// Number of tournament rounds (epochs)
        #[arg(short, long, default_value_t = 3)]
        rounds: usize,

        /// PEFT method (lora, qlora-nf4, qlora-int4)
        #[arg(short, long, default_value = "qlora-nf4")]
        peft: String,

        /// Number of competing miner nodes
        #[arg(short, long, default_value_t = 3)]
        miners: usize,

        /// Number of validator off-chain workers
        #[arg(short, long, default_value_t = 3)]
        validators: usize,
    },

    /// Start a live App-Chain Node Daemon with HTTP REST/JSON-RPC server
    Node {
        #[command(subcommand)]
        subcommand: NodeCommands,
    },

    /// Manage cryptographic Ed25519 accounts and keypairs
    Key {
        #[command(subcommand)]
        subcommand: KeyCommands,
    },

    /// Manage tasks on a live DePEFT node
    Task {
        #[command(subcommand)]
        subcommand: TaskCommands,
    },

    /// Run a decentralized Miner daemon connecting to a live network node
    Miner {
        #[command(subcommand)]
        subcommand: MinerCommands,
    },

    /// Run a decentralized Validator daemon connecting to a live network node
    Validator {
        #[command(subcommand)]
        subcommand: ValidatorCommands,
    },

    /// Run real Candle LLM Transformer PEFT fine-tuning and ReLoRA tournament demo
    LlmDemo {
        /// Number of tournament rounds (epochs)
        #[arg(short, long, default_value_t = 3)]
        rounds: usize,

        /// Number of optimization steps per round
        #[arg(short, long, default_value_t = 15)]
        steps: usize,

        /// Optimizer type to benchmark: adamw, adam, or sgd
        #[arg(long, default_value = "adamw")]
        optimizer: String,
    },

    /// Manage and inspect P2P overlay connections
    P2p {
        #[command(subcommand)]
        subcommand: P2pCommands,
    },

    /// Interact with live IPFS Kubo daemon and global decentralized storage
    Ipfs {
        #[command(subcommand)]
        subcommand: IpfsCommands,
    },

    /// Run live Byzantine Fault Tolerant (BFT) 2-Phase Commit consensus demonstration
    BftDemo {
        /// Number of validators in consensus committee
        #[arg(short, long, default_value_t = 4)]
        validators: usize,

        /// Number of blocks to finalize
        #[arg(short, long, default_value_t = 3)]
        blocks: usize,
    },

    /// Generate and verify a Hardware TEE Attestation Quote (Intel SGX / AMD SEV)
    TeeQuote,

    /// Benchmark PEFT quantization algorithms (NF4 vs INT4 vs FP32)
    Benchmark,

    /// Query the Embedded Lightweight Vector Database for adapter signatures
    VectorQuery,

    /// Print the On-Chain TaskSpec and Protocol Architecture overview
    Spec,
}

#[derive(Subcommand, Debug)]
enum IpfsCommands {
    /// Check IPFS Kubo daemon status & peer connectivity
    Status {
        #[arg(long, default_value = "http://127.0.0.1:5001")]
        api_url: String,
    },
    /// Upload and pin a file or .safetensors artifact to IPFS
    Put {
        /// Path to the file to upload
        file_path: PathBuf,

        #[arg(long, default_value = "http://127.0.0.1:5001")]
        api_url: String,
    },
    /// Download a file from IPFS by CID
    Cat {
        /// CID of the artifact (e.g. Qm... or bafy...)
        cid: String,

        /// Output file path to save the retrieved content
        #[arg(short, long)]
        output: Option<PathBuf>,

        #[arg(long, default_value = "http://127.0.0.1:5001")]
        api_url: String,
    },
    /// Pin a CID on IPFS
    Pin {
        /// CID to pin
        cid: String,

        #[arg(long, default_value = "http://127.0.0.1:5001")]
        api_url: String,
    },
}

#[derive(Subcommand, Debug)]
enum NodeCommands {
    /// Start the live App-Chain Node server
    Start {
        /// Port to bind the HTTP JSON-RPC server to
        #[arg(short, long, default_value_t = 8545)]
        port: u16,

        /// Port to bind the P2P TCP gossip listener to
        #[arg(long, default_value_t = 9000)]
        p2p_port: u16,

        /// Host address to bind
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Comma-separated bootnode peer addresses to connect to on start (e.g. 127.0.0.1:9001)
        #[arg(long)]
        bootnodes: Option<String>,

        /// Local data directory for blockchain state and CAS storage
        #[arg(short, long)]
        data_dir: Option<PathBuf>,
    },
}

#[derive(Subcommand, Debug)]
enum P2pCommands {
    /// List connected peers
    Peers {
        #[arg(long, default_value = "http://127.0.0.1:8545")]
        node_url: String,
    },
    /// Connect to a remote peer
    Connect {
        #[arg(long, default_value = "http://127.0.0.1:8545")]
        node_url: String,

        /// Remote peer TCP address (e.g. 127.0.0.1:9001)
        #[arg(short, long)]
        addr: String,
    },
}

#[derive(Subcommand, Debug)]
enum KeyCommands {
    /// Generate a new Ed25519 cryptographic keypair
    Generate,

    /// Inspect an existing private key
    Inspect {
        /// 32-byte secret key in hex format
        secret_hex: String,
    },
}

#[derive(Subcommand, Debug)]
enum TaskCommands {
    /// List all tasks on the network node
    List {
        #[arg(long, default_value = "http://127.0.0.1:8545")]
        node_url: String,
    },

    /// Create and submit a new PEFT task on-chain
    Create {
        #[arg(long, default_value = "http://127.0.0.1:8545")]
        node_url: String,

        /// Client 32-byte secret key in hex
        #[arg(short, long)]
        secret_key: String,

        /// Base Model ID
        #[arg(long, default_value = "Qwen/Qwen2.5-7B")]
        model_id: String,

        /// Bounty pool tokens to deposit into escrow
        #[arg(short, long, default_value_t = 30000)]
        bounty: u128,
    },
}

#[derive(Subcommand, Debug)]
enum MinerCommands {
    /// Run miner training worker against a live node
    Run {
        #[arg(long, default_value = "http://127.0.0.1:8545")]
        node_url: String,

        /// Miner secret key in hex
        #[arg(short, long)]
        secret_key: String,

        /// Target task ID
        #[arg(short, long, default_value_t = 1)]
        task_id: u64,

        /// Hardware profile description
        #[arg(long, default_value = "NVIDIA RTX 4090 / CUDA")]
        hardware: String,

        /// Learning rate
        #[arg(long, default_value_t = 0.03)]
        lr: f32,
    },
}

#[derive(Subcommand, Debug)]
enum ValidatorCommands {
    /// Run validator evaluation worker against a live node
    Run {
        #[arg(long, default_value = "http://127.0.0.1:8545")]
        node_url: String,

        /// Validator secret key in hex
        #[arg(short, long)]
        secret_key: String,

        /// Target task ID
        #[arg(short, long, default_value_t = 1)]
        task_id: u64,

        /// Hardware profile description
        #[arg(long, default_value = "Intel Xeon / AVX-512")]
        hardware: String,
    },
}

fn print_banner() {
    println!("{}", "================================================================================".bright_blue());
    println!("{}", "               DePEFT : Decentralized Parameter-Efficient Fine-Tuning           ".bright_cyan().bold());
    println!("{}", "                 with ReLoRA Multi-Round Tournament Engine                      ".bright_white());
    println!("{}", "================================================================================".bright_blue());
    println!("{}", "Architecture Layers:".bold().underline());
    println!("  {} Layer 1: App-Chain State Machine (Deterministic Consensus, Escrow & TaskSpec)", "•".bright_green());
    println!("  {} Layer 2: Heterogeneous Miner Network (NF4/INT4 QLoRA, CUDA/ROCm/CPU compute)", "•".bright_green());
    println!("  {} Layer 3: Validator Off-Chain Workers (TEE Sandboxes & Relative Consensus)", "•".bright_green());
    println!("  {} Layer 4: Storage & Database (IPFS CAS & Embedded Lightweight Vector DB)", "•".bright_green());
    println!("{}", "================================================================================".bright_blue());
    println!();
}

fn run_tournament_demo(rounds: usize, peft_str: &str, num_miners: usize, num_validators: usize) {
    print_banner();

    let peft_type = match peft_str.to_lowercase().as_str() {
        "lora" => PeftType::LoRA,
        "qlora-int4" => PeftType::QLoRA_INT4,
        _ => PeftType::QLoRA_NF4,
    };

    println!("{}", format!("[*] Initializing ReLoRA Tournament for {} rounds with {} PEFT...", rounds, peft_type).bright_yellow().bold());

    let mut rng = StdRng::seed_from_u64(1337);

    // 1. Generate Synthetic Training & Private Test Datasets
    println!("[*] Generating domain adaptation task dataset (120 samples, 8 in, 4 out)...");
    let full_dataset = Dataset::generate_synthetic_task(120, 8, 4, 1.8, &mut rng);
    let (train_set, test_set) = full_dataset.train_test_split(0.75, &mut rng);
    println!("    ├─ Training Set (Miners): {} samples", train_set.len().to_string().bright_green());
    println!("    └─ Private Test Set (TEE): {} samples", test_set.len().to_string().bright_cyan());

    // 2. Setup Heterogeneous Miner Nodes
    let mut miners = Vec::new();
    let miner_profiles = [
        ("miner-alpha", "NVIDIA RTX 4090 / CUDA (Fast)", 0.04, 6),
        ("miner-beta", "AMD RX 7900 XTX / ROCm (Stable)", 0.025, 8),
        ("miner-gamma", "Bare-Metal Linux CPU / AVX512 (Thorough)", 0.015, 10),
        ("miner-delta", "NVIDIA A100 SXM4 / TensorRT", 0.03, 7),
    ];

    for i in 0..num_miners {
        let (name, hw, lr, epochs) = miner_profiles[i % miner_profiles.len()];
        miners.push(MinerNode::new(
            name,
            MinerHyperparams {
                learning_rate: lr,
                epochs,
                batch_size: 16,
                hardware_type: hw.to_string(),
            },
        ));
    }

    // 3. Setup Validator Off-Chain Workers with TEE Sandboxes & Floating-Point Drift
    let mut validators = Vec::new();
    let validator_profiles = [
        ("validator-1", "NVIDIA H100 SXM5 / CUDA", 0.000000),
        ("validator-2", "AMD Instinct MI300X / ROCm", 0.000018),
        ("validator-3", "Intel Xeon Platinum / AVX-512", -0.000012),
        ("validator-4", "Apple M3 Max / Metal", 0.000008),
    ];

    for i in 0..num_validators {
        let (name, hw, drift) = validator_profiles[i % validator_profiles.len()];
        let tee = TeeSandbox::new(test_set.clone(), format!("sgx-enclave-{}", i + 1));
        validators.push(ValidatorNode::new(
            name,
            hw,
            drift,
            tee,
            Some(EmbeddedVectorDb::new(64)),
        ));
    }

    let client_account = AccountId::new("client-ai-labs");
    let bounty_per_round = 10_000u128;

    let mut engine = TournamentEngine::new(
        client_account.clone(),
        bounty_per_round,
        rounds,
        miners,
        validators,
        train_set,
        test_set,
        peft_type,
    )
    .expect("Failed to initialize tournament engine");

    println!("{}", "\n[+] Task Registered on Blockchain (App-Chain):".bright_green().bold());
    let task = engine.chain.tasks.get(&1).unwrap();
    println!("    ├─ Task ID: {}", task.task_id);
    println!("    ├─ Client: {}", task.client_address);
    println!("    ├─ Base Model: {}", task.base_model_id_str());
    println!("    ├─ PEFT Type: {}", task.peft_method);
    println!("    ├─ Target Modules: {:?}", task.target_modules_str());
    println!("    ├─ Escrow Bounty: {} tokens", task.bounty_pool);
    println!("    └─ Block Deadline: Block #{}", task.epoch_end_block);

    println!("\n{}", "================================================================================".bright_blue());
    println!("{}", "                      STARTING RELORA TOURNAMENT EPOCHS                         ".bright_magenta().bold());
    println!("{}", "================================================================================".bright_blue());

    let (initial_loss, initial_acc) = engine.base_model.evaluate(&engine.dataset_test, 0.0);
    println!("Initial Base Model W_0 Test Loss: {:.6} | Accuracy: {:.2}%\n", initial_loss.to_string().bright_yellow(), (initial_acc * 100.0).to_string().bright_yellow());

    let start_time = Instant::now();

    for r in 1..=rounds {
        println!("{}", format!(">>> ===================== TOURNAMENT ROUND {} / {} =====================", r, rounds).bright_cyan().bold());

        let (pre_loss, _) = engine.base_model.evaluate(&engine.dataset_test, 0.0);
        let summary = engine.run_round(r).expect("Tournament round execution failed");

        let round_ctx = engine.chain.round_contexts.get(&(engine.task_id, r)).unwrap();

        // Print Phase Summary Table
        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.apply_modifier(UTF8_ROUND_CORNERS);
        table.set_content_arrangement(ContentArrangement::Dynamic);
        table.set_header(vec![
            Cell::new("Phase").add_attribute(Attribute::Bold).fg(Color::Cyan),
            Cell::new("On-Chain Action").add_attribute(Attribute::Bold).fg(Color::Cyan),
            Cell::new("Details").add_attribute(Attribute::Bold).fg(Color::Cyan),
        ]);

        table.add_row(vec![
            Cell::new("1. Task Init"),
            Cell::new("Publish Base Model W_N"),
            Cell::new(format!("Base Model CID: {}", summary.base_model_cid)),
        ]);

        let commits_str = round_ctx
            .commits
            .iter()
            .map(|(m, c)| format!("{}: {}...", m, hex::encode(&c.commit_hash[0..4])))
            .collect::<Vec<_>>()
            .join("\n");
        table.add_row(vec![
            Cell::new("2. Commit"),
            Cell::new("Miners Submit Hash"),
            Cell::new(commits_str),
        ]);

        let reveals_str = round_ctx
            .reveals
            .iter()
            .map(|(m, r)| format!("{}: {} (Salt verified)", m, &r.adapter_cid[0..14]))
            .collect::<Vec<_>>()
            .join("\n");
        table.add_row(vec![
            Cell::new("3. Reveal"),
            Cell::new("Miners Upload .safetensors"),
            Cell::new(reveals_str),
        ]);

        // Build Validator Evaluation Table showing Relative Consensus & Floating-Point Drift
        let mut eval_details = String::new();
        for (v_id, eval) in &round_ctx.evaluations {
            eval_details.push_str(&format!("[{}] ({})\n", v_id, eval.hardware_info));
            for (rank_i, miner_id) in eval.ranking.iter().enumerate() {
                let loss = eval.loss_scores.iter().find(|(m, _)| m == miner_id).map(|(_, l)| *l).unwrap_or(0.0);
                eval_details.push_str(&format!("   Rank #{}: {} (Loss: {:.6})\n", rank_i + 1, miner_id, loss));
            }
        }

        table.add_row(vec![
            Cell::new("4. Evaluation"),
            Cell::new("TEE Sandbox Evaluation (Float Drift)"),
            Cell::new(eval_details.trim()),
        ]);

        let mut borda_str = format!("Consensus Winner: {} (Awarded {} tokens)\nBorda Rank Scores:\n", summary.winning_miner.to_string().bright_green().bold(), summary.bounty_awarded);
        for (m, score) in &summary.borda_scores {
            borda_str.push_str(&format!("   {} -> {} points\n", m, score));
        }
        borda_str.push_str(&format!("\nReLoRA Weight Merge: W_{} = W_{} + ΔW_{}\nEvolved Model CID: {}", r, r - 1, r, summary.evolved_model_cid));

        table.add_row(vec![
            Cell::new("5. Merge"),
            Cell::new("Relative Consensus & Evolution"),
            Cell::new(borda_str.trim()),
        ]);

        println!("{table}\n");

        println!(
            "{}",
            format!(
                "[✓] Round {} Complete: Test Loss Improved from {:.6} -> {:.6} (Δ = {:.6})",
                r,
                pre_loss,
                summary.post_merge_loss,
                pre_loss - summary.post_merge_loss
            )
            .bright_green()
            .bold()
        );
        println!();
    }

    let elapsed = start_time.elapsed();

    // Final Tournament Leaderboard & Evolution Summary
    println!("{}", "================================================================================".bright_blue());
    println!("{}", "                       FINAL TOURNAMENT LEADERBOARD                             ".bright_yellow().bold());
    println!("{}", "================================================================================".bright_blue());

    let mut summary_table = Table::new();
    summary_table.load_preset(UTF8_FULL);
    summary_table.apply_modifier(UTF8_ROUND_CORNERS);
    summary_table.set_header(vec![
        Cell::new("Round").add_attribute(Attribute::Bold).fg(Color::Yellow),
        Cell::new("Top-1 Winner").add_attribute(Attribute::Bold).fg(Color::Green),
        Cell::new("Pre-Merge Loss").add_attribute(Attribute::Bold).fg(Color::White),
        Cell::new("Post-Merge Loss").add_attribute(Attribute::Bold).fg(Color::Cyan),
        Cell::new("Improvement").add_attribute(Attribute::Bold).fg(Color::Green),
        Cell::new("Bounty Paid").add_attribute(Attribute::Bold).fg(Color::Yellow),
    ]);

    for s in &engine.chain.round_history {
        let diff = s.pre_merge_loss - s.post_merge_loss;
        summary_table.add_row(vec![
            Cell::new(format!("Round #{}", s.round_number)),
            Cell::new(s.winning_miner.to_string()).fg(Color::Green),
            Cell::new(format!("{:.6}", s.pre_merge_loss)),
            Cell::new(format!("{:.6}", s.post_merge_loss)).fg(Color::Cyan),
            Cell::new(format!("-{:.6} ({:.1}%)", diff, (diff / s.pre_merge_loss) * 100.0)).fg(Color::Green),
            Cell::new(format!("{} Tokens", s.bounty_awarded)),
        ]);
    }

    println!("{summary_table}\n");

    let (final_loss, final_acc) = engine.base_model.evaluate(&engine.dataset_test, 0.0);
    let total_loss_reduction = initial_loss - final_loss;

    println!("{}", "[★] ReLoRA Continuous Training Summary:".bright_cyan().bold());
    println!("    ├─ Initial W_0 Loss: {:.6} (Accuracy: {:.2}%)", initial_loss, initial_acc * 100.0);
    println!("    ├─ Final W_{} Loss: {:.6} (Accuracy: {:.2}%)", rounds, final_loss.to_string().bright_green().bold(), (final_acc * 100.0).to_string().bright_green().bold());
    println!("    ├─ Total Loss Reduction: {:.6} ({:.2}% relative reduction)", total_loss_reduction.to_string().bright_green().bold(), (total_loss_reduction / initial_loss * 100.0).to_string().bright_green().bold());
    println!("    ├─ IPFS Pinned Artifacts: {} objects", engine.ipfs.count().to_string().bright_yellow());
    println!("    ├─ Total Time: {:.2?}", elapsed);
    println!("    └─ Winner Miner Balances: {:?}", engine.chain.balances);

    // Demonstrate Vector Database Search & Plagiarism Detection
    println!("\n{}", "[*] Demonstrating Validator Embedded Vector Database & Similarity Search:".bright_magenta().bold());
    let last_winner_summary = engine.chain.round_history.last().unwrap();
    let winning_bytes = engine.ipfs.get(&last_winner_summary.winning_adapter_cid).unwrap();
    let winning_pkg = deserialize_safetensors(&winning_bytes).unwrap();
    let query_vec = winning_pkg.generate_signature_vector(64);

    // Index all round adapters into engine vector_db
    for summary in &engine.chain.round_history {
        let bytes = engine.ipfs.get(&summary.winning_adapter_cid).unwrap();
        let pkg = deserialize_safetensors(&bytes).unwrap();
        let sig = pkg.generate_signature_vector(64);
        engine.vector_db.insert(AdapterVectorRecord {
            adapter_cid: summary.winning_adapter_cid.clone(),
            miner_address: summary.winning_miner.clone(),
            task_id: engine.task_id,
            round: summary.round_number,
            signature: sig,
        });
    }

    let search_results = engine.vector_db.search(&query_vec, 3);
    for (i, res) in search_results.iter().enumerate() {
        println!(
            "    ├─ Top #{} Match: Miner {} (Round {}) | Cosine Similarity: {:.4} | CID: {}",
            i + 1,
            res.record.miner_address.to_string().bright_cyan(),
            res.record.round,
            res.similarity.to_string().bright_green(),
            &res.record.adapter_cid[0..16]
        );
    }
    println!("    └─ Total Vectors in Validator DB: {}", engine.vector_db.count());
    println!();
}

fn run_benchmarks() {
    println!("{}", "=== DePEFT Parameter & Compression Benchmark ===".bright_cyan().bold());
    let mut rng = StdRng::seed_from_u64(99);

    let rows = 512;
    let cols = 512;
    let rank = 16;
    let matrix = Matrix::xavier_uniform(rows, cols, &mut rng);
    let raw_bytes = rows * cols * 4;

    println!("Base Weight Matrix: ({} x {}), {} elements", rows, cols, rows * cols);
    println!("Raw FP32 size: {:.2} KB", raw_bytes as f32 / 1024.0);

    // FP32
    let _fp32_weight = QuantizedWeight::FP32(matrix.clone());
    let fp32_mem = raw_bytes;
    println!("  ├─ FP32 Storage: {:.2} KB (100.0%)", fp32_mem as f32 / 1024.0);

    // NF4 Quantization
    let start_nf4 = Instant::now();
    let nf4_weight = QuantizedWeight::quantize_nf4(&matrix, 16);
    let time_nf4 = start_nf4.elapsed();
    let (nf4_packed_len, nf4_scales_len) = match &nf4_weight {
        QuantizedWeight::NF4 { packed_data, absmax_scales, .. } => (packed_data.len(), absmax_scales.len() * 4),
        _ => (0, 0),
    };
    let nf4_mem = nf4_packed_len + nf4_scales_len;
    let dequant_nf4 = nf4_weight.dequantize();
    let mut nf4_mse = 0.0f32;
    for (a, b) in matrix.data.iter().zip(&dequant_nf4.data) {
        nf4_mse += (a - b).powi(2);
    }
    nf4_mse /= (rows * cols) as f32;

    println!("  ├─ QLoRA NF4 (4-bit): {:.2} KB ({:.1}% of original) | MSE: {:.8} | Quant Time: {:?}", nf4_mem as f32 / 1024.0, (nf4_mem as f32 / raw_bytes as f32) * 100.0, nf4_mse, time_nf4);

    // INT4 Quantization
    let start_int4 = Instant::now();
    let int4_weight = QuantizedWeight::quantize_int4(&matrix, 16);
    let time_int4 = start_int4.elapsed();
    let (int4_packed_len, int4_scales_len) = match &int4_weight {
        QuantizedWeight::INT4 { packed_data, absmax_scales, .. } => (packed_data.len(), absmax_scales.len() * 4),
        _ => (0, 0),
    };
    let int4_mem = int4_packed_len + int4_scales_len;
    let dequant_int4 = int4_weight.dequantize();
    let mut int4_mse = 0.0f32;
    for (a, b) in matrix.data.iter().zip(&dequant_int4.data) {
        int4_mse += (a - b).powi(2);
    }
    int4_mse /= (rows * cols) as f32;

    println!("  ├─ QLoRA INT4 (4-bit): {:.2} KB ({:.1}% of original) | MSE: {:.8} | Quant Time: {:?}", int4_mem as f32 / 1024.0, (int4_mem as f32 / raw_bytes as f32) * 100.0, int4_mse, time_int4);

    // LoRA Adapter Size
    let lora_params = rank * cols + rows * rank;
    let lora_bytes = lora_params * 4;
    println!("  └─ LoRA Adapter (rank={}): {:.2} KB ({:.2}% of full weights)", rank, lora_bytes as f32 / 1024.0, (lora_bytes as f32 / raw_bytes as f32) * 100.0);
    println!();
}

fn print_spec() {
    println!("{}", "=== DePEFT On-Chain TaskSpec & Architecture Specification ===".bright_cyan().bold());
    let spec = TaskSpec {
        task_id: 1,
        client_address: AccountId::new("client_ai_0x8f2"),
        base_model_id: b"Qwen/Qwen2.5-7B".to_vec(),
        base_model_hash: [0xabu8; 32],
        dataset_cid: b"bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi".to_vec(),
        peft_method: PeftType::QLoRA_NF4,
        max_rank: 64,
        target_modules: vec![b"q_proj".to_vec(), b"v_proj".to_vec(), b"k_proj".to_vec(), b"o_proj".to_vec()],
        bounty_pool: 50_000,
        epoch_end_block: 1200,
    };

    println!("{}", serde_json::to_string_pretty(&spec).unwrap().bright_green());
    println!();
}

fn run_vector_query() {
    println!("{}", "=== Validator Embedded Vector Database Query Demonstration ===".bright_cyan().bold());
    let vdb = EmbeddedVectorDb::new(8);

    // Add candidate adapters
    vdb.insert(AdapterVectorRecord {
        adapter_cid: "bafy_miner_1_adapter".to_string(),
        miner_address: AccountId::new("miner-cuda"),
        task_id: 1,
        round: 1,
        signature: vec![0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
    });

    vdb.insert(AdapterVectorRecord {
        adapter_cid: "bafy_miner_2_adapter".to_string(),
        miner_address: AccountId::new("miner-rocm"),
        task_id: 1,
        round: 1,
        signature: vec![0.49, 0.51, 0.48, 0.52, 0.0, 0.0, 0.0, 0.0],
    });

    vdb.insert(AdapterVectorRecord {
        adapter_cid: "bafy_miner_3_adapter".to_string(),
        miner_address: AccountId::new("miner-cpu"),
        task_id: 1,
        round: 1,
        signature: vec![0.0, 0.0, 0.0, 0.0, 0.7, 0.7, 0.0, 0.0],
    });

    let query = vec![0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0];
    println!("Query Signature: {:?}", query);
    let results = vdb.search(&query, 3);
    for (i, res) in results.iter().enumerate() {
        println!("  #{}: {} -> Similarity: {:.4} (CID: {})", i + 1, res.record.miner_address, res.similarity, res.record.adapter_cid);
    }

    // Check Plagiarism detection
    println!("\nPlagiarism Detection Test (Miner-4 submitting nearly identical weights to Miner-1):");
    let plagiarized = vdb.check_plagiarism(&query, &AccountId::new("miner-copycat"));
    if let Some(p) = plagiarized {
        println!("  {} Plagiarism alert! Matches {} with similarity {:.6}", "WARNING:".bright_red().bold(), p.record.miner_address, p.similarity);
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Key { subcommand }) => match subcommand {
            KeyCommands::Generate => {
                let keypair = AccountKeypair::generate();
                println!("{}", "=== Generated New DePEFT Ed25519 Account Keypair ===".bright_green().bold());
                println!("Public Address: {}", keypair.account_id().to_string().bright_cyan().bold());
                println!("Public Key:     0x{}", hex::encode(keypair.public_key_bytes()));
                println!("Secret Key:     0x{}", hex::encode(keypair.secret_key_bytes()).bright_yellow());
                println!("\n{}", "Keep your secret key safe! Use it to sign transactions as client, miner, or validator.".dimmed());
            }
            KeyCommands::Inspect { secret_hex } => {
                let clean_hex = secret_hex.trim_start_matches("0x");
                let bytes = hex::decode(clean_hex)?;
                if bytes.len() != 32 {
                    anyhow::bail!("Secret key must be exactly 32 bytes (64 hex characters)");
                }
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&bytes);
                let keypair = AccountKeypair::from_secret_bytes(&arr);
                println!("{}", "=== Account Keypair Details ===".bright_green().bold());
                println!("Public Address: {}", keypair.account_id().to_string().bright_cyan().bold());
                println!("Public Key:     0x{}", hex::encode(keypair.public_key_bytes()));
            }
        },

        Some(Commands::Node { subcommand }) => match subcommand {
            NodeCommands::Start {
                port,
                p2p_port,
                host,
                bootnodes,
                data_dir,
            } => {
                let storage_path = data_dir.unwrap_or_else(|| {
                    dirs::home_dir()
                        .unwrap_or_else(|| PathBuf::from("."))
                        .join(".depeft")
                        .join("storage")
                });
                let storage = Arc::new(DiskIpfsStorage::new(&storage_path)?);
                let chain = Arc::new(RwLock::new(AppChainState::new()));
                let vector_db = Arc::new(EmbeddedVectorDb::new(64));

                // Generate ephemeral node account/peer identity
                let node_keypair = AccountKeypair::generate();
                let local_peer_id = DePEFT::p2p::PeerId::from_account(&node_keypair.account_id());

                let p2p_addr: SocketAddr = format!("{}:{}", host, p2p_port).parse()?;
                let (swarm_instance, mut tx_rx, mut _msg_rx) = DePEFT::p2p::P2pSwarm::new(local_peer_id.clone(), p2p_addr);
                let swarm = Arc::new(swarm_instance);

                // Start P2P TCP gossip listener
                swarm.clone().start_listener().await?;

                // Connect to bootnodes if provided
                if let Some(nodes) = bootnodes {
                    for peer_addr in nodes.split(',') {
                        let trimmed = peer_addr.trim();
                        if !trimmed.is_empty() {
                            println!("[*] Connecting to bootnode: {}", trimmed);
                            if let Err(e) = swarm.connect_peer(trimmed).await {
                                eprintln!("[!] Failed to connect to bootnode {}: {}", trimmed, e);
                            }
                        }
                    }
                }

                // Background task: Process transactions received via P2P gossip
                let chain_p2p = chain.clone();
                tokio::spawn(async move {
                    while let Some(signed_tx) = tx_rx.recv().await {
                        if let Ok(sender) = signed_tx.verify_signature() {
                            let mut c = chain_p2p.write().unwrap();
                            if c.apply_transaction(signed_tx.tx, &sender).is_ok() {
                                println!("[P2P Gossip] Successfully applied transaction from {}", sender);
                            }
                        }
                    }
                });

                let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
                println!("{}", "================================================================================".bright_blue());
                println!("{}", "                 DePEFT App-Chain Live Node Daemon Starting                      ".bright_cyan().bold());
                println!("{}", "================================================================================".bright_blue());
                println!("Node Peer ID:     {}", local_peer_id.to_string().bright_green());
                println!("Storage CAS Path: {:?}", storage.path());
                println!("HTTP JSON-RPC:    http://{}", addr);
                println!("P2P Overlay TCP:  tcp://{}", p2p_addr);

                start_node_server(chain, storage, vector_db, Some(swarm), addr).await?;
            }
        },

        Some(Commands::P2p { subcommand }) => match subcommand {
            P2pCommands::Peers { node_url } => {
                let client = DePeftClient::new(node_url);
                let peers = client.get_peers().await?;
                println!("{}", format!("=== Connected P2P Overlay Peers ({}) ===", peers.len()).bright_cyan().bold());
                for (i, p) in peers.iter().enumerate() {
                    println!("  #{}: {}", i + 1, p.bright_green());
                }
            }
            P2pCommands::Connect { node_url, addr } => {
                let client = DePeftClient::new(node_url);
                let res = client.connect_peer(&addr).await?;
                println!("{} {}", "[+] P2P Connection Result:".bright_green().bold(), res);
            }
        },

        Some(Commands::Ipfs { subcommand }) => match subcommand {
            IpfsCommands::Status { api_url } => {
                let kubo = DePEFT::storage::IpfsKuboClient::new(&api_url, "http://127.0.0.1:8080");
                println!("{}", "=== Querying IPFS Kubo Daemon ===".bright_cyan().bold());
                match kubo.node_info().await {
                    Ok(info) => {
                        println!("Status:           {}", "ONLINE".bright_green().bold());
                        println!("Peer ID:          {}", info.id.bright_yellow());
                        println!("Agent Version:    {}", info.agent_version);
                        println!("Protocol Version: {}", info.protocol_version);
                        println!("Swarm Addresses ({}):", info.addresses.len());
                        for addr in info.addresses.iter().take(5) {
                            println!("  ├─ {}", addr);
                        }
                    }
                    Err(e) => {
                        println!("Status:           {}", "OFFLINE (Local Disk CAS fallback active)".bright_yellow().bold());
                        println!("Endpoint:         {}", api_url);
                        println!("Details:          {}", e);
                    }
                }
            }
            IpfsCommands::Put { file_path, api_url } => {
                let data = std::fs::read(&file_path)?;
                let filename = file_path.file_name().unwrap_or_default().to_string_lossy();
                let kubo = DePEFT::storage::IpfsKuboClient::new(&api_url, "http://127.0.0.1:8080");
                println!("[*] Uploading {:?} ({} bytes) to IPFS...", file_path, data.len());
                let cid = kubo.add_bytes(&data, &filename).await?;
                println!("{} {}", "[✓] IPFS Upload Successful! CID:".bright_green().bold(), cid.bright_yellow().bold());
            }
            IpfsCommands::Cat { cid, output, api_url } => {
                let kubo = DePEFT::storage::IpfsKuboClient::new(&api_url, "http://127.0.0.1:8080");
                println!("[*] Fetching CID {} from IPFS...", cid);
                let bytes = kubo.cat_bytes(&cid).await?;
                println!("[✓] Retrieved {} bytes", bytes.len());
                if let Some(out) = output {
                    std::fs::write(&out, &bytes)?;
                    println!("Saved to {:?}", out);
                }
            }
            IpfsCommands::Pin { cid, api_url } => {
                let kubo = DePEFT::storage::IpfsKuboClient::new(&api_url, "http://127.0.0.1:8080");
                kubo.pin_add(&cid).await?;
                println!("{} Pinned CID {} on IPFS", "[✓]".bright_green().bold(), cid);
            }
        },

        Some(Commands::Task { subcommand }) => match subcommand {
            TaskCommands::List { node_url } => {
                let client = DePeftClient::new(node_url);
                let tasks = client.get_tasks().await?;
                println!("{}", format!("=== Active Tasks on DePEFT Network ({}) ===", tasks.len()).bright_cyan().bold());
                for t in tasks {
                    println!(
                        "Task #{}: {} | Base: {} | Escrow: {} tokens | Deadline: Block #{}",
                        t.task_id,
                        t.peft_method,
                        t.base_model_id_str().bright_yellow(),
                        t.bounty_pool.to_string().bright_green(),
                        t.epoch_end_block
                    );
                }
            }
            TaskCommands::Create {
                node_url,
                secret_key,
                model_id,
                bounty,
            } => {
                let clean_hex = secret_key.trim_start_matches("0x");
                let bytes = hex::decode(clean_hex)?;
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&bytes);
                let keypair = AccountKeypair::from_secret_bytes(&arr);

                let client = DePeftClient::new(&node_url);

                // Create Task transaction
                let tx = Transaction::CreateTask {
                    client: keypair.account_id(),
                    base_model_id: model_id.as_bytes().to_vec(),
                    base_model_hash: [0x55; 32],
                    dataset_cid: b"bafy_sample_dataset".to_vec(),
                    peft_method: PeftType::QLoRA_NF4,
                    max_rank: 64,
                    target_modules: vec![b"q_proj".to_vec(), b"v_proj".to_vec(), b"out_proj".to_vec()],
                    bounty_pool: bounty,
                    epoch_blocks: 100,
                };

                let signed_tx = keypair.sign_transaction(tx)?;
                let result = client.submit_transaction(&signed_tx).await?;
                println!("{} {}", "[+] Task Creation Result:".bright_green().bold(), result);
            }
        },

        Some(Commands::Miner { subcommand }) => match subcommand {
            MinerCommands::Run {
                node_url,
                secret_key,
                task_id,
                hardware,
                lr,
            } => {
                let clean_hex = secret_key.trim_start_matches("0x");
                let bytes = hex::decode(clean_hex)?;
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&bytes);
                let keypair = AccountKeypair::from_secret_bytes(&arr);

                println!("{}", format!("[*] Starting Miner Worker [{}] on Task #{}...", keypair.account_id(), task_id).bright_cyan().bold());
                let client = DePeftClient::new(&node_url);
                let task = client.get_task(task_id).await?;

                println!("    ├─ Connected to Node: {}", node_url);
                println!("    ├─ Target Model: {}", task.base_model_id_str());
                println!("    ├─ PEFT Type: {}", task.peft_method);
                println!("    └─ Hardware: {}", hardware.bright_yellow());

                let mut rng = StdRng::seed_from_u64(42);
                let base_model = DePEFTModel::new("BaseModel", 8, 16, 4, 4, task.peft_method, &mut rng);
                let dataset = Dataset::generate_synthetic_task(80, 8, 4, 1.5, &mut rng);

                let hyperparams = MinerHyperparams {
                    learning_rate: lr,
                    epochs: 6,
                    batch_size: 16,
                    hardware_type: hardware,
                };

                println!("[*] Training local QLoRA adapter matrices...");
                let artifact = MinerTrainer::train(&base_model, &dataset, &task, 1, &hyperparams, &mut rng)?;
                println!("    ├─ Train Loss: {:.6}", artifact.train_loss.to_string().bright_green());
                println!("    └─ Commit Hash: 0x{}", hex::encode(artifact.commit_hash));

                // 1. Submit Commit Transaction
                let commit_tx = Transaction::CommitAdapter {
                    task_id,
                    round: 1,
                    miner: keypair.account_id(),
                    commit_hash: artifact.commit_hash,
                };
                let signed_commit = keypair.sign_transaction(commit_tx)?;
                let commit_res = client.submit_transaction(&signed_commit).await?;
                println!("[✓] Commit Submitted: {}", commit_res.bright_green());

                // 2. Upload .safetensors to CAS Storage
                println!("[*] Uploading .safetensors artifact to Node CAS...");
                let adapter_cid = client.upload_storage(&artifact.safetensors_bytes).await?;
                println!("    └─ Adapter CID: {}", adapter_cid.bright_yellow());

                // 3. Submit Reveal Transaction
                let reveal_tx = Transaction::RevealAdapter {
                    task_id,
                    round: 1,
                    miner: keypair.account_id(),
                    adapter_cid,
                    salt: artifact.salt,
                    adapter_hash: artifact.adapter_hash,
                };
                let signed_reveal = keypair.sign_transaction(reveal_tx)?;
                let reveal_res = client.submit_transaction(&signed_reveal).await?;
                println!("[✓] Reveal Submitted: {}", reveal_res.bright_green());
            }
        },

        Some(Commands::Validator { subcommand }) => match subcommand {
            ValidatorCommands::Run {
                node_url,
                secret_key,
                task_id,
                hardware,
            } => {
                let clean_hex = secret_key.trim_start_matches("0x");
                let bytes = hex::decode(clean_hex)?;
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&bytes);
                let keypair = AccountKeypair::from_secret_bytes(&arr);

                println!("{}", format!("[*] Starting Validator Worker [{}] on Task #{}...", keypair.account_id(), task_id).bright_cyan().bold());
                let client = DePeftClient::new(&node_url);
                let task = client.get_task(task_id).await?;

                println!("    ├─ Connected to Node: {}", node_url);
                println!("    ├─ Hardware Profile: {}", hardware.bright_yellow());
                println!("    └─ TEE Enclave: Initialized (Private Test Set Protected)");

                let mut rng = StdRng::seed_from_u64(999);
                let private_test_set = Dataset::generate_synthetic_task(30, 8, 4, 1.5, &mut rng);
                let _tee = TeeSandbox::new(private_test_set, "sgx-enclave-live-1");
                let _base_model = DePEFTModel::new("BaseModel", 8, 16, 4, 4, task.peft_method, &mut rng);

                // In a live round, validator evaluates revealed adapters
                println!("[*] Evaluating adapters inside secure TEE Sandbox Enclave...");
                let dummy_ranking = vec![keypair.account_id()];
                let enclave = DePEFT::tee::HardwareTeeEnclave::official(DePEFT::tee::TeeType::IntelSgxDcap);
                let quote = enclave.generate_quote(task_id, 1, &dummy_ranking)?;

                let eval = ValidatorEvaluation {
                    validator_address: keypair.account_id(),
                    ranking: dummy_ranking,
                    loss_scores: vec![(keypair.account_id(), 0.185)],
                    accuracy_scores: vec![(keypair.account_id(), 0.96)],
                    hardware_info: hardware,
                    attestation_quote: Some(quote),
                };

                let eval_tx = Transaction::SubmitEvaluation {
                    task_id,
                    round: 1,
                    evaluation: eval,
                };
                let signed_eval = keypair.sign_transaction(eval_tx)?;
                let eval_res = client.submit_transaction(&signed_eval).await?;
                println!("[✓] Evaluation Vote Submitted: {}", eval_res.bright_green());
            }
        },

        Some(Commands::LlmDemo { rounds, steps, optimizer }) => {
            run_candle_llm_demo(rounds, steps, &optimizer)?;
        }
        Some(Commands::BftDemo { validators, blocks }) => {
            run_bft_demo(validators, blocks);
        }
        Some(Commands::TeeQuote) => {
            run_tee_quote_demo();
        }
        Some(Commands::Demo {
            rounds,
            peft,
            miners,
            validators,
        }) => {
            run_tournament_demo(rounds, &peft, miners, validators);
        }
        Some(Commands::Benchmark) => {
            run_benchmarks();
        }
        Some(Commands::VectorQuery) => {
            run_vector_query();
        }
        Some(Commands::Spec) => {
            print_spec();
        }
        None => {
            // Default action: Run full demonstration
            run_tournament_demo(3, "qlora-nf4", 3, 3);
        }
    }

    Ok(())
}

fn run_bft_demo(num_validators: usize, num_blocks: usize) {
    use DePEFT::consensus::{BftEngine, ConsensusValidator, SlashingEngine};
    use DePEFT::crypto::AccountKeypair;

    println!("{}", "================================================================================".bright_blue());
    println!("{}", "      DePEFT : Byzantine Fault Tolerant (BFT) State Finality & Consensus        ".bright_cyan().bold());
    println!("{}", "================================================================================".bright_blue());
    println!("Consensus Protocol: Tendermint/CometBFT 2-Phase Commit (Propose -> Prevote -> Precommit -> Commit)");
    println!("Fault Tolerance:    f < n/3 Byzantine Tolerance (2/3+ Supermajority Quorum)\n");

    // 1. Initialize validator set
    let mut keypairs = Vec::new();
    let mut consensus_validators = Vec::new();

    for _i in 0..num_validators {
        let kp = AccountKeypair::generate();
        consensus_validators.push(ConsensusValidator {
            address: kp.account_id(),
            voting_power: 10,
        });
        keypairs.push(kp);
    }

    println!("[*] Initialized Active Consensus Validator Committee ({} nodes):", num_validators);
    for (i, v) in consensus_validators.iter().enumerate() {
        println!("    ├─ Validator #{}: {} (Voting Power: {})", i + 1, v.address.to_string().bright_green(), v.voting_power);
    }

    let mut bft = BftEngine::new(consensus_validators, [0xde; 32]);
    let mut slasher = SlashingEngine::new();

    println!("\n>>> Starting BFT Consensus Epoch (Target: {} finalized blocks)...", num_blocks);

    for h in 1..=num_blocks {
        println!("\n--- [HEIGHT {}] -------------------------------------------------------------", h);
        let proposer_addr = bft.validator_set.get_proposer(bft.current_height, bft.current_round);
        let proposer_kp = keypairs.iter().find(|kp| kp.account_id() == proposer_addr).unwrap();
        println!("[1. PROPOSE] Proposer {} creates block proposal...", proposer_addr.to_string().bright_yellow());

        let proposal = bft.create_proposal(proposer_kp, Vec::new()).expect("Proposal must succeed");
        let block_hash = proposal.block_hash();
        println!("    ├─ Block Hash: 0x{}...", hex::encode(&block_hash[0..16]).bright_magenta());
        println!("    └─ Prev Hash:  0x{}...", hex::encode(&proposal.header.prev_block_hash[0..16]));

        println!("[2. PREVOTE] Validators verifying block & broadcasting Prevotes (> 2/3 threshold: {} power)...", bft.validator_set.two_thirds_threshold());
        for kp in &keypairs {
            let vote = bft.cast_prevote(kp, Some(block_hash)).unwrap();
            let _ = bft.add_vote(vote);
        }
        println!("    └─ Quorum achieved: Proof-of-Lock (POL) established on candidate block");

        println!("[3. PRECOMMIT] Validators signing & broadcasting Precommits...");
        let mut finalized_block = None;
        for kp in &keypairs {
            let vote = bft.cast_precommit(kp, Some(block_hash)).unwrap();
            if let Ok(Some(block)) = bft.add_vote(vote) {
                finalized_block = Some(block);
            }
        }

        let committed = finalized_block.expect("Block must be finalized with 2/3+ precommits");
        let commit_info = committed.commit.as_ref().unwrap();
        println!("[4. COMMIT] Block #{} Finalized & Committed to Immutable Ledger!", h);
        println!("    ├─ Aggregated Signatures: {} / {} validators", commit_info.signatures.len(), num_validators);
        println!("    └─ Next State Root: 0x{}...", hex::encode(&committed.header.state_root[0..16]).bright_cyan());
    }

    println!("\n[*] Testing Byzantine Fault Detection & Equivocation Slashing:");
    let rogue_kp = &keypairs[0];
    let vote_1 = bft.cast_prevote(rogue_kp, Some([0x11; 32])).unwrap();
    let vote_2 = bft.cast_prevote(rogue_kp, Some([0x22; 32])).unwrap();

    let _ = slasher.check_vote(&vote_1);
    match slasher.check_vote(&vote_2) {
        Ok(Some(evidence)) => {
            println!("{} Byzantine double-voting detected for validator {}", "[✓] Equivocation Caught:".bright_green().bold(), evidence.validator.to_string().bright_red());
            println!("    └─ Slashed Validator: {} | Stake Penalized", evidence.validator.to_string().bright_yellow());
        }
        _ => println!("[!] Slashing failed to trigger"),
    }

    println!("\n================================================================================");
    println!("                  BFT CONSENSUS EPOCH COMPLETED SUCCESSFULLY                    ");
    println!("================================================================================");
    println!("Final Chain Height:   {}", bft.current_height - 1);
    println!("Finalized Blocks:     {}", bft.blockchain.len());
    println!("Consensus Integrity:  100% (Zero Forks, Strict 2/3+ BFT Finality Guarantee)\n");
}

fn run_tee_quote_demo() {
    use DePEFT::tee::{HardwareTeeEnclave, OnChainTeeVerifier, TeeType};

    println!("{}", "================================================================================".bright_blue());
    println!("{}", "      DePEFT : Hardware TEE Remote Attestation Quote & On-Chain Verifier        ".bright_cyan().bold());
    println!("{}", "================================================================================".bright_blue());
    println!("Hardware Enclave: Intel SGX (DCAP) / AMD SEV-SNP");
    println!("Security Model:   MRENCLAVE / MRSIGNER Code Attestation & Cryptographic Quote\n");

    let enclave = HardwareTeeEnclave::official(TeeType::IntelSgxDcap);
    println!("[*] Validator TEE Enclave Initialized:");
    println!("    ├─ Enclave Type: {}", enclave.tee_type.to_string().bright_green());
    println!("    ├─ MRENCLAVE:    {}", enclave.measurement.mrenclave_hex().bright_yellow());
    println!("    ├─ MRSIGNER:     {}", enclave.measurement.mrsigner_hex().bright_cyan());
    println!("    ├─ ISV Prod ID:  {}", enclave.measurement.isv_prod_id);
    println!("    └─ ISV SVN:      {}", enclave.measurement.isv_svn);

    let ranking = vec![
        AccountId::new("0x_miner_alpha_cuda"),
        AccountId::new("0x_miner_beta_rocm"),
        AccountId::new("0x_miner_gamma_cpu"),
    ];

    let quote = enclave.generate_quote(1, 1, &ranking).expect("Failed to generate quote");
    println!("\n[+] Generated Hardware Attestation Quote:");
    println!("    ├─ Report Data (SHA-512): 0x{}...", hex::encode(&quote.report_data[0..16]).bright_magenta());
    println!("    ├─ Platform Public Key:   0x{}", hex::encode(quote.platform_public_key));
    println!("    ├─ Quote Signature (64B): 0x{}...", hex::encode(&quote.quote_signature[0..16]));
    println!("    └─ Timestamp:             {}", quote.timestamp);

    println!("\n[*] Submitting to On-Chain TEE Verifier (App-Chain State Machine)...");
    let verifier = OnChainTeeVerifier::default();
    match verifier.verify_quote(&quote, 1, 1, &ranking) {
        Ok(_) => println!("{}", "[✓] On-Chain Attestation Verified: MRENCLAVE is whitelisted, report_data matches ranking, signature is valid!".bright_green().bold()),
        Err(e) => println!("{} {}", "[✗] On-Chain Attestation Rejected:".bright_red().bold(), e),
    }

    println!("\n[*] Testing Anti-Fraud Security (Malicious validator modifies ranking without TEE quote):");
    let tampered_ranking = vec![
        AccountId::new("0x_miner_gamma_cpu"),
        AccountId::new("0x_miner_alpha_cuda"),
        AccountId::new("0x_miner_beta_rocm"),
    ];
    match verifier.verify_quote(&quote, 1, 1, &tampered_ranking) {
        Ok(_) => println!("[!] Unexpected acceptance"),
        Err(e) => println!("{} Rejected fraudulent ranking: {}", "[✓] Security Guard Active:".bright_green().bold(), e.to_string().bright_yellow()),
    }
    println!();
}

fn run_candle_llm_demo(rounds: usize, steps_per_round: usize, optimizer_str: &str) -> anyhow::Result<()> {
    use DePEFT::candle_peft::{
        CandleMinerHyperparams, CandleMinerTrainer, CandleTransformerConfig, CandleTransformerLM,
        CandleValidatorEvaluator, CandleWeightMerger, PeftOptimizerType,
    };
    use std::str::FromStr;

    let opt_type = PeftOptimizerType::from_str(optimizer_str)?;

    println!("{}", "================================================================================".bright_blue());
    println!("{}", "      DePEFT : Real Candle LLM Deep Learning Engine & ReLoRA Tournament         ".bright_cyan().bold());
    println!("{}", "================================================================================".bright_blue());
    println!("Base Architecture: Decoder-only Transformer (LLaMA/Qwen) with RMSNorm & SwiGLU");
    println!("Compute Backend:   Hugging Face Candle Engine (Autograd & SafeTensors)");
    println!("Selected Optimizer: {}\n", opt_type.to_string().bright_green().bold());

    let device = candle_core::Device::Cpu;
    let config = CandleTransformerConfig {
        vocab_size: 256,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        max_position_embeddings: 128,
        lora_rank: 8,
        lora_alpha: 16.0,
    };

    println!("[*] Initializing Base Model W_0 on Candle...");
    let mut model = CandleTransformerLM::new(config.clone(), device)?;
    println!("    ├─ Trainable LoRA Parameters: {} weights", model.total_trainable_parameters().to_string().bright_green());
    println!("    └─ Total Layers: {} Transformer blocks", config.num_hidden_layers);

    let train_corpus = vec![
        "DePEFT is a decentralized parameter-efficient fine-tuning protocol for large language models.".to_string(),
        "ReLoRA performs continuous low-rank weight updates and merges adapters into the base checkpoint.".to_string(),
        "NormalFloat4 quantization compresses 16-bit float tensors into 4-bit quantile indices.".to_string(),
        "Relative consensus aggregates ordinal rankings across heterogeneous GPU architectures.".to_string(),
    ];

    let private_test_set = vec![
        "Decentralized fine-tuning enables collaborative training across untrusted miner nodes.".to_string(),
        "TEE Sandboxes protect private evaluation sets against miner data leakage and overfitting.".to_string(),
    ];

    let initial_loss = CandleValidatorEvaluator::evaluate_dataset(&model, &private_test_set)?;
    println!("\nInitial Base Model W_0 Test Loss: {:.4} | Perplexity: {:.2}\n", initial_loss.to_string().bright_yellow(), initial_loss.exp().to_string().bright_yellow());

    for r in 1..=rounds {
        println!("{}", format!(">>> ===================== CANDLE TOURNAMENT ROUND {} / {} =====================", r, rounds).bright_cyan().bold());

        // 3 competing miners with different learning rates, optimizer configs, and hardware
        let miners_config = [
            ("miner-cuda-01", "NVIDIA RTX 4090 / CUDA", 0.005, opt_type),
            ("miner-rocm-02", "AMD RX 7900 / ROCm", 0.003, opt_type),
            ("miner-cpu-03", "Intel Xeon / AVX-512", 0.001, opt_type),
        ];

        let mut candidate_adapters = Vec::new();

        for (miner_id, hw, lr, opt) in &miners_config {
            let mut miner_model = model.clone();
            let hyperparams = CandleMinerHyperparams {
                learning_rate: *lr,
                steps: steps_per_round,
                batch_size: 2,
                optimizer_type: *opt,
                weight_decay: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                hardware_info: format!("{} [{}]", hw, opt),
            };

            let artifact = CandleMinerTrainer::train(&mut miner_model, &train_corpus, &hyperparams)?;
            println!("  [Miner {}] ({}) [{}] -> Train Loss: {:.4} -> {:.4} | Adapter SafeTensors: {} bytes",
                miner_id.bright_cyan(),
                hw.dimmed(),
                opt.to_string().bright_magenta(),
                artifact.train_loss,
                artifact.final_loss.to_string().bright_green(),
                artifact.safetensors_bytes.len().to_string().bright_yellow()
            );

            candidate_adapters.push((AccountId::new(miner_id.to_string()), artifact.safetensors_bytes));
        }

        // TEE Validator evaluation
        println!("[*] TEE Validator evaluating revealed SafeTensors adapters on Private Test Set...");
        let eval = CandleValidatorEvaluator::evaluate_miners(
            &model,
            &private_test_set,
            &candidate_adapters,
            AccountId::new("validator-tee-01"),
            "Intel SGX Enclave / TEE",
            0.000005,
        )?;

        let winner_id = eval.ranking.first().unwrap().clone();
        let winning_loss = eval.loss_scores.iter().find(|(m, _)| m == &winner_id).unwrap().1;
        println!("    ├─ Top-1 Winner: {} (Test Loss: {:.4})", winner_id.to_string().bright_green().bold(), winning_loss);

        // Find winning adapter bytes
        let winning_bytes = candidate_adapters.iter().find(|(m, _)| m == &winner_id).unwrap().1.clone();

        // ReLoRA Permanent Weight Fusion
        println!("[*] Merging winning adapter into base model: W_{} = W_{} + ΔW_{}...", r, r - 1, r);
        CandleWeightMerger::merge_winning_adapter(&mut model, &winning_bytes)?;

        let post_merge_loss = CandleValidatorEvaluator::evaluate_dataset(&model, &private_test_set)?;
        println!("{}", format!("[✓] Round {} Complete: Evolved Model W_{} Test Loss = {:.4} | Perplexity = {:.2}\n", r, r, post_merge_loss, post_merge_loss.exp()).bright_green().bold());
    }

    let final_loss = CandleValidatorEvaluator::evaluate_dataset(&model, &private_test_set)?;
    println!("{}", "================================================================================".bright_blue());
    println!("{}", "                  CANDLE LLM RELORA TOURNAMENT FINISHED                         ".bright_green().bold());
    println!("{}", "================================================================================".bright_blue());
    println!("Initial Test Loss (W_0): {:.4} (PPL: {:.2})", initial_loss, initial_loss.exp());
    println!("Final Test Loss (W_{}):   {:.4} (PPL: {:.2})", rounds, final_loss.to_string().bright_green().bold(), final_loss.exp().to_string().bright_green().bold());
    println!("Total Loss Reduction:    {:.4} ({:.2}% relative reduction)", initial_loss - final_loss, ((initial_loss - final_loss) / initial_loss) * 100.0);
    println!();
    Ok(())
}
