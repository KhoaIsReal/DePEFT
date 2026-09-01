use DePEFT::blockchain::types::{AccountId, PeftType, TaskSpec, ValidatorEvaluation};
use DePEFT::blockchain::{AppChainState, RelativeConsensusEngine, Transaction};
use DePEFT::miner::{MinerHyperparams, MinerNode};
use DePEFT::ml::dataset::Dataset;
use DePEFT::ml::model::DePEFTModel;
use DePEFT::ml::tensor::{Matrix, QuantizedWeight};
use DePEFT::storage::safetensors::{deserialize_safetensors, serialize_safetensors};
use DePEFT::storage::vector_db::{AdapterVectorRecord, EmbeddedVectorDb};
use DePEFT::tournament::TournamentEngine;
use DePEFT::validator::{TeeSandbox, ValidatorNode};
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn test_task_spec_data_structure() {
    let spec = TaskSpec {
        task_id: 42,
        client_address: AccountId::new("client-bob"),
        base_model_id: b"Qwen/Qwen2.5-7B".to_vec(),
        base_model_hash: [0x55; 32],
        dataset_cid: b"bafybeicid123456789".to_vec(),
        peft_method: PeftType::QLoRA_NF4,
        max_rank: 64,
        target_modules: vec![b"q_proj".to_vec(), b"v_proj".to_vec()],
        bounty_pool: 25_000,
        epoch_end_block: 500,
    };

    assert_eq!(spec.task_id, 42);
    assert_eq!(spec.base_model_id_str(), "Qwen/Qwen2.5-7B");
    assert_eq!(spec.dataset_cid_str(), "bafybeicid123456789");
    assert_eq!(spec.target_modules_str(), vec!["q_proj", "v_proj"]);
    assert_eq!(spec.bounty_pool, 25_000);
}

#[test]
fn test_relative_consensus_borda_aggregation() {
    let m1 = AccountId::new("miner-alpha");
    let m2 = AccountId::new("miner-beta");
    let m3 = AccountId::new("miner-gamma");

    let candidates = vec![m1.clone(), m2.clone(), m3.clone()];

    // Validator 1 (NVIDIA): m1 > m2 > m3
    let v1 = ValidatorEvaluation {
        validator_address: AccountId::new("v1"),
        ranking: vec![m1.clone(), m2.clone(), m3.clone()],
        loss_scores: vec![(m1.clone(), 0.12), (m2.clone(), 0.15), (m3.clone(), 0.20)],
        accuracy_scores: vec![(m1.clone(), 0.95), (m2.clone(), 0.90), (m3.clone(), 0.85)],
        hardware_info: "NVIDIA RTX 4090".to_string(),
        attestation_quote: None,
    };

    // Validator 2 (AMD): m1 > m3 > m2 (slight float difference in lower ranks)
    let v2 = ValidatorEvaluation {
        validator_address: AccountId::new("v2"),
        ranking: vec![m1.clone(), m3.clone(), m2.clone()],
        loss_scores: vec![(m1.clone(), 0.120015), (m3.clone(), 0.150010), (m2.clone(), 0.150020)],
        accuracy_scores: vec![(m1.clone(), 0.95), (m3.clone(), 0.90), (m2.clone(), 0.89)],
        hardware_info: "AMD RX 7900".to_string(),
        attestation_quote: None,
    };

    // Validator 3 (CPU): m1 > m2 > m3
    let v3 = ValidatorEvaluation {
        validator_address: AccountId::new("v3"),
        ranking: vec![m1.clone(), m2.clone(), m3.clone()],
        loss_scores: vec![(m1.clone(), 0.119998), (m2.clone(), 0.149990), (m3.clone(), 0.199990)],
        accuracy_scores: vec![(m1.clone(), 0.95), (m2.clone(), 0.90), (m3.clone(), 0.85)],
        hardware_info: "Intel CPU AVX-512".to_string(),
        attestation_quote: None,
    };

    let consensus = RelativeConsensusEngine::aggregate(&[v1, v2, v3], &candidates)
        .expect("Consensus must succeed");

    assert_eq!(consensus.winner, m1, "Miner Alpha should be undisputed winner");
    assert_eq!(consensus.consensus_ranking[0], m1);
    assert_eq!(consensus.agreement_rate, 1.0);
}

#[test]
fn test_commit_reveal_anti_collusion_verification() {
    let mut chain = AppChainState::new();
    let client = AccountId::new("client-1");
    let miner = AccountId::new("miner-1");

    chain.mint(client.clone(), 10_000);

    // Create task
    chain
        .apply_transaction(
            Transaction::CreateTask {
                client: client.clone(),
                nonce: chain.nonce_of(&client),
                base_model_id: b"test-model".to_vec(),
                base_model_hash: [0; 32],
                dataset_cid: b"test-cid".to_vec(),
                peft_method: PeftType::QLoRA_NF4,
                max_rank: 16,
                target_modules: vec![b"q_proj".to_vec()],
                bounty_pool: 5_000,
                epoch_blocks: 50,
            },
            &client,
        )
        .unwrap();

    let task_id = 1;
    chain.start_round(task_id, 1, "bafy_base_w0".to_string()).unwrap();

    let adapter_hash = [0x42u8; 32];
    let salt = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let commit_hash = AppChainState::compute_commit_hash(&adapter_hash, &salt);

    // Miner commits hash
    chain
        .apply_transaction(
            Transaction::CommitAdapter {
                task_id,
                round: 1,
                miner: miner.clone(),
                nonce: chain.nonce_of(&miner),
                commit_hash,
            },
            &miner,
        )
        .unwrap();

    // Transition to Reveal phase
    chain
        .set_round_phase(task_id, 1, DePEFT::blockchain::RoundPhase::RevealPhase)
        .unwrap();

    // Valid reveal succeeds
    let valid_reveal = Transaction::RevealAdapter {
        task_id,
        round: 1,
        miner: miner.clone(),
        nonce: chain.nonce_of(&miner),
        adapter_cid: "bafy_adapter_1".to_string(),
        salt: salt.clone(),
        adapter_hash,
    };
    assert!(chain.apply_transaction(valid_reveal, &miner).is_ok());

    // Invalid salt reveal fails
    let fake_miner = AccountId::new("miner-cheater");
    chain
        .round_contexts
        .get_mut(&(task_id, 1))
        .unwrap()
        .commits
        .insert(
            fake_miner.clone(),
            DePEFT::blockchain::CommitRecord {
                miner_address: fake_miner.clone(),
                commit_hash: [0x99; 32],
                submitted_at_block: 2,
            },
        );

    let invalid_reveal = Transaction::RevealAdapter {
        task_id,
        round: 1,
        miner: fake_miner.clone(),
        nonce: chain.nonce_of(&fake_miner),
        adapter_cid: "bafy_fake".to_string(),
        salt: vec![0, 0, 0, 0],
        adapter_hash: [0x11; 32],
    };
    assert!(chain.apply_transaction(invalid_reveal, &fake_miner).is_err());
}

#[test]
fn test_nf4_quantization_and_dequantization() {
    let mut rng = StdRng::seed_from_u64(12345);
    let original = Matrix::xavier_uniform(64, 64, &mut rng);

    let quant_nf4 = QuantizedWeight::quantize_nf4(&original, 64);
    let dequant = quant_nf4.dequantize();

    assert_eq!(original.rows, dequant.rows);
    assert_eq!(original.cols, dequant.cols);

    // Verify low quantization error (NF4 optimal for normal distributions)
    let mut mse = 0.0f32;
    for (a, b) in original.data.iter().zip(&dequant.data) {
        mse += (a - b).powi(2);
    }
    mse /= (original.rows * original.cols) as f32;

    assert!(mse < 0.005, "NF4 quantization MSE should be small: got {}", mse);
}

#[test]
fn test_safetensors_serialization_roundtrip() {
    let mut rng = StdRng::seed_from_u64(777);
    let model = DePEFTModel::new("test-llama", 8, 16, 4, 4, PeftType::QLoRA_NF4, &mut rng);
    let adapter_pkg = model.export_adapters(1);

    let bytes = serialize_safetensors(&adapter_pkg).expect("Serialization failed");
    assert!(bytes.len() > 8);

    let restored_pkg = deserialize_safetensors(&bytes).expect("Deserialization failed");
    assert_eq!(restored_pkg.model_id, adapter_pkg.model_id);
    assert_eq!(restored_pkg.round, adapter_pkg.round);
    assert_eq!(restored_pkg.modules.len(), adapter_pkg.modules.len());

    for (name, orig_mod) in &adapter_pkg.modules {
        let restored_mod = restored_pkg.modules.get(name).expect("Module missing");
        assert_eq!(orig_mod.rank, restored_mod.rank);
        assert_eq!(orig_mod.lora_a.data, restored_mod.lora_a.data);
        assert_eq!(orig_mod.lora_b.data, restored_mod.lora_b.data);
    }
}

#[test]
fn test_embedded_vector_db_similarity_and_plagiarism() {
    let vdb = EmbeddedVectorDb::new(4);
    let m1 = AccountId::new("miner-original");
    let m2 = AccountId::new("miner-colluder");

    vdb.insert(AdapterVectorRecord {
        adapter_cid: "bafy_m1".to_string(),
        miner_address: m1.clone(),
        task_id: 1,
        round: 1,
        signature: vec![1.0, 0.0, 0.0, 0.0],
    });

    let query_m1 = vec![1.0, 0.0, 0.0, 0.0];
    let top = vdb.search(&query_m1, 1);
    assert_eq!(top.len(), 1);
    assert!((top[0].similarity - 1.0).abs() < 1e-5);

    // Plagiarism check: Miner 2 submitting 99.99% identical signature
    let query_m2 = vec![0.99999, 0.0001, 0.0, 0.0];
    let plagiarism = vdb.check_plagiarism(&query_m2, &m2);
    assert!(plagiarism.is_some(), "Plagiarism should be detected");
    assert_eq!(plagiarism.unwrap().record.miner_address, m1);
}

#[test]
fn test_relora_tournament_multi_round_convergence() {
    let mut rng = StdRng::seed_from_u64(42);
    let full_data = Dataset::generate_synthetic_task(80, 8, 4, 1.5, &mut rng);
    let (train_set, test_set) = full_data.train_test_split(0.7, &mut rng);

    let miners = vec![
        MinerNode::new(
            "miner-fast",
            MinerHyperparams {
                learning_rate: 0.08,
                epochs: 6,
                batch_size: 16,
                hardware_type: "NVIDIA CUDA".to_string(),
            },
        ),
        MinerNode::new(
            "miner-steady",
            MinerHyperparams {
                learning_rate: 0.04,
                epochs: 8,
                batch_size: 16,
                hardware_type: "AMD ROCm".to_string(),
            },
        ),
    ];

    let validators = vec![
        ValidatorNode::new(
            "val-cuda",
            "NVIDIA CUDA",
            0.0,
            TeeSandbox::new(test_set.clone(), "tee-1"),
            Some(EmbeddedVectorDb::new(64)),
        ),
        ValidatorNode::new(
            "val-rocm",
            "AMD ROCm",
            0.000015,
            TeeSandbox::new(test_set.clone(), "tee-2"),
            Some(EmbeddedVectorDb::new(64)),
        ),
    ];

    let mut engine = TournamentEngine::new(
        AccountId::new("client-foundation"),
        5_000,
        2,
        miners,
        validators,
        train_set,
        test_set,
        PeftType::QLoRA_NF4,
    )
    .unwrap();

    let (loss_r0, _) = engine.base_model.evaluate(&engine.dataset_test, 0.0);

    // Run Round 1
    let summary_r1 = engine.run_round(1).unwrap();
    println!("R1: pre={:.6}, post={:.6}", summary_r1.pre_merge_loss, summary_r1.post_merge_loss);

    // Run Round 2 (ReLoRA continuous training on evolved W_1)
    let summary_r2 = engine.run_round(2).unwrap();
    println!("R2: pre={:.6}, post={:.6}", summary_r2.pre_merge_loss, summary_r2.post_merge_loss);

    let (final_loss, _) = engine.base_model.evaluate(&engine.dataset_test, 0.0);
    assert!(
        final_loss < loss_r0,
        "ReLoRA multi-round training must continuously reduce loss (Initial: {:.4} -> Final: {:.4})",
        loss_r0,
        final_loss
    );
}

#[test]
fn test_ed25519_cryptographic_signing_and_verification() {
    use DePEFT::crypto::AccountKeypair;

    let keypair = AccountKeypair::generate();
    let account_id = keypair.account_id();
    assert!(account_id.0.starts_with("0x"));

    let tx = Transaction::CreateTask {
        client: account_id.clone(),
        nonce: 0,
        base_model_id: b"test-model".to_vec(),
        base_model_hash: [0x77; 32],
        dataset_cid: b"bafy_test".to_vec(),
        peft_method: PeftType::QLoRA_NF4,
        max_rank: 32,
        target_modules: vec![b"q_proj".to_vec()],
        bounty_pool: 15_000,
        epoch_blocks: 100,
    };

    let signed_tx = keypair.sign_transaction(tx.clone()).expect("Signing failed");
    assert_eq!(signed_tx.signature.len(), 64);

    // Valid verification returns original sender
    let recovered_sender = signed_tx.verify_signature().expect("Verification failed");
    assert_eq!(recovered_sender, account_id);

    // Tampered transaction must fail verification
    let mut tampered_tx = signed_tx.clone();
    if let Transaction::CreateTask { ref mut bounty_pool, .. } = tampered_tx.tx {
        *bounty_pool = 999_999; // Attacker modified bounty
    }
    assert!(tampered_tx.verify_signature().is_err(), "Tampered transaction signature must fail");
}

#[test]
fn test_disk_ipfs_storage_persistence() {
    use DePEFT::storage::DiskIpfsStorage;
    let temp_dir = std::env::temp_dir().join(format!("depeft_test_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));

    let storage = DiskIpfsStorage::new(&temp_dir).expect("Failed to init storage");
    let payload = b"Hello DePEFT Decentralized Storage!";

    let cid = storage.put(payload).expect("Failed to put bytes");
    assert!(cid.starts_with("bafy"));
    assert!(storage.contains(&cid));

    let retrieved = storage.get(&cid).expect("Failed to get bytes");
    assert_eq!(retrieved, payload);

    // Clean up
    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn test_live_node_http_rpc_integration() {
    use DePEFT::client::DePeftClient;
    use DePEFT::crypto::AccountKeypair;
    use DePEFT::storage::DiskIpfsStorage;
    use DePEFT::storage::vector_db::EmbeddedVectorDb;
    use std::net::SocketAddr;
    use std::sync::{Arc, RwLock};

    let temp_dir = std::env::temp_dir().join(format!("depeft_rpc_test_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let storage = Arc::new(DiskIpfsStorage::new(&temp_dir).unwrap());
    let chain = Arc::new(RwLock::new(AppChainState::new()));
    let vector_db = Arc::new(EmbeddedVectorDb::new(64));

    // Mint tokens for test client
    let client_keypair = AccountKeypair::generate();
    chain.write().unwrap().mint(client_keypair.account_id(), 50_000);

    // Bind to random available local port
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    let local_addr = listener.local_addr().unwrap();

    let server_chain = chain.clone();
    let server_storage = storage.clone();
    let server_vdb = vector_db.clone();

    // Spawn server in background tokio task
    tokio::spawn(async move {
        let app = DePEFT::node::create_app(DePEFT::node::NodeContext {
            chain: server_chain,
            storage: server_storage,
            vector_db: server_vdb,
            swarm: None,
        });
        axum::serve(listener, app).await.unwrap();
    });

    let client = DePeftClient::new(format!("http://{}", local_addr));

    // 1. Test Node Status
    let status = client.get_status().await.expect("Failed to get node status");
    assert_eq!(status.block_height, 1);
    assert_eq!(status.tasks_count, 0);

    // 2. Test CAS Artifact Upload & Download
    let artifact_data = b"DePEFT LoRA Safetensors Binary Bytes";
    let cid = client.upload_storage(artifact_data).await.expect("Upload failed");
    assert!(cid.starts_with("bafy"));

    let downloaded = client.download_storage(&cid).await.expect("Download failed");
    assert_eq!(downloaded, artifact_data);

    // 3. Test Signed Transaction Broadcast (CreateTask)
    let create_task_tx = Transaction::CreateTask {
        client: client_keypair.account_id(),
        nonce: 0,
        base_model_id: b"Qwen/Qwen2.5-7B".to_vec(),
        base_model_hash: [0x88; 32],
        dataset_cid: cid.as_bytes().to_vec(),
        peft_method: PeftType::QLoRA_NF4,
        max_rank: 64,
        target_modules: vec![b"q_proj".to_vec()],
        bounty_pool: 20_000,
        epoch_blocks: 50,
    };

    let signed_tx = client_keypair.sign_transaction(create_task_tx).unwrap();
    let res = client.submit_transaction(&signed_tx).await.expect("Submit tx failed");
    assert!(res.contains("successfully committed"));

    // 4. Verify Task Exists on Node
    let tasks = client.get_tasks().await.expect("Failed to fetch tasks");
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].base_model_id_str(), "Qwen/Qwen2.5-7B");

    // Clean up
    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn test_p2p_swarm_bidirectional_gossip_and_deduplication() {
    use DePEFT::crypto::AccountKeypair;
    use DePEFT::p2p::{P2pMessage, P2pSwarm, PeerId};
    use std::sync::Arc;

    let kp_a = AccountKeypair::generate();
    let kp_b = AccountKeypair::generate();

    let peer_a = PeerId::from_account(&kp_a.account_id());
    let peer_b = PeerId::from_account(&kp_b.account_id());

    // Bind Node A to random port
    let listener_a = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr_a = listener_a.local_addr().unwrap();
    drop(listener_a); // free port for swarm

    // Bind Node B to random port
    let listener_b = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr_b = listener_b.local_addr().unwrap();
    drop(listener_b);

    let (swarm_a, _tx_rx_a, mut msg_rx_a) = P2pSwarm::new(peer_a.clone(), addr_a);
    let (swarm_b, mut tx_rx_b, _msg_rx_b) = P2pSwarm::new(peer_b.clone(), addr_b);

    let swarm_a = Arc::new(swarm_a);
    let swarm_b = Arc::new(swarm_b);

    swarm_a.clone().start_listener().await.unwrap();
    swarm_b.clone().start_listener().await.unwrap();

    // Node B connects to Node A over TCP
    let connected_id = swarm_b.connect_peer(&addr_a.to_string()).await.expect("Failed to connect P2P peers");
    assert_eq!(connected_id, peer_a);

    // Wait a brief moment for handshake to settle
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    assert_eq!(swarm_a.peer_count(), 1);
    assert_eq!(swarm_b.peer_count(), 1);

    // Node A broadcasts a signed transaction
    let tx = Transaction::CreateTask {
        client: kp_a.account_id(),
        nonce: 0,
        base_model_id: b"test-gossip-model".to_vec(),
        base_model_hash: [0xaa; 32],
        dataset_cid: b"bafy_gossip".to_vec(),
        peft_method: PeftType::QLoRA_NF4,
        max_rank: 64,
        target_modules: vec![b"q_proj".to_vec()],
        bounty_pool: 10_000,
        epoch_blocks: 50,
    };
    let signed_tx = kp_a.sign_transaction(tx).unwrap();

    swarm_a.broadcast_transaction(signed_tx.clone());

    // Node B must receive the gossiped transaction
    let received_tx = tokio::time::timeout(tokio::time::Duration::from_secs(2), tx_rx_b.recv())
        .await
        .expect("Timeout waiting for P2P transaction")
        .expect("Channel closed");

    assert_eq!(received_tx.verify_signature().unwrap(), kp_a.account_id());

    // Test CID Announcement gossip
    swarm_b.broadcast(P2pMessage::AnnounceCid {
        cid: "bafy_adapter_gossiped_123".to_string(),
        size_bytes: 4096,
        provider: peer_b.clone(),
    });

    let (from_peer, received_msg) = tokio::time::timeout(tokio::time::Duration::from_secs(2), msg_rx_a.recv())
        .await
        .expect("Timeout waiting for CID announcement")
        .expect("Channel closed");

    assert_eq!(from_peer, peer_b);
    if let P2pMessage::AnnounceCid { cid, .. } = received_msg {
        assert_eq!(cid, "bafy_adapter_gossiped_123");
    } else {
        panic!("Expected AnnounceCid message");
    }
}

#[test]
fn test_candle_lora_linear_forward_and_merge() {
    use candle_core::{Device, Tensor};
    use DePEFT::candle_peft::CandleLoraLinear;

    let device = Device::Cpu;
    let base_w = Tensor::randn(0f32, 0.1, (16, 32), &device).unwrap();
    let mut lora = CandleLoraLinear::new(base_w.clone(), 4, 8.0, &device).unwrap();

    let x = Tensor::randn(0f32, 1.0, (2, 32), &device).unwrap();

    // Initial forward pass
    let y_init = lora.forward(&x).unwrap();
    assert_eq!(y_init.dims2().unwrap(), (2, 16));

    // Initially lora_b is 0, so y_init matches base_w forward
    let base_y = x.matmul(&base_w.t().unwrap()).unwrap();
    let diff = (&y_init - &base_y).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(diff < 1e-5, "At init, LoRA output must match base model");

    // Perform ReLoRA merge
    lora.merge_and_reset().unwrap();
    let y_merged = lora.forward(&x).unwrap();
    assert_eq!(y_merged.dims2().unwrap(), (2, 16));
}

#[test]
fn test_candle_llm_transformer_relora_tournament() {
    use DePEFT::candle_peft::{
        CandleMinerHyperparams, CandleMinerTrainer, CandleTransformerConfig, CandleTransformerLM,
        CandleValidatorEvaluator, CandleWeightMerger,
    };

    let device = candle_core::Device::Cpu;
    let config = CandleTransformerConfig {
        vocab_size: 256,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        max_position_embeddings: 64,
        lora_rank: 4,
        lora_alpha: 8.0,
    };

    let mut model = CandleTransformerLM::new(config, device).unwrap();

    let train_corpus = vec![
        "DePEFT parameter-efficient fine-tuning on Candle LLM.".to_string(),
        "ReLoRA multi-round continuous weight evolution.".to_string(),
    ];

    let private_test_set = vec![
        "Decentralized fine-tuning over P2P network.".to_string(),
    ];

    let initial_loss = CandleValidatorEvaluator::evaluate_dataset(&model, &private_test_set).unwrap();

    // Miner trains adapter
    let hyperparams = CandleMinerHyperparams {
        learning_rate: 0.01,
        steps: 10,
        batch_size: 2,
        optimizer_type: DePEFT::candle_peft::PeftOptimizerType::AdamW,
        hardware_info: "CPU Test".to_string(),
        ..Default::default()
    };

    let mut miner_model = model.clone();
    let artifact = CandleMinerTrainer::train(&mut miner_model, &train_corpus, &hyperparams).unwrap();
    assert!(artifact.final_loss <= artifact.train_loss, "Training steps must reduce or equal loss");
    assert!(!artifact.safetensors_bytes.is_empty());

    // Validator evaluates
    let candidate_adapters = vec![(AccountId::new("miner-1"), artifact.safetensors_bytes.clone())];
    let eval = CandleValidatorEvaluator::evaluate_miners(
        &model,
        &private_test_set,
        &candidate_adapters,
        AccountId::new("validator-1"),
        "CPU",
        0.0,
    )
    .unwrap();

    assert_eq!(eval.ranking.len(), 1);
    assert_eq!(eval.ranking[0], AccountId::new("miner-1"));

    // ReLoRA merge winning adapter
    CandleWeightMerger::merge_winning_adapter(&mut model, &artifact.safetensors_bytes).unwrap();

    let evolved_loss = CandleValidatorEvaluator::evaluate_dataset(&model, &private_test_set).unwrap();
    println!("Candle ReLoRA Initial Loss: {:.4}, Evolved Loss: {:.4}", initial_loss, evolved_loss);
}

#[test]
fn test_hardware_tee_remote_attestation_and_on_chain_verification() {
    use DePEFT::blockchain::types::AccountId;
    use DePEFT::tee::{HardwareTeeEnclave, OnChainTeeVerifier, TeeType};

    let m1 = AccountId::new("miner-alpha");
    let m2 = AccountId::new("miner-beta");
    let ranking = vec![m1.clone(), m2.clone()];

    // 1. Official Validator Enclave generates quote
    let enclave = HardwareTeeEnclave::official(TeeType::IntelSgxDcap);
    let quote = enclave.generate_quote(10, 2, &ranking).expect("Failed to generate quote");

    // 2. On-Chain Verifier checks valid quote
    let verifier = OnChainTeeVerifier::default();
    assert!(verifier.verify_quote(&quote, 10, 2, &ranking).is_ok(), "Valid quote must pass on-chain verification");

    // 3. Tampered ranking (e.g. malicious validator tried to flip winner to miner-beta)
    let tampered_ranking = vec![m2.clone(), m1.clone()];
    let tamper_res = verifier.verify_quote(&quote, 10, 2, &tampered_ranking);
    assert!(tamper_res.is_err(), "Tampered ranking must fail report_data check");

    // 4. Rogue enclave with unapproved MRENCLAVE measurement
    let rogue_enclave = HardwareTeeEnclave::new(TeeType::IntelSgxDcap, "malicious-unapproved-enclave");
    let rogue_quote = rogue_enclave.generate_quote(10, 2, &ranking).unwrap();
    let rogue_res = verifier.verify_quote(&rogue_quote, 10, 2, &ranking);
    assert!(rogue_res.is_err(), "Unapproved MRENCLAVE must be rejected on-chain");

    // 5. Forged quote signature
    let mut forged_quote = quote.clone();
    forged_quote.quote_signature[0] ^= 0xff;
    let forge_res = verifier.verify_quote(&forged_quote, 10, 2, &ranking);
    assert!(forge_res.is_err(), "Forged signature must fail cryptographic check");
}

#[tokio::test]
async fn test_hybrid_storage_and_ipfs_cas_caching() {
    use DePEFT::storage::{DiskIpfsStorage, HybridStorageManager, IpfsKuboClient};
    use std::sync::Arc;

    let temp_dir = std::env::temp_dir().join(format!("depeft_test_storage_{}", rand::random::<u64>()));
    let local_cas = Arc::new(DiskIpfsStorage::new(&temp_dir).unwrap());
    let kubo_client = Some(Arc::new(IpfsKuboClient::new("http://127.0.0.1:5001", "http://127.0.0.1:8080")));

    let manager = HybridStorageManager::new(local_cas.clone(), kubo_client);

    // 1. Store model weights / dataset
    let model_data = b"depeft_qwen2.5_lora_adapter_binary_safetensors_bytes_payload";
    let cid = manager.put(model_data, "adapter.safetensors").await.expect("Must store");
    assert!(cid.starts_with("bafy") || cid.starts_with("Qm"), "Must generate valid IPFS CID");

    // 2. Local CAS contains check
    assert!(manager.contains_local(&cid));

    // 3. Fast retrieval
    let retrieved = manager.get(&cid).await.expect("Must retrieve").expect("Must exist");
    assert_eq!(retrieved, model_data);

    // Clean up temp dir
    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_bft_consensus_2_phase_commit_and_equivocation_slashing() {
    use DePEFT::consensus::{BftEngine, ConsensusValidator, SlashingEngine};
    use DePEFT::crypto::AccountKeypair;

    let kp1 = AccountKeypair::generate();
    let kp2 = AccountKeypair::generate();
    let kp3 = AccountKeypair::generate();
    let kp4 = AccountKeypair::generate();

    let validators = vec![
        ConsensusValidator { address: kp1.account_id(), voting_power: 10 },
        ConsensusValidator { address: kp2.account_id(), voting_power: 10 },
        ConsensusValidator { address: kp3.account_id(), voting_power: 10 },
        ConsensusValidator { address: kp4.account_id(), voting_power: 10 },
    ];

    let mut bft = BftEngine::new(validators, [0xaa; 32]);
    let mut slasher = SlashingEngine::new();

    assert_eq!(bft.validator_set.total_voting_power, 40);
    assert_eq!(bft.validator_set.two_thirds_threshold(), 27);

    // Height 1 Proposal
    let proposer = bft.validator_set.get_proposer(1, 0);
    let proposer_kp = if kp1.account_id() == proposer {
        &kp1
    } else if kp2.account_id() == proposer {
        &kp2
    } else if kp3.account_id() == proposer {
        &kp3
    } else {
        &kp4
    };

    let proposal = bft.create_proposal(proposer_kp, Vec::new()).expect("Proposal succeeds");
    let block_hash = proposal.block_hash();

    // Round 0 Prevotes (3/4 validators = 30 power >= 27)
    let v1 = bft.cast_prevote(&kp1, Some(block_hash)).unwrap();
    let v2 = bft.cast_prevote(&kp2, Some(block_hash)).unwrap();
    let v3 = bft.cast_prevote(&kp3, Some(block_hash)).unwrap();

    let _ = bft.add_vote(v1);
    let _ = bft.add_vote(v2);
    let _ = bft.add_vote(v3);

    // Round 0 Precommits
    let pc1 = bft.cast_precommit(&kp1, Some(block_hash)).unwrap();
    let pc2 = bft.cast_precommit(&kp2, Some(block_hash)).unwrap();
    let pc3 = bft.cast_precommit(&kp3, Some(block_hash)).unwrap();

    let _ = bft.add_vote(pc1);
    let _ = bft.add_vote(pc2);
    let finalized = bft.add_vote(pc3).unwrap();

    assert!(finalized.is_some(), "Block must finalize once 2/3+ precommits are gathered");
    let block = finalized.unwrap();
    assert_eq!(block.header.height, 1);
    assert_eq!(bft.current_height, 2, "Consensus state advances to height 2");
    assert_eq!(bft.blockchain.len(), 1);

    // Test Byzantine Equivocation Slashing
    let vote_a = bft.cast_prevote(&kp4, Some([0x11; 32])).unwrap();
    let vote_b = bft.cast_prevote(&kp4, Some([0x22; 32])).unwrap();

    assert!(slasher.check_vote(&vote_a).unwrap().is_none());
    let evidence = slasher.check_vote(&vote_b).unwrap();
    assert!(evidence.is_some(), "Double voting must be detected");
    assert!(slasher.is_slashed(&kp4.account_id()));
}
