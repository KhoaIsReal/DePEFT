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
        reward_distribution: DePEFT::blockchain::types::RewardDistribution::TopKDecay {
            top_k: 5,
            decay_rate: 0.5,
        },
        merge_strategy: DePEFT::blockchain::types::MergeStrategy::EnsembleWeighted { top_k: 5 },
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
                reward_distribution: Default::default(),
                merge_strategy: Default::default(),
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
fn test_evaluation_requires_a_complete_unique_revealed_ranking() {
    let mut chain = AppChainState::new();
    chain.tee_verifier.enforce_attestation = false;
    let client = AccountId::new("client-ranking");
    let miner_a = AccountId::new("miner-ranking-a");
    let miner_b = AccountId::new("miner-ranking-b");
    let validator = AccountId::new("validator-ranking");
    chain.mint(client.clone(), 10_000);

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
                reward_distribution: Default::default(),
                merge_strategy: Default::default(),
            },
            &client,
        )
        .unwrap();
    chain.start_round(1, 1, "bafy_base".to_string()).unwrap();

    for miner in [&miner_a, &miner_b] {
        let adapter_hash = if miner == &miner_a { [1; 32] } else { [2; 32] };
        let salt = vec![1; 16];
        chain
            .apply_transaction(
                Transaction::CommitAdapter {
                    task_id: 1,
                    round: 1,
                    miner: (*miner).clone(),
                    nonce: chain.nonce_of(miner),
                    commit_hash: AppChainState::compute_commit_hash(&adapter_hash, &salt),
                },
                miner,
            )
            .unwrap();
    }
    chain
        .set_round_phase(1, 1, DePEFT::blockchain::RoundPhase::RevealPhase)
        .unwrap();
    for miner in [&miner_a, &miner_b] {
        let adapter_hash = if miner == &miner_a { [1; 32] } else { [2; 32] };
        chain
            .apply_transaction(
                Transaction::RevealAdapter {
                    task_id: 1,
                    round: 1,
                    miner: (*miner).clone(),
                    nonce: chain.nonce_of(miner),
                    adapter_cid: format!("bafy_adapter_{}", miner.as_str()),
                    salt: vec![1; 16],
                    adapter_hash,
                },
                miner,
            )
            .unwrap();
    }
    chain
        .set_round_phase(1, 1, DePEFT::blockchain::RoundPhase::EvaluationPhase)
        .unwrap();

    let incomplete = Transaction::SubmitEvaluation {
        task_id: 1,
        round: 1,
        nonce: chain.nonce_of(&validator),
        evaluation: ValidatorEvaluation {
            validator_address: validator.clone(),
            ranking: vec![miner_a.clone()],
            loss_scores: vec![],
            accuracy_scores: vec![],
            hardware_info: "test".to_string(),
            attestation_quote: None,
        },
    };
    assert!(chain.apply_transaction(incomplete, &validator).is_err());

    let duplicate = Transaction::SubmitEvaluation {
        task_id: 1,
        round: 1,
        nonce: chain.nonce_of(&validator),
        evaluation: ValidatorEvaluation {
            validator_address: validator.clone(),
            ranking: vec![miner_a.clone(), miner_a.clone()],
            loss_scores: vec![],
            accuracy_scores: vec![],
            hardware_info: "test".to_string(),
            attestation_quote: None,
        },
    };
    assert!(chain.apply_transaction(duplicate, &validator).is_err());

    let valid = Transaction::SubmitEvaluation {
        task_id: 1,
        round: 1,
        nonce: chain.nonce_of(&validator),
        evaluation: ValidatorEvaluation {
            validator_address: validator.clone(),
            ranking: vec![miner_a, miner_b],
            loss_scores: vec![],
            accuracy_scores: vec![],
            hardware_info: "test".to_string(),
            attestation_quote: None,
        },
    };
    assert!(chain.apply_transaction(valid, &validator).is_ok());
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
        reward_distribution: Default::default(),
        merge_strategy: Default::default(),
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

#[test]
fn test_chain_store_persists_and_verifies_canonical_state() {
    use DePEFT::storage::ChainStore;

    let db_path = std::env::temp_dir().join(format!(
        "depeft_chain_store_{}.sqlite",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut state = AppChainState::new();
    state.mint(AccountId::new("persisted-account"), 42);
    state.advance_block();
    let expected_hash = ChainStore::state_hash(&state).unwrap();

    {
        let mut store = ChainStore::open(&db_path).unwrap();
        assert_eq!(store.save(&state).unwrap(), expected_hash);
    }
    let store = ChainStore::open(&db_path).unwrap();
    let restored = store.load().unwrap().expect("state must be persisted");
    assert_eq!(restored.block_height, state.block_height);
    assert_eq!(restored.balance_of(&AccountId::new("persisted-account")), 42);
    assert_eq!(ChainStore::state_hash(&restored).unwrap(), expected_hash);

    let _ = std::fs::remove_file(&db_path);
    let _ = std::fs::remove_file(format!("{}-wal", db_path.display()));
    let _ = std::fs::remove_file(format!("{}-shm", db_path.display()));
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
        let app = DePEFT::node::create_app(DePEFT::node::NodeContext::new(
            server_chain,
            server_storage,
            server_vdb,
            None,
        )
        .with_security(DePEFT::node::NodeSecurityConfig::development()));
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
        reward_distribution: Default::default(),
        merge_strategy: Default::default(),
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
        reward_distribution: Default::default(),
        merge_strategy: Default::default(),
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
        1,
        1,
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

    // 2. Production verifier fails closed until a trust root is provisioned.
    assert!(OnChainTeeVerifier::default()
        .verify_quote(&quote, 10, 2, &ranking)
        .is_err());

    // 3. Explicit test trust source admits the quote.
    let mut verifier = OnChainTeeVerifier::default();
    verifier.trust_quote_source(enclave.measurement.mrenclave, enclave.platform_public_key());
    assert!(verifier.verify_quote(&quote, 10, 2, &ranking).is_ok(), "Valid quote must pass on-chain verification");

    // 4. Tampered ranking (e.g. malicious validator tried to flip winner to miner-beta)
    let tampered_ranking = vec![m2.clone(), m1.clone()];
    let tamper_res = verifier.verify_quote(&quote, 10, 2, &tampered_ranking);
    assert!(tamper_res.is_err(), "Tampered ranking must fail report_data check");

    // 5. Rogue enclave with unapproved MRENCLAVE measurement
    let rogue_enclave = HardwareTeeEnclave::new(TeeType::IntelSgxDcap, "malicious-unapproved-enclave");
    let rogue_quote = rogue_enclave.generate_quote(10, 2, &ranking).unwrap();
    let rogue_res = verifier.verify_quote(&rogue_quote, 10, 2, &ranking);
    assert!(rogue_res.is_err(), "Unapproved MRENCLAVE must be rejected on-chain");

    // 6. Forged quote signature
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

#[test]
fn test_security_tee_platform_key_spoofing_rejected() {
    use DePEFT::blockchain::types::AccountId;
    use DePEFT::crypto::AccountKeypair;
    use DePEFT::tee::{HardwareTeeEnclave, OnChainTeeVerifier, TeeType};

    let m1 = AccountId::new("miner-alpha");
    let m2 = AccountId::new("miner-beta");
    let ranking = vec![m1.clone(), m2.clone()];

    // Attacker creates their own enclave with unapproved platform keypair
    let rogue_keypair = AccountKeypair::generate();
    let enclave = HardwareTeeEnclave::official(TeeType::IntelSgxDcap);
    let mut spoofed_quote = enclave.generate_quote(1, 1, &ranking).unwrap();

    // Attacker replaces platform public key with their own and re-signs
    spoofed_quote.platform_public_key = rogue_keypair.public_key_bytes();
    let mut payload = Vec::new();
    payload.push(spoofed_quote.tee_type as u8);
    payload.extend_from_slice(&spoofed_quote.measurement.mrenclave);
    payload.extend_from_slice(&spoofed_quote.measurement.mrsigner);
    payload.extend_from_slice(&spoofed_quote.report_data);
    payload.extend_from_slice(&spoofed_quote.timestamp.to_be_bytes());
    spoofed_quote.quote_signature = rogue_keypair.sign_message(&payload);

    let mut verifier = OnChainTeeVerifier::default();
    // Trust the simulator's measurement, but not the attacker-controlled key.
    verifier.register_mrenclave(enclave.measurement.mrenclave);
    verifier.register_platform_key(enclave.platform_public_key());
    let result = verifier.verify_quote(&spoofed_quote, 1, 1, &ranking);
    assert!(
        result.is_err(),
        "Spoofed platform key must be rejected by Root of Trust whitelist"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Unauthorized TEE Platform Public Key"));
}

#[test]
fn test_security_relative_consensus_validator_deduplication() {
    use DePEFT::blockchain::relative_consensus::RelativeConsensusEngine;
    use DePEFT::blockchain::types::{AccountId, ValidatorEvaluation};

    let m1 = AccountId::new("miner-honest");
    let m2 = AccountId::new("miner-rogue");
    let candidates = vec![m1.clone(), m2.clone()];

    let val1 = AccountId::new("val-1");
    let val2 = AccountId::new("val-2");

    let eval_honest_1 = ValidatorEvaluation {
        validator_address: val1.clone(),
        ranking: vec![m1.clone(), m2.clone()],
        loss_scores: vec![(m1.clone(), 0.1), (m2.clone(), 0.5)],
        accuracy_scores: vec![(m1.clone(), 0.9), (m2.clone(), 0.5)],
        hardware_info: "CPU".to_string(),
        attestation_quote: None,
    };

    let eval_honest_2 = ValidatorEvaluation {
        validator_address: val2.clone(),
        ranking: vec![m1.clone(), m2.clone()],
        loss_scores: vec![(m1.clone(), 0.1), (m2.clone(), 0.5)],
        accuracy_scores: vec![(m1.clone(), 0.9), (m2.clone(), 0.5)],
        hardware_info: "CPU".to_string(),
        attestation_quote: None,
    };

    // Rogue validator attempts to submit multiple duplicate evaluations to skew Borda count
    let eval_rogue_dupe = ValidatorEvaluation {
        validator_address: val1.clone(),
        ranking: vec![m2.clone(), m1.clone()],
        loss_scores: vec![(m2.clone(), 0.01), (m1.clone(), 0.9)],
        accuracy_scores: vec![(m2.clone(), 0.99), (m1.clone(), 0.1)],
        hardware_info: "CPU".to_string(),
        attestation_quote: None,
    };

    let evals = vec![eval_honest_1, eval_honest_2, eval_rogue_dupe];
    let res = RelativeConsensusEngine::aggregate(&evals, &candidates).unwrap();
    assert_eq!(res.winner, m1, "Consensus winner must be honest miner despite duplicate evaluation submission attempt");
}

#[test]
fn test_security_slashing_forged_vote_signature_rejected() {
    use DePEFT::consensus::{SlashingEngine, Vote, VoteType};
    use DePEFT::crypto::AccountKeypair;

    let kp_victim = AccountKeypair::generate();
    let kp_attacker = AccountKeypair::generate();

    let mut slasher = SlashingEngine::new();

    // Victim submits valid vote A
    let sign_bytes = Vote::sign_bytes(VoteType::Prevote, 1, 0, Some([0x11; 32]));
    let sig_valid = kp_victim.sign_message(&sign_bytes);
    let vote_a = Vote {
        vote_type: VoteType::Prevote,
        height: 1,
        round: 0,
        block_hash: Some([0x11; 32]),
        validator: kp_victim.account_id(),
        signature: sig_valid,
    };
    assert!(slasher.check_vote(&vote_a).unwrap().is_none());

    // Attacker crafts forged conflicting vote B pretending to be victim (signed with attacker key)
    let sign_bytes_b = Vote::sign_bytes(VoteType::Prevote, 1, 0, Some([0x22; 32]));
    let sig_forged = kp_attacker.sign_message(&sign_bytes_b);
    let vote_b_forged = Vote {
        vote_type: VoteType::Prevote,
        height: 1,
        round: 0,
        block_hash: Some([0x22; 32]),
        validator: kp_victim.account_id(), // Target honest victim
        signature: sig_forged,
    };

    // Slashing check must fail cryptographic signature verification and NOT slash victim
    let check_res = slasher.check_vote(&vote_b_forged);
    assert!(check_res.is_err(), "Forged vote signature must be rejected");
    assert!(!slasher.is_slashed(&kp_victim.account_id()), "Honest validator must not be slashed by forged evidence");
}

#[test]
fn test_security_vector_db_nan_and_dimension_safety() {
    use DePEFT::blockchain::types::AccountId;
    use DePEFT::storage::vector_db::{AdapterVectorRecord, EmbeddedVectorDb};

    let vdb = EmbeddedVectorDb::new(4);
    let miner = AccountId::new("miner-test");

    // 1. Dimension mismatch must be ignored gracefully without panic
    vdb.insert(AdapterVectorRecord {
        adapter_cid: "bafy_bad_dim".to_string(),
        miner_address: miner.clone(),
        task_id: 1,
        round: 1,
        signature: vec![1.0, 2.0], // len 2 != 4
    });
    assert_eq!(vdb.count(), 0);

    // 2. NaN / Inf vector must be ignored gracefully without panic
    vdb.insert(AdapterVectorRecord {
        adapter_cid: "bafy_nan".to_string(),
        miner_address: miner.clone(),
        task_id: 1,
        round: 1,
        signature: vec![f32::NAN, 1.0, 0.0, 0.0],
    });
    assert_eq!(vdb.count(), 0);

    // 3. Valid vector inserted successfully
    vdb.insert(AdapterVectorRecord {
        adapter_cid: "bafy_valid".to_string(),
        miner_address: miner,
        task_id: 1,
        round: 1,
        signature: vec![1.0, 0.0, 0.0, 0.0],
    });
    assert_eq!(vdb.count(), 1);
}

#[test]
fn test_security_bft_receive_proposal_verification() {
    use DePEFT::consensus::{BftEngine, ConsensusValidator};
    use DePEFT::crypto::AccountKeypair;

    let kp1 = AccountKeypair::generate();
    let kp2 = AccountKeypair::generate();

    let validators = vec![
        ConsensusValidator { address: kp1.account_id(), voting_power: 10 },
        ConsensusValidator { address: kp2.account_id(), voting_power: 10 },
    ];

    let mut bft = BftEngine::new(validators, [0xaa; 32]);
    let proposer = bft.validator_set.get_proposer(1, 0);
    let (proposer_kp, rogue_kp) = if kp1.account_id() == proposer {
        (&kp1, &kp2)
    } else {
        (&kp2, &kp1)
    };

    // 1. Valid proposal created by true proposer must be accepted
    let valid_block = bft.create_proposal(proposer_kp, Vec::new()).unwrap();
    let mut bft_node2 = BftEngine::new(
        vec![
            ConsensusValidator { address: kp1.account_id(), voting_power: 10 },
            ConsensusValidator { address: kp2.account_id(), voting_power: 10 },
        ],
        [0xaa; 32],
    );
    assert!(bft_node2.receive_proposal(valid_block).is_ok());

    // 2. Proposal from unauthorized proposer must be rejected
    let mut rogue_bft = BftEngine::new(
        vec![
            ConsensusValidator { address: kp1.account_id(), voting_power: 10 },
            ConsensusValidator { address: kp2.account_id(), voting_power: 10 },
        ],
        [0xaa; 32],
    );
    let mut fake_block = bft_node2.round_state.proposal.clone().unwrap();
    fake_block.header.proposer = rogue_kp.account_id();
    assert!(rogue_bft.receive_proposal(fake_block).is_err());
}

#[test]
fn test_security_state_rejects_zero_bounty_and_zero_epoch_tasks() {
    use DePEFT::blockchain::state::AppChainState;
    use DePEFT::blockchain::transactions::Transaction;
    use DePEFT::blockchain::types::PeftType;
    use DePEFT::crypto::AccountKeypair;

    let mut state = AppChainState::new();
    let client_kp = AccountKeypair::generate();
    state.mint(client_kp.account_id(), 10_000);

    // 1. Zero bounty must be rejected
    let tx_zero_bounty = Transaction::CreateTask {
        client: client_kp.account_id(),
        nonce: 0,
        base_model_id: b"test-model".to_vec(),
        base_model_hash: [0x11; 32],
        dataset_cid: b"bafy_dataset".to_vec(),
        peft_method: PeftType::QLoRA_NF4,
        max_rank: 16,
        target_modules: vec![b"q_proj".to_vec()],
        bounty_pool: 0,
        epoch_blocks: 10,
        reward_distribution: Default::default(),
        merge_strategy: Default::default(),
    };
    assert!(state.apply_transaction(tx_zero_bounty, &client_kp.account_id()).is_err());

    // 2. Too short epoch (< 5 blocks) must be rejected
    let tx_short_epoch = Transaction::CreateTask {
        client: client_kp.account_id(),
        nonce: 0,
        base_model_id: b"test-model".to_vec(),
        base_model_hash: [0x11; 32],
        dataset_cid: b"bafy_dataset".to_vec(),
        peft_method: PeftType::QLoRA_NF4,
        max_rank: 16,
        target_modules: vec![b"q_proj".to_vec()],
        bounty_pool: 1_000,
        epoch_blocks: 2,
        reward_distribution: Default::default(),
        merge_strategy: Default::default(),
    };
    assert!(state.apply_transaction(tx_short_epoch, &client_kp.account_id()).is_err());
}

#[test]
fn test_security_safetensors_out_of_bounds_offsets_rejected() {
    use DePEFT::storage::deserialize_safetensors;

    // Header specifying offsets [0, 999999] while binary payload is only 16 bytes
    let header_json = r#"{"__metadata__":{"model_id":"m1"},"layer.lora_a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,999999]},"layer.lora_b":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
    let header_len = header_json.len() as u64;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&header_len.to_le_bytes());
    bytes.extend_from_slice(header_json.as_bytes());
    bytes.extend_from_slice(&[0u8; 16]); // only 16 bytes payload

    let res = deserialize_safetensors(&bytes);
    assert!(res.is_err(), "Out of bounds data_offsets must be rejected without panic");
}

#[test]
fn test_security_quantization_zero_block_size_safety() {
    use DePEFT::ml::tensor::{Matrix, QuantizedWeight};

    let mat = Matrix::zeros(4, 4);
    // Block size 0 should not trigger integer division by zero panic
    let q_nf4 = QuantizedWeight::quantize_nf4(&mat, 0);
    let dequant_nf4 = q_nf4.dequantize();
    assert_eq!(dequant_nf4.rows, 4);
    assert_eq!(dequant_nf4.cols, 4);

    let q_int4 = QuantizedWeight::quantize_int4(&mat, 0);
    let dequant_int4 = q_int4.dequantize();
    assert_eq!(dequant_int4.rows, 4);
    assert_eq!(dequant_int4.cols, 4);
}

#[test]
fn test_security_p2p_seen_cache_fifo_eviction() {
    use DePEFT::p2p::{P2pSwarm, PeerId};
    use std::net::SocketAddr;

    let addr: SocketAddr = "127.0.0.1:29999".parse().unwrap();
    let (swarm, _tx, _msg) = P2pSwarm::new(PeerId("local_peer".to_string()), addr);

    let msg1 = [1u8; 32];
    swarm.mark_seen(msg1);
    assert!(swarm.is_seen(&msg1));

    // Fill up seen cache up to MAX_SEEN_CACHE + 1
    for i in 0..10_001 {
        let mut hash = [0u8; 32];
        hash[0..4].copy_from_slice(&(i as u32).to_le_bytes());
        swarm.mark_seen(hash);
    }

    // Latest message should definitely be seen
    let mut latest = [0u8; 32];
    latest[0..4].copy_from_slice(&(10_000u32).to_le_bytes());
    assert!(swarm.is_seen(&latest));
}

#[test]
fn test_security_dataset_train_test_split_nan_ratio_clamped() {
    use DePEFT::ml::dataset::Dataset;
    use rand::SeedableRng;

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let ds = Dataset::generate_synthetic_task(20, 4, 2, 1.0, &mut rng);

    // Split with NaN ratio
    let (train, test) = ds.train_test_split(f32::NAN, &mut rng);
    assert_eq!(train.len() + test.len(), 20);

    // Split with out-of-bounds ratio (1.5)
    let (train_high, test_high) = ds.train_test_split(1.5, &mut rng);
    assert_eq!(train_high.len(), 20);
    assert_eq!(test_high.len(), 0);
}

#[test]
fn test_security_validator_tolerates_corrupted_miner_cid() {
    use DePEFT::blockchain::types::PeftType;
    use DePEFT::blockchain::transactions::Transaction;
    use DePEFT::crypto::AccountKeypair;
    use DePEFT::ml::dataset::Dataset;
    use DePEFT::ml::model::DePEFTModel;
    use DePEFT::storage::ipfs::IpfsStorage;
    use DePEFT::validator::tee::TeeSandbox;
    use DePEFT::validator::worker::ValidatorNode;
    use rand::SeedableRng;
    use sha2::Digest;

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let ipfs = IpfsStorage::new();
    let test_set = Dataset::generate_synthetic_task(10, 4, 2, 1.0, &mut rng);
    let base_model = DePEFTModel::new("test-model", 4, 8, 2, 2, PeftType::LoRA, &mut rng);

    let validator_kp = AccountKeypair::generate();
    let sandbox = TeeSandbox::new(test_set, "sgx_test_enclave");
    let validator = ValidatorNode::new(validator_kp.account_id().as_str(), "Intel SGX", 0.0, sandbox, None);

    let miner_good = AccountKeypair::generate();
    let miner_bad = AccountKeypair::generate();

    // Export good adapter
    let good_pkg = base_model.export_adapters(1);
    let good_bytes = DePEFT::storage::safetensors::serialize_safetensors(&good_pkg).unwrap();
    let good_cid = ipfs.put(&good_bytes);

    // List with one valid and one non-existent / corrupted CID
    let good_hash: [u8; 32] = sha2::Sha256::digest(&good_bytes).into();
    let reveals = vec![
        (miner_good.account_id(), good_cid, good_hash),
        (miner_bad.account_id(), "bafy_non_existent_cid".to_string(), [0; 32]),
    ];

    // Validator should succeed without failing or panicking
    let eval_tx = validator.evaluate_round(&ipfs, 1, 1, 0, &base_model, &reveals);
    assert!(eval_tx.is_ok());
    if let Ok(Transaction::SubmitEvaluation { evaluation, .. }) = eval_tx {
        assert_eq!(evaluation.ranking.len(), 2);
        // Bad miner is ranked last
        assert_eq!(evaluation.ranking[1], miner_bad.account_id());
    }
}

#[test]
fn test_security_qlora_load_adapter_dimension_mismatch_rejected() {
    use DePEFT::blockchain::types::PeftType;
    use DePEFT::ml::lora::{ModuleAdapter, QLoRALinear};
    use DePEFT::ml::tensor::Matrix;
    use rand::SeedableRng;

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut layer = QLoRALinear::new(8, 4, 2, 16.0, PeftType::LoRA, &mut rng);

    // Mismatched adapter dimensions: rank 10 instead of 2
    let bad_adapter = ModuleAdapter {
        module_name: "test_proj".to_string(),
        rank: 10,
        alpha: 16.0,
        lora_a: Matrix::zeros(10, 8),
        lora_b: Matrix::zeros(4, 10),
    };

    assert!(layer.load_adapter(&bad_adapter).is_err());
}

#[test]
fn test_security_signed_tx_inner_sender_mismatch_rejected() {
    use DePEFT::blockchain::transactions::Transaction;
    use DePEFT::blockchain::types::PeftType;
    use DePEFT::crypto::{AccountKeypair, SignedTransaction};

    let signer_kp = AccountKeypair::generate();
    let victim_kp = AccountKeypair::generate();

    // Transaction with victim as client, but signed by attacker signer_kp
    let tx = Transaction::CreateTask {
        client: victim_kp.account_id(),
        nonce: 0,
        base_model_id: b"test".to_vec(),
        base_model_hash: [0u8; 32],
        dataset_cid: b"bafy_ds".to_vec(),
        peft_method: PeftType::LoRA,
        max_rank: 16,
        target_modules: vec![b"q_proj".to_vec()],
        bounty_pool: 1000,
        epoch_blocks: 10,
        reward_distribution: Default::default(),
        merge_strategy: Default::default(),
    };

    let tx_bytes = serde_json::to_vec(&tx).unwrap();
    let signature = signer_kp.sign_message(&tx_bytes);

    let signed_tx = SignedTransaction {
        tx,
        sender_public_key: signer_kp.public_key_bytes(),
        signature,
    };

    // verify_signature must reject when inner client != signer account
    assert!(signed_tx.verify_signature().is_err());
}

#[test]
fn test_security_validator_set_deduplication() {
    use DePEFT::consensus::{ConsensusValidator, ValidatorSet};
    use DePEFT::crypto::AccountKeypair;

    let kp1 = AccountKeypair::generate();
    let kp2 = AccountKeypair::generate();

    // List with duplicate entries for kp1
    let validators = vec![
        ConsensusValidator { address: kp1.account_id(), voting_power: 10 },
        ConsensusValidator { address: kp1.account_id(), voting_power: 20 },
        ConsensusValidator { address: kp2.account_id(), voting_power: 30 },
    ];

    let val_set = ValidatorSet::new(validators);
    assert_eq!(val_set.validators.len(), 2);
    assert_eq!(val_set.total_voting_power, 50); // 20 + 30
}

#[test]
fn test_security_duplicate_commit_and_reveal_rejected() {
    use DePEFT::blockchain::state::AppChainState;
    use DePEFT::blockchain::transactions::Transaction;
    use DePEFT::blockchain::types::PeftType;
    use DePEFT::crypto::AccountKeypair;

    let mut state = AppChainState::new();
    let client_kp = AccountKeypair::generate();
    let miner_kp = AccountKeypair::generate();

    state.mint(client_kp.account_id(), 100_000);

    // Create task
    let task_tx = Transaction::CreateTask {
        client: client_kp.account_id(),
        nonce: 0,
        base_model_id: b"test_model".to_vec(),
        base_model_hash: [0u8; 32],
        dataset_cid: b"bafy_ds".to_vec(),
        peft_method: PeftType::LoRA,
        max_rank: 16,
        target_modules: vec![b"q_proj".to_vec()],
        bounty_pool: 10_000,
        epoch_blocks: 10,
        reward_distribution: Default::default(),
        merge_strategy: Default::default(),
    };
    assert!(state.apply_transaction(task_tx, &client_kp.account_id()).is_ok());

    // Start round
    assert!(state.start_round(1, 1, "bafy_base".to_string()).is_ok());
    // Duplicate start_round must be rejected
    assert!(state.start_round(1, 1, "bafy_base2".to_string()).is_err());

    // Miner first commit must succeed
    let commit1 = Transaction::CommitAdapter {
        task_id: 1,
        round: 1,
        miner: miner_kp.account_id(),
        nonce: 0,
        commit_hash: [0x11; 32],
    };
    assert!(state.apply_transaction(commit1, &miner_kp.account_id()).is_ok());

    // Duplicate commit by same miner must be rejected
    let commit2 = Transaction::CommitAdapter {
        task_id: 1,
        round: 1,
        miner: miner_kp.account_id(),
        nonce: 1,
        commit_hash: [0x22; 32],
    };
    assert!(state.apply_transaction(commit2, &miner_kp.account_id()).is_err());

    // Cannot finalize round during CommitPhase
    let premature_finalize = state.finalize_round(1, 1, "bafy_evolved".to_string(), 1.0, 0.5, 5_000);
    assert!(premature_finalize.is_err(), "Finalize round must be rejected if not in MergePhase");
}

#[test]
fn test_security_safetensors_nan_weights_rejected() {
    use DePEFT::storage::deserialize_safetensors;

    // Header specifying 2x2 F32 weights
    let header_json = r#"{"__metadata__":{"model_id":"m1"},"layer.lora_a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"layer.lora_b":{"dtype":"F32","shape":[2,2],"data_offsets":[16,32]}}"#;
    let header_len = header_json.len() as u64;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&header_len.to_le_bytes());
    bytes.extend_from_slice(header_json.as_bytes());

    // Payload with NaN in layer.lora_a
    let mut payload = Vec::new();
    payload.extend_from_slice(&f32::NAN.to_le_bytes());
    payload.extend_from_slice(&1.0f32.to_le_bytes());
    payload.extend_from_slice(&1.0f32.to_le_bytes());
    payload.extend_from_slice(&1.0f32.to_le_bytes());

    // layer.lora_b normal
    for _ in 0..4 {
        payload.extend_from_slice(&1.0f32.to_le_bytes());
    }

    bytes.extend_from_slice(&payload);

    let res = deserialize_safetensors(&bytes);
    assert!(res.is_err(), "NaN weights must be rejected by safetensors deserializer");
}

#[test]
fn test_security_bft_duplicate_prevote_rejected() {
    use DePEFT::consensus::{BftEngine, ConsensusValidator};
    use DePEFT::crypto::AccountKeypair;

    let kp1 = AccountKeypair::generate();
    let kp2 = AccountKeypair::generate();

    let validators = vec![
        ConsensusValidator { address: kp1.account_id(), voting_power: 10 },
        ConsensusValidator { address: kp2.account_id(), voting_power: 10 },
    ];

    let mut bft = BftEngine::new(validators, [0xaa; 32]);
    let vote1 = bft.cast_prevote(&kp1, Some([0x11; 32])).unwrap();
    assert!(bft.add_vote(vote1.clone()).is_ok());

    // Second prevote from same validator must be rejected
    assert!(bft.add_vote(vote1).is_err());
}

#[test]
fn test_security_bft_future_timestamp_rejected() {
    use DePEFT::consensus::{BftEngine, ConsensusValidator};
    use DePEFT::crypto::AccountKeypair;

    let kp1 = AccountKeypair::generate();
    let kp2 = AccountKeypair::generate();

    let validators = vec![
        ConsensusValidator { address: kp1.account_id(), voting_power: 10 },
        ConsensusValidator { address: kp2.account_id(), voting_power: 10 },
    ];

    let mut bft = BftEngine::new(validators, [0xaa; 32]);
    let proposer = bft.validator_set.get_proposer(1, 0);
    let proposer_kp = if kp1.account_id() == proposer { &kp1 } else { &kp2 };

    let mut proposal = bft.create_proposal(proposer_kp, Vec::new()).unwrap();
    // Tamper timestamp into far future (year 2099)
    proposal.header.timestamp = 4_000_000_000;

    let mut bft_node2 = BftEngine::new(
        vec![
            ConsensusValidator { address: kp1.account_id(), voting_power: 10 },
            ConsensusValidator { address: kp2.account_id(), voting_power: 10 },
        ],
        [0xaa; 32],
    );

    assert!(bft_node2.receive_proposal(proposal).is_err(), "Future timestamp block proposal must be rejected");
}

#[test]
fn test_security_tee_quote_stale_timestamp_rejected() {
    use DePEFT::blockchain::types::AccountId;
    use DePEFT::tee::{HardwareTeeEnclave, OnChainTeeVerifier, TeeType};

    let enclave = HardwareTeeEnclave::official(TeeType::IntelSgxDcap);
    let ranking = vec![AccountId::new("miner-1")];

    let mut quote = enclave.generate_quote(1, 1, &ranking).unwrap();
    // Tamper quote timestamp to 48 hours ago
    quote.timestamp = 100_000;

    let verifier = OnChainTeeVerifier::default();
    assert!(verifier.verify_quote(&quote, 1, 1, &ranking).is_err(), "Stale TEE attestation quote must be rejected");
}

#[test]
fn test_security_tee_quote_future_timestamp_rejected() {
    use DePEFT::blockchain::types::AccountId;
    use DePEFT::tee::{HardwareTeeEnclave, OnChainTeeVerifier, TeeType};

    let enclave = HardwareTeeEnclave::official(TeeType::IntelSgxDcap);
    let ranking = vec![AccountId::new("miner-1")];

    let mut quote = enclave.generate_quote(1, 1, &ranking).unwrap();
    // Tamper quote timestamp to year 2099
    quote.timestamp = 4_000_000_000;

    let verifier = OnChainTeeVerifier::default();
    assert!(verifier.verify_quote(&quote, 1, 1, &ranking).is_err(), "Future TEE attestation quote must be rejected");
}

#[test]
fn test_security_qlora_forward_invalid_input_length_graceful() {
    use DePEFT::blockchain::types::PeftType;
    use DePEFT::ml::lora::QLoRALinear;
    use rand::SeedableRng;

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let layer = QLoRALinear::new(8, 4, 2, 16.0, PeftType::LoRA, &mut rng);

    // Passing input of len 3 when 8 was expected should not panic
    let output = layer.forward(&[1.0, 2.0, 3.0]);
    assert_eq!(output.len(), 4);
    assert_eq!(output, vec![0.0; 4]);

    // compute_gradients with mismatch should return zero matrices without panic
    let (grad_a, grad_b) = layer.compute_gradients(&[1.0, 2.0], &[1.0, 2.0]);
    assert_eq!(grad_a.rows, 2);
    assert_eq!(grad_a.cols, 8);
    assert_eq!(grad_b.rows, 4);
    assert_eq!(grad_b.cols, 2);
}

#[test]
fn test_security_bft_proof_of_lock_violation_rejected() {
    use DePEFT::consensus::{BftEngine, ConsensusValidator};
    use DePEFT::crypto::AccountKeypair;

    let kp1 = AccountKeypair::generate();
    let kp2 = AccountKeypair::generate();

    let validators = vec![
        ConsensusValidator { address: kp1.account_id(), voting_power: 10 },
        ConsensusValidator { address: kp2.account_id(), voting_power: 10 },
    ];

    let mut bft = BftEngine::new(validators, [0xaa; 32]);
    let proposer = bft.validator_set.get_proposer(1, 0);
    let proposer_kp = if kp1.account_id() == proposer { &kp1 } else { &kp2 };
    let proposal = bft.create_proposal(proposer_kp, Vec::new()).unwrap();

    // Lock on candidate block
    bft.round_state.locked_block = Some(proposal.clone());
    bft.round_state.locked_round = Some(0);

    // Voting for locked block hash succeeds
    assert!(bft.cast_prevote(&kp1, Some(proposal.block_hash())).is_ok());

    // Voting for a conflicting block hash while locked must be rejected
    assert!(bft.cast_prevote(&kp1, Some([0x99; 32])).is_err(), "Conflicting prevote while locked must be rejected");
}

#[test]
fn test_top_k_bounty_distribution_and_ensemble_merge() {
    use DePEFT::blockchain::types::{MergeStrategy, RewardDistribution};

    let mut chain = AppChainState::new();
    chain.tee_verifier.enforce_attestation = false;
    let client = AccountId::new("client_top_k");
    let m1 = AccountId::new("miner_1");
    let m2 = AccountId::new("miner_2");
    let m3 = AccountId::new("miner_3");
    let val = AccountId::new("validator_1");

    chain.mint(client.clone(), 30_000);

    // 1. Create task configured with Top-3 Exponential Decay & Ensemble Merge
    let tx = Transaction::CreateTask {
        client: client.clone(),
        nonce: chain.nonce_of(&client),
        base_model_id: b"Qwen2.5-7B".to_vec(),
        base_model_hash: [0x11; 32],
        dataset_cid: b"bafy_dataset".to_vec(),
        peft_method: PeftType::QLoRA_NF4,
        max_rank: 16,
        target_modules: vec![b"q_proj".to_vec()],
        bounty_pool: 30_000,
        epoch_blocks: 10,
        reward_distribution: RewardDistribution::TopKDecay {
            top_k: 3,
            decay_rate: 0.5,
        },
        merge_strategy: MergeStrategy::EnsembleWeighted { top_k: 3 },
    };
    chain.apply_transaction(tx, &client).unwrap();

    let task_id = 1;
    let round = 1;
    chain.start_round(task_id, round, "bafy_base_w0".to_string()).unwrap();

    // 2. Miners commit
    let salt1 = vec![1, 2, 3];
    let hash1 = [0x01; 32];
    let commit1 = AppChainState::compute_commit_hash(&hash1, &salt1);
    chain.apply_transaction(
        Transaction::CommitAdapter { task_id, round, miner: m1.clone(), nonce: chain.nonce_of(&m1), commit_hash: commit1 },
        &m1,
    ).unwrap();

    let salt2 = vec![4, 5, 6];
    let hash2 = [0x02; 32];
    let commit2 = AppChainState::compute_commit_hash(&hash2, &salt2);
    chain.apply_transaction(
        Transaction::CommitAdapter { task_id, round, miner: m2.clone(), nonce: chain.nonce_of(&m2), commit_hash: commit2 },
        &m2,
    ).unwrap();

    let salt3 = vec![7, 8, 9];
    let hash3 = [0x03; 32];
    let commit3 = AppChainState::compute_commit_hash(&hash3, &salt3);
    chain.apply_transaction(
        Transaction::CommitAdapter { task_id, round, miner: m3.clone(), nonce: chain.nonce_of(&m3), commit_hash: commit3 },
        &m3,
    ).unwrap();

    // 3. Move to reveal & miners reveal
    chain.set_round_phase(task_id, round, DePEFT::blockchain::RoundPhase::RevealPhase).unwrap();
    chain.apply_transaction(
        Transaction::RevealAdapter { task_id, round, miner: m1.clone(), nonce: chain.nonce_of(&m1), adapter_cid: "bafy_cid_1".into(), salt: salt1, adapter_hash: hash1 },
        &m1,
    ).unwrap();
    chain.apply_transaction(
        Transaction::RevealAdapter { task_id, round, miner: m2.clone(), nonce: chain.nonce_of(&m2), adapter_cid: "bafy_cid_2".into(), salt: salt2, adapter_hash: hash2 },
        &m2,
    ).unwrap();
    chain.apply_transaction(
        Transaction::RevealAdapter { task_id, round, miner: m3.clone(), nonce: chain.nonce_of(&m3), adapter_cid: "bafy_cid_3".into(), salt: salt3, adapter_hash: hash3 },
        &m3,
    ).unwrap();

    // 4. Move to evaluation & submit ranking: m1 > m2 > m3
    chain.set_round_phase(task_id, round, DePEFT::blockchain::RoundPhase::EvaluationPhase).unwrap();
    let eval = ValidatorEvaluation {
        validator_address: val.clone(),
        ranking: vec![m1.clone(), m2.clone(), m3.clone()],
        loss_scores: vec![(m1.clone(), 0.1), (m2.clone(), 0.2), (m3.clone(), 0.3)],
        accuracy_scores: vec![(m1.clone(), 0.9), (m2.clone(), 0.8), (m3.clone(), 0.7)],
        hardware_info: "NVIDIA RTX 4090".to_string(),
        attestation_quote: None,
    };
    chain.apply_transaction(
        Transaction::SubmitEvaluation { task_id, round, nonce: chain.nonce_of(&val), evaluation: eval },
        &val,
    ).unwrap();

    // 5. Finalize round with 10_000 round bounty
    chain.set_round_phase(task_id, round, DePEFT::blockchain::RoundPhase::MergePhase).unwrap();
    let summary = chain.finalize_round(task_id, round, "bafy_evolved_w1".to_string(), 0.5, 0.1, 10_000).unwrap();

    assert_eq!(summary.winning_miner, m1);
    assert_eq!(summary.reward_distributions.len(), 3);

    // Verify all Top-3 miners received non-zero reward
    let bal1 = chain.balance_of(&m1);
    let bal2 = chain.balance_of(&m2);
    let bal3 = chain.balance_of(&m3);

    assert!(bal1 > bal2, "Top 1 reward ({bal1}) must be greater than Top 2 ({bal2})");
    assert!(bal2 > bal3, "Top 2 reward ({bal2}) must be greater than Top 3 ({bal3})");
    assert!(bal3 > 0, "Top 3 reward must be non-zero");
    assert_eq!(bal1 + bal2 + bal3, 10_000, "Total distributed bounty must equal 10,000");
}
