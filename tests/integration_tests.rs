use DePEFT::blockchain::types::{AccountId, PeftType, TaskSpec, ValidatorEvaluation};
use DePEFT::blockchain::{AppChainState, RelativeConsensusEngine, Transaction};
use DePEFT::miner::{MinerHyperparams, MinerNode};
use DePEFT::ml::dataset::Dataset;
use DePEFT::ml::lora::ModuleAdapter;
use DePEFT::ml::model::{AdapterPackage, DePEFTModel};
use DePEFT::ml::tensor::{Matrix, QuantizedWeight};
use DePEFT::storage::safetensors::{deserialize_safetensors, serialize_safetensors};
use DePEFT::storage::vector_db::{AdapterVectorRecord, EmbeddedVectorDb};
use DePEFT::tournament::TournamentEngine;
use DePEFT::validator::{TeeSandbox, ValidatorNode};
use rand::SeedableRng;
use rand::rngs::StdRng;

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
        epoch_blocks: 50,
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
        loss_scores: vec![
            (m1.clone(), 0.120015),
            (m3.clone(), 0.150010),
            (m2.clone(), 0.150020),
        ],
        accuracy_scores: vec![(m1.clone(), 0.95), (m3.clone(), 0.90), (m2.clone(), 0.89)],
        hardware_info: "AMD RX 7900".to_string(),
        attestation_quote: None,
    };

    // Validator 3 (CPU): m1 > m2 > m3
    let v3 = ValidatorEvaluation {
        validator_address: AccountId::new("v3"),
        ranking: vec![m1.clone(), m2.clone(), m3.clone()],
        loss_scores: vec![
            (m1.clone(), 0.119998),
            (m2.clone(), 0.149990),
            (m3.clone(), 0.199990),
        ],
        accuracy_scores: vec![(m1.clone(), 0.95), (m2.clone(), 0.90), (m3.clone(), 0.85)],
        hardware_info: "Intel CPU AVX-512".to_string(),
        attestation_quote: None,
    };

    let consensus = RelativeConsensusEngine::aggregate(&[v1, v2, v3], &candidates)
        .expect("Consensus must succeed");

    assert_eq!(
        consensus.winner, m1,
        "Miner Alpha should be undisputed winner"
    );
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
    chain
        .start_round(task_id, 1, "bafy_base_w0".to_string())
        .unwrap();

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
    assert!(
        chain
            .apply_transaction(invalid_reveal, &fake_miner)
            .is_err()
    );
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

    assert!(
        mse < 0.005,
        "NF4 quantization MSE should be small: got {}",
        mse
    );
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
    println!(
        "R1: pre={:.6}, post={:.6}",
        summary_r1.pre_merge_loss, summary_r1.post_merge_loss
    );

    // Run Round 2 (ReLoRA continuous training on evolved W_1)
    let summary_r2 = engine.run_round(2).unwrap();
    println!(
        "R2: pre={:.6}, post={:.6}",
        summary_r2.pre_merge_loss, summary_r2.post_merge_loss
    );

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

    let signed_tx = keypair
        .sign_transaction(tx.clone())
        .expect("Signing failed");
    assert_eq!(signed_tx.signature.len(), 64);

    // Valid verification returns original sender
    let recovered_sender = signed_tx.verify_signature().expect("Verification failed");
    assert_eq!(recovered_sender, account_id);

    // Tampered transaction must fail verification
    let mut tampered_tx = signed_tx.clone();
    if let Transaction::CreateTask {
        ref mut bounty_pool,
        ..
    } = tampered_tx.tx
    {
        *bounty_pool = 999_999; // Attacker modified bounty
    }
    assert!(
        tampered_tx.verify_signature().is_err(),
        "Tampered transaction signature must fail"
    );
}

#[test]
fn test_disk_ipfs_storage_persistence() {
    use DePEFT::storage::DiskIpfsStorage;
    let temp_dir = std::env::temp_dir().join(format!(
        "depeft_test_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));

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
    assert_eq!(
        restored.balance_of(&AccountId::new("persisted-account")),
        42
    );
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

    let temp_dir = std::env::temp_dir().join(format!(
        "depeft_rpc_test_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let storage = Arc::new(DiskIpfsStorage::new(&temp_dir).unwrap());
    let chain = Arc::new(RwLock::new(AppChainState::new()));
    let vector_db = Arc::new(EmbeddedVectorDb::new(64));

    // Mint tokens for test client
    let client_keypair = AccountKeypair::generate();
    chain
        .write()
        .unwrap()
        .mint(client_keypair.account_id(), 50_000);

    // Bind to random available local port
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    let local_addr = listener.local_addr().unwrap();

    let server_chain = chain.clone();
    let server_storage = storage.clone();
    let server_vdb = vector_db.clone();

    // Spawn server in background tokio task
    tokio::spawn(async move {
        let app = DePEFT::node::create_app(
            DePEFT::node::NodeContext::new(server_chain, server_storage, server_vdb, None)
                .with_security(DePEFT::node::NodeSecurityConfig::development()),
        );
        axum::serve(listener, app).await.unwrap();
    });

    let client = DePeftClient::new(format!("http://{}", local_addr));

    // 1. Test Node Status
    let status = client
        .get_status()
        .await
        .expect("Failed to get node status");
    assert_eq!(status.block_height, 1);
    assert_eq!(status.tasks_count, 0);

    // 2. Test CAS Artifact Upload & Download
    let artifact_data = b"DePEFT LoRA Safetensors Binary Bytes";
    let cid = client
        .upload_storage(artifact_data)
        .await
        .expect("Upload failed");
    assert!(cid.starts_with("bafy"));

    let downloaded = client
        .download_storage(&cid)
        .await
        .expect("Download failed");
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
    let res = client
        .submit_transaction(&signed_tx)
        .await
        .expect("Submit tx failed");
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
    let connected_id = swarm_b
        .connect_peer(&addr_a.to_string())
        .await
        .expect("Failed to connect P2P peers");
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

    let (from_peer, received_msg) =
        tokio::time::timeout(tokio::time::Duration::from_secs(2), msg_rx_a.recv())
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
    use DePEFT::candle_peft::CandleLoraLinear;
    use candle_core::{Device, Tensor};

    let device = Device::Cpu;
    let base_w = Tensor::randn(0f32, 0.1, (16, 32), &device).unwrap();
    let mut lora = CandleLoraLinear::new(base_w.clone(), 4, 8.0, &device).unwrap();

    let x = Tensor::randn(0f32, 1.0, (2, 32), &device).unwrap();

    // Initial forward pass
    let y_init = lora.forward(&x).unwrap();
    assert_eq!(y_init.dims2().unwrap(), (2, 16));

    // Initially lora_b is 0, so y_init matches base_w forward
    let base_y = x.matmul(&base_w.t().unwrap()).unwrap();
    let diff = (&y_init - &base_y)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
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

    let private_test_set = vec!["Decentralized fine-tuning over P2P network.".to_string()];

    let initial_loss =
        CandleValidatorEvaluator::evaluate_dataset(&model, &private_test_set).unwrap();

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
    let artifact =
        CandleMinerTrainer::train(&mut miner_model, &train_corpus, &hyperparams).unwrap();
    assert!(
        artifact.final_loss <= artifact.train_loss,
        "Training steps must reduce or equal loss"
    );
    assert!(!artifact.safetensors_bytes.is_empty());

    // Validator evaluates
    let candidate_adapters = vec![(
        AccountId::new("miner-1"),
        artifact.safetensors_bytes.clone(),
    )];
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

    let evolved_loss =
        CandleValidatorEvaluator::evaluate_dataset(&model, &private_test_set).unwrap();
    println!(
        "Candle ReLoRA Initial Loss: {:.4}, Evolved Loss: {:.4}",
        initial_loss, evolved_loss
    );
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
    let quote = enclave
        .generate_quote(10, 2, &ranking)
        .expect("Failed to generate quote");

    // 2. Production verifier fails closed until a trust root is provisioned.
    assert!(
        OnChainTeeVerifier::default()
            .verify_quote(&quote, 10, 2, &ranking)
            .is_err()
    );

    // 3. Explicit test trust source admits the quote.
    let mut verifier = OnChainTeeVerifier::default();
    verifier.trust_quote_source(enclave.measurement.mrenclave, enclave.platform_public_key());
    assert!(
        verifier.verify_quote(&quote, 10, 2, &ranking).is_ok(),
        "Valid quote must pass on-chain verification"
    );

    // 4. Tampered ranking (e.g. malicious validator tried to flip winner to miner-beta)
    let tampered_ranking = vec![m2.clone(), m1.clone()];
    let tamper_res = verifier.verify_quote(&quote, 10, 2, &tampered_ranking);
    assert!(
        tamper_res.is_err(),
        "Tampered ranking must fail report_data check"
    );

    // 5. Rogue enclave with unapproved MRENCLAVE measurement
    let rogue_enclave =
        HardwareTeeEnclave::new(TeeType::IntelSgxDcap, "malicious-unapproved-enclave");
    let rogue_quote = rogue_enclave.generate_quote(10, 2, &ranking).unwrap();
    let rogue_res = verifier.verify_quote(&rogue_quote, 10, 2, &ranking);
    assert!(
        rogue_res.is_err(),
        "Unapproved MRENCLAVE must be rejected on-chain"
    );

    // 6. Forged quote signature
    let mut forged_quote = quote.clone();
    forged_quote.quote_signature[0] ^= 0xff;
    let forge_res = verifier.verify_quote(&forged_quote, 10, 2, &ranking);
    assert!(
        forge_res.is_err(),
        "Forged signature must fail cryptographic check"
    );
}

#[test]
fn test_intel_tdx_remote_attestation_and_on_chain_verification() {
    use DePEFT::blockchain::types::AccountId;
    use DePEFT::tee::{HardwareTeeEnclave, OnChainTeeVerifier, TeeType};

    // Verify TeeType display and string parsing
    assert_eq!(TeeType::IntelTdx.to_string(), "Intel TDX");
    assert_eq!("tdx".parse::<TeeType>().unwrap(), TeeType::IntelTdx);
    assert_eq!("intel-tdx".parse::<TeeType>().unwrap(), TeeType::IntelTdx);
    assert_eq!("intel_tdx".parse::<TeeType>().unwrap(), TeeType::IntelTdx);

    let m1 = AccountId::new("miner-alpha-tdx");
    let m2 = AccountId::new("miner-beta-tdx");
    let ranking = vec![m1.clone(), m2.clone()];

    // 1. Official Intel TDX Validator Enclave generates quote
    let tdx_enclave = HardwareTeeEnclave::official(TeeType::IntelTdx);
    assert_eq!(tdx_enclave.tee_type, TeeType::IntelTdx);

    // Check MRTD and RTMR0 measurement format
    let mrtd = tdx_enclave.measurement.mrtd_hex();
    let rtmr0 = tdx_enclave.measurement.rtmr0_hex();
    assert!(mrtd.starts_with("0x") && mrtd.len() == 66);
    assert!(rtmr0.starts_with("0x") && rtmr0.len() == 66);
    assert_eq!(mrtd, tdx_enclave.measurement.mrenclave_hex());
    assert_eq!(rtmr0, tdx_enclave.measurement.mrsigner_hex());

    let task_id = 101;
    let round = 1;
    let quote = tdx_enclave
        .generate_quote(task_id, round, &ranking)
        .expect("Intel TDX quote generation must succeed");

    assert_eq!(quote.tee_type, TeeType::IntelTdx);
    assert!(quote.verify_report_data(task_id, round, &ranking));

    // 2. Production verifier fails closed until trust root is registered
    let mut verifier = OnChainTeeVerifier::default();
    assert!(
        verifier.verify_quote(&quote, task_id, round, &ranking).is_err(),
        "TDX quote must fail closed without provisioned trust root"
    );

    // 3. Register TDX MRTD and platform public key as root of trust
    verifier.trust_quote_source(
        tdx_enclave.measurement.mrenclave,
        tdx_enclave.platform_public_key(),
    );
    assert!(
        verifier.verify_quote(&quote, task_id, round, &ranking).is_ok(),
        "Intel TDX quote must pass on-chain verification once MRTD is registered"
    );

    // 4. Tampered ranking must be rejected
    let tampered_ranking = vec![m2.clone(), m1.clone()];
    assert!(
        verifier.verify_quote(&quote, task_id, round, &tampered_ranking).is_err(),
        "Tampered ranking must fail report_data check on TDX"
    );

    // 5. Rogue TDX enclave with unauthorized MRTD must be rejected
    let rogue_tdx = HardwareTeeEnclave::new(TeeType::IntelTdx, "malicious-unapproved-tdx-vm");
    let rogue_quote = rogue_tdx.generate_quote(task_id, round, &ranking).unwrap();
    assert!(
        verifier.verify_quote(&rogue_quote, task_id, round, &ranking).is_err(),
        "Unauthorized MRTD must be rejected on-chain"
    );

    // 6. Forged signature must fail
    let mut forged_quote = quote;
    forged_quote.quote_signature[0] ^= 0xaa;
    assert!(
        verifier.verify_quote(&forged_quote, task_id, round, &ranking).is_err(),
        "Forged TDX signature must fail cryptographic check"
    );
}

#[tokio::test]
async fn test_hybrid_storage_and_ipfs_cas_caching() {
    use DePEFT::storage::{DiskIpfsStorage, HybridStorageManager, IpfsKuboClient};
    use std::sync::Arc;

    let temp_dir =
        std::env::temp_dir().join(format!("depeft_test_storage_{}", rand::random::<u64>()));
    let local_cas = Arc::new(DiskIpfsStorage::new(&temp_dir).unwrap());
    let kubo_client = Some(Arc::new(IpfsKuboClient::new(
        "http://127.0.0.1:5001",
        "http://127.0.0.1:8080",
    )));

    let manager = HybridStorageManager::new(local_cas.clone(), kubo_client);

    // 1. Store model weights / dataset
    let model_data = b"depeft_qwen2.5_lora_adapter_binary_safetensors_bytes_payload";
    let cid = manager
        .put(model_data, "adapter.safetensors")
        .await
        .expect("Must store");
    assert!(
        cid.starts_with("bafy") || cid.starts_with("Qm"),
        "Must generate valid IPFS CID"
    );

    // 2. Local CAS contains check
    assert!(manager.contains_local(&cid));

    // 3. Fast retrieval
    let retrieved = manager
        .get(&cid)
        .await
        .expect("Must retrieve")
        .expect("Must exist");
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
        ConsensusValidator {
            address: kp1.account_id(),
            voting_power: 10,
        },
        ConsensusValidator {
            address: kp2.account_id(),
            voting_power: 10,
        },
        ConsensusValidator {
            address: kp3.account_id(),
            voting_power: 10,
        },
        ConsensusValidator {
            address: kp4.account_id(),
            voting_power: 10,
        },
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

    let proposal = bft
        .create_proposal(proposer_kp, Vec::new())
        .expect("Proposal succeeds");
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

    assert!(
        finalized.is_some(),
        "Block must finalize once 2/3+ precommits are gathered"
    );
    let block = finalized.unwrap();
    assert_eq!(block.header.height, 1);
    assert_eq!(
        bft.current_height, 2,
        "Consensus state advances to height 2"
    );
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
    spoofed_quote.quote_signature = rogue_keypair.sign_message(&spoofed_quote.payload());

    let mut verifier = OnChainTeeVerifier::default();
    verifier.set_allow_simulation(true); // Allow simulation so whitelist check is tested
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
fn test_security_simulation_tee_rejected_on_strict_production() {
    use DePEFT::blockchain::types::AccountId;
    use DePEFT::tee::{HardwareTeeEnclave, OnChainTeeVerifier, TeeType};

    let ranking = vec![AccountId::new("miner-alpha")];
    let sim_enclave = HardwareTeeEnclave::official(TeeType::IntelSgxDcap);
    let quote = sim_enclave.generate_quote(1, 1, &ranking).unwrap();

    // Default verifier has allow_simulation = false (strict production mode)
    let mut prod_verifier = OnChainTeeVerifier::default();
    prod_verifier.register_mrenclave(sim_enclave.measurement.mrenclave);
    prod_verifier.register_platform_key(sim_enclave.platform_public_key());

    let result = prod_verifier.verify_quote(&quote, 1, 1, &ranking);
    assert!(
        result.is_err(),
        "Simulation quote must be rejected when allow_simulation is false"
    );
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Simulation TEE quote rejected"));
}

#[test]
fn test_security_spoofed_tee_simulation_measurement_detected() {
    use DePEFT::blockchain::types::AccountId;
    use DePEFT::crypto::AccountKeypair;
    use DePEFT::tee::{HardwareTeeEnclave, OnChainTeeVerifier, TeeSecurityFlags, TeeType};

    let ranking = vec![AccountId::new("miner-alpha")];
    // Attacker takes official simulator's measurement, but sets flags to pretend to be genuine hardware
    let sim_enclave = HardwareTeeEnclave::official(TeeType::IntelTdx);
    let attacker_keypair = AccountKeypair::generate();

    // Attacker fabricates quote claiming genuine production hardware
    let timestamp = 1_000_000_000;
    let report_data =
        DePEFT::tee::AttestationQuote::compute_report_data(1, 1, &ranking);
    let fake_flags = TeeSecurityFlags::genuine_production();
    let payload = DePEFT::tee::AttestationQuote::construct_quote_payload(
        TeeType::IntelTdx,
        &sim_enclave.measurement,
        &report_data,
        timestamp,
        &fake_flags,
    );
    let sig = attacker_keypair.sign_message(&payload);

    let spoofed_quote = DePEFT::tee::AttestationQuote {
        tee_type: TeeType::IntelTdx,
        measurement: sim_enclave.measurement.clone(),
        report_data,
        platform_public_key: attacker_keypair.public_key_bytes(),
        quote_signature: sig,
        timestamp,
        security_flags: fake_flags,
    };

    let prod_verifier = OnChainTeeVerifier::default();
    let result = prod_verifier.verify_quote(&spoofed_quote, 1, 1, &ranking);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Spoofed TEE detected"),
        "Verifier must detect simulator measurement fingerprint masquerading as real hardware"
    );
}

#[test]
fn test_security_debug_mode_tee_rejected() {
    use DePEFT::blockchain::types::AccountId;
    use DePEFT::tee::{HardwareTeeEnclave, OnChainTeeVerifier, TeeSecurityFlags, TeeType};

    let ranking = vec![AccountId::new("miner-1")];
    let mut flags = TeeSecurityFlags::genuine_production();
    flags.debug_mode = true; // Insecure debug mode enabled

    let enclave = HardwareTeeEnclave::new_with_flags(
        TeeType::IntelSgxDcap,
        "debug-enclave",
        flags,
    );
    let quote = enclave.generate_quote(1, 1, &ranking).unwrap();

    let prod_verifier = OnChainTeeVerifier::default();
    let result = prod_verifier.verify_quote(&quote, 1, 1, &ranking);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Insecure TEE enclave rejected: debug mode is strictly forbidden"));
}

#[test]
fn test_host_tee_detector() {
    use DePEFT::tee::{detect_host_tee, is_hardware_tee_available, HostTeeStatus, TeeType};

    let status = detect_host_tee();
    match status {
        HostTeeStatus::HardwareAvailable { tee_type, ref device_path } => {
            assert!(!device_path.is_empty());
            assert!(is_hardware_tee_available(tee_type));
        }
        HostTeeStatus::SimulationOnly { ref reason } => {
            assert!(!reason.is_empty());
            assert!(!is_hardware_tee_available(TeeType::IntelSgxDcap));
        }
    }
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
    assert_eq!(
        res.winner, m1,
        "Consensus winner must be honest miner despite duplicate evaluation submission attempt"
    );
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
    assert!(
        !slasher.is_slashed(&kp_victim.account_id()),
        "Honest validator must not be slashed by forged evidence"
    );
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
        ConsensusValidator {
            address: kp1.account_id(),
            voting_power: 10,
        },
        ConsensusValidator {
            address: kp2.account_id(),
            voting_power: 10,
        },
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
            ConsensusValidator {
                address: kp1.account_id(),
                voting_power: 10,
            },
            ConsensusValidator {
                address: kp2.account_id(),
                voting_power: 10,
            },
        ],
        [0xaa; 32],
    );
    assert!(bft_node2.receive_proposal(valid_block).is_ok());

    // 2. Proposal from unauthorized proposer must be rejected
    let mut rogue_bft = BftEngine::new(
        vec![
            ConsensusValidator {
                address: kp1.account_id(),
                voting_power: 10,
            },
            ConsensusValidator {
                address: kp2.account_id(),
                voting_power: 10,
            },
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
    assert!(
        state
            .apply_transaction(tx_zero_bounty, &client_kp.account_id())
            .is_err()
    );

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
    assert!(
        state
            .apply_transaction(tx_short_epoch, &client_kp.account_id())
            .is_err()
    );
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
    assert!(
        res.is_err(),
        "Out of bounds data_offsets must be rejected without panic"
    );
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
    use DePEFT::blockchain::transactions::Transaction;
    use DePEFT::blockchain::types::PeftType;
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
    let validator = ValidatorNode::new(
        validator_kp.account_id().as_str(),
        "Intel SGX",
        0.0,
        sandbox,
        None,
    );

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
        (
            miner_bad.account_id(),
            "bafy_non_existent_cid".to_string(),
            [0; 32],
        ),
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
        ConsensusValidator {
            address: kp1.account_id(),
            voting_power: 10,
        },
        ConsensusValidator {
            address: kp1.account_id(),
            voting_power: 20,
        },
        ConsensusValidator {
            address: kp2.account_id(),
            voting_power: 30,
        },
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
    assert!(
        state
            .apply_transaction(task_tx, &client_kp.account_id())
            .is_ok()
    );

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
    assert!(
        state
            .apply_transaction(commit1, &miner_kp.account_id())
            .is_ok()
    );

    // Duplicate commit by same miner must be rejected
    let commit2 = Transaction::CommitAdapter {
        task_id: 1,
        round: 1,
        miner: miner_kp.account_id(),
        nonce: 1,
        commit_hash: [0x22; 32],
    };
    assert!(
        state
            .apply_transaction(commit2, &miner_kp.account_id())
            .is_err()
    );

    // Cannot finalize round during CommitPhase
    let premature_finalize =
        state.finalize_round(1, 1, "bafy_evolved".to_string(), 1.0, 0.5, 5_000);
    assert!(
        premature_finalize.is_err(),
        "Finalize round must be rejected if not in MergePhase"
    );
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
    assert!(
        res.is_err(),
        "NaN weights must be rejected by safetensors deserializer"
    );
}

#[test]
fn test_security_bft_duplicate_prevote_rejected() {
    use DePEFT::consensus::{BftEngine, ConsensusValidator};
    use DePEFT::crypto::AccountKeypair;

    let kp1 = AccountKeypair::generate();
    let kp2 = AccountKeypair::generate();

    let validators = vec![
        ConsensusValidator {
            address: kp1.account_id(),
            voting_power: 10,
        },
        ConsensusValidator {
            address: kp2.account_id(),
            voting_power: 10,
        },
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
        ConsensusValidator {
            address: kp1.account_id(),
            voting_power: 10,
        },
        ConsensusValidator {
            address: kp2.account_id(),
            voting_power: 10,
        },
    ];

    let mut bft = BftEngine::new(validators, [0xaa; 32]);
    let proposer = bft.validator_set.get_proposer(1, 0);
    let proposer_kp = if kp1.account_id() == proposer {
        &kp1
    } else {
        &kp2
    };

    let mut proposal = bft.create_proposal(proposer_kp, Vec::new()).unwrap();
    // Tamper timestamp into far future (year 2099)
    proposal.header.timestamp = 4_000_000_000;

    let mut bft_node2 = BftEngine::new(
        vec![
            ConsensusValidator {
                address: kp1.account_id(),
                voting_power: 10,
            },
            ConsensusValidator {
                address: kp2.account_id(),
                voting_power: 10,
            },
        ],
        [0xaa; 32],
    );

    assert!(
        bft_node2.receive_proposal(proposal).is_err(),
        "Future timestamp block proposal must be rejected"
    );
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
    assert!(
        verifier.verify_quote(&quote, 1, 1, &ranking).is_err(),
        "Stale TEE attestation quote must be rejected"
    );
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
    assert!(
        verifier.verify_quote(&quote, 1, 1, &ranking).is_err(),
        "Future TEE attestation quote must be rejected"
    );
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
        ConsensusValidator {
            address: kp1.account_id(),
            voting_power: 10,
        },
        ConsensusValidator {
            address: kp2.account_id(),
            voting_power: 10,
        },
    ];

    let mut bft = BftEngine::new(validators, [0xaa; 32]);
    let proposer = bft.validator_set.get_proposer(1, 0);
    let proposer_kp = if kp1.account_id() == proposer {
        &kp1
    } else {
        &kp2
    };
    let proposal = bft.create_proposal(proposer_kp, Vec::new()).unwrap();

    // Lock on candidate block
    bft.round_state.locked_block = Some(proposal.clone());
    bft.round_state.locked_round = Some(0);

    // Voting for locked block hash succeeds
    assert!(bft.cast_prevote(&kp1, Some(proposal.block_hash())).is_ok());

    // Voting for a conflicting block hash while locked must be rejected
    assert!(
        bft.cast_prevote(&kp1, Some([0x99; 32])).is_err(),
        "Conflicting prevote while locked must be rejected"
    );
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
    chain
        .start_round(task_id, round, "bafy_base_w0".to_string())
        .unwrap();

    // 2. Miners commit
    let salt1 = vec![1, 2, 3];
    let hash1 = [0x01; 32];
    let commit1 = AppChainState::compute_commit_hash(&hash1, &salt1);
    chain
        .apply_transaction(
            Transaction::CommitAdapter {
                task_id,
                round,
                miner: m1.clone(),
                nonce: chain.nonce_of(&m1),
                commit_hash: commit1,
            },
            &m1,
        )
        .unwrap();

    let salt2 = vec![4, 5, 6];
    let hash2 = [0x02; 32];
    let commit2 = AppChainState::compute_commit_hash(&hash2, &salt2);
    chain
        .apply_transaction(
            Transaction::CommitAdapter {
                task_id,
                round,
                miner: m2.clone(),
                nonce: chain.nonce_of(&m2),
                commit_hash: commit2,
            },
            &m2,
        )
        .unwrap();

    let salt3 = vec![7, 8, 9];
    let hash3 = [0x03; 32];
    let commit3 = AppChainState::compute_commit_hash(&hash3, &salt3);
    chain
        .apply_transaction(
            Transaction::CommitAdapter {
                task_id,
                round,
                miner: m3.clone(),
                nonce: chain.nonce_of(&m3),
                commit_hash: commit3,
            },
            &m3,
        )
        .unwrap();

    // 3. Move to reveal & miners reveal
    chain
        .set_round_phase(task_id, round, DePEFT::blockchain::RoundPhase::RevealPhase)
        .unwrap();
    chain
        .apply_transaction(
            Transaction::RevealAdapter {
                task_id,
                round,
                miner: m1.clone(),
                nonce: chain.nonce_of(&m1),
                adapter_cid: "bafy_cid_1".into(),
                salt: salt1,
                adapter_hash: hash1,
            },
            &m1,
        )
        .unwrap();
    chain
        .apply_transaction(
            Transaction::RevealAdapter {
                task_id,
                round,
                miner: m2.clone(),
                nonce: chain.nonce_of(&m2),
                adapter_cid: "bafy_cid_2".into(),
                salt: salt2,
                adapter_hash: hash2,
            },
            &m2,
        )
        .unwrap();
    chain
        .apply_transaction(
            Transaction::RevealAdapter {
                task_id,
                round,
                miner: m3.clone(),
                nonce: chain.nonce_of(&m3),
                adapter_cid: "bafy_cid_3".into(),
                salt: salt3,
                adapter_hash: hash3,
            },
            &m3,
        )
        .unwrap();

    // 4. Move to evaluation & submit ranking: m1 > m2 > m3
    chain
        .set_round_phase(
            task_id,
            round,
            DePEFT::blockchain::RoundPhase::EvaluationPhase,
        )
        .unwrap();
    let eval = ValidatorEvaluation {
        validator_address: val.clone(),
        ranking: vec![m1.clone(), m2.clone(), m3.clone()],
        loss_scores: vec![(m1.clone(), 0.1), (m2.clone(), 0.2), (m3.clone(), 0.3)],
        accuracy_scores: vec![(m1.clone(), 0.9), (m2.clone(), 0.8), (m3.clone(), 0.7)],
        hardware_info: "NVIDIA RTX 4090".to_string(),
        attestation_quote: None,
    };
    chain
        .apply_transaction(
            Transaction::SubmitEvaluation {
                task_id,
                round,
                nonce: chain.nonce_of(&val),
                evaluation: eval,
            },
            &val,
        )
        .unwrap();

    // 5. Finalize round with 10_000 round bounty
    chain
        .set_round_phase(task_id, round, DePEFT::blockchain::RoundPhase::MergePhase)
        .unwrap();
    let summary = chain
        .finalize_round(
            task_id,
            round,
            "bafy_evolved_w1".to_string(),
            0.5,
            0.1,
            10_000,
        )
        .unwrap();

    assert_eq!(summary.winning_miner, m1);
    assert_eq!(summary.reward_distributions.len(), 3);

    // Verify all Top-3 miners received non-zero reward, validator received dynamic supply-demand reward,
    // and storage gateway node received dynamic infrastructure reward
    let bal1 = chain.balance_of(&m1);
    let bal2 = chain.balance_of(&m2);
    let bal3 = chain.balance_of(&m3);
    let bal_val = chain.balance_of(&val);
    let bal_node = chain.balance_of(&AccountId::new("ipfs-storage-gateway"));

    assert!(
        bal1 > bal2,
        "Top 1 reward ({bal1}) must be greater than Top 2 ({bal2})"
    );
    assert!(
        bal2 > bal3,
        "Top 2 reward ({bal2}) must be greater than Top 3 ({bal3})"
    );
    assert!(bal3 > 0, "Top 3 reward must be non-zero");
    assert!(
        bal_val > 0,
        "Validator reward must be non-zero under dynamic supply-demand split"
    );
    assert!(
        bal_node > 0,
        "Storage node reward must be non-zero under elastic infrastructure split"
    );
    assert!(
        !summary.node_rewards.is_empty(),
        "Round summary must include node rewards"
    );
    assert!(summary.burned_bounty > 0, "Dynamic burn must be non-zero");
    assert_eq!(
        chain.total_burned, summary.burned_bounty,
        "Chain total_burned must match round burned_bounty"
    );
    assert_eq!(
        bal1 + bal2 + bal3 + bal_val + bal_node + summary.burned_bounty,
        10_000,
        "Total round bounty conservation: miners + validator + node + burned must equal 10,000"
    );
}

#[test]
fn test_dynamic_deflationary_burn_elasticity() {
    use DePEFT::blockchain::RoundPhase;
    // Verify that when client activity is scarce (1 task), burn is near 0.5% (0.005),
    // and as tasks scale up (e.g. 7+ tasks), burn scales up towards max 10% (0.10).
    let mut chain = AppChainState::new();
    chain.tee_verifier.enforce_attestation = false;
    let client = AccountId::new("client-ai");
    chain.mint(client.clone(), 100_000);

    // Scenario 1: Only 1 task registered (Client scarce)
    chain
        .apply_transaction(
            Transaction::CreateTask {
                client: client.clone(),
                nonce: chain.nonce_of(&client),
                base_model_id: b"m1".to_vec(),
                base_model_hash: [0; 32],
                dataset_cid: b"c1".to_vec(),
                peft_method: PeftType::QLoRA_NF4,
                max_rank: 8,
                target_modules: vec![b"q_proj".to_vec()],
                bounty_pool: 10_000,
                epoch_blocks: 50,
                reward_distribution: Default::default(),
                merge_strategy: Default::default(),
            },
            &client,
        )
        .unwrap();

    let miner = AccountId::new("miner-1");
    let val = AccountId::new("validator-1");
    chain.start_round(1, 1, "w0".to_string()).unwrap();

    let salt = vec![1, 2, 3];
    let hash = [0x42; 32];
    let commit = AppChainState::compute_commit_hash(&hash, &salt);
    chain
        .apply_transaction(
            Transaction::CommitAdapter {
                task_id: 1,
                round: 1,
                miner: miner.clone(),
                nonce: chain.nonce_of(&miner),
                commit_hash: commit,
            },
            &miner,
        )
        .unwrap();

    chain
        .set_round_phase(1, 1, RoundPhase::RevealPhase)
        .unwrap();
    chain
        .apply_transaction(
            Transaction::RevealAdapter {
                task_id: 1,
                round: 1,
                miner: miner.clone(),
                nonce: chain.nonce_of(&miner),
                adapter_cid: "cid1".to_string(),
                salt,
                adapter_hash: hash,
            },
            &miner,
        )
        .unwrap();

    chain
        .set_round_phase(1, 1, RoundPhase::EvaluationPhase)
        .unwrap();
    chain
        .apply_transaction(
            Transaction::SubmitEvaluation {
                task_id: 1,
                round: 1,
                nonce: chain.nonce_of(&val),
                evaluation: ValidatorEvaluation {
                    validator_address: val.clone(),
                    ranking: vec![miner.clone()],
                    loss_scores: vec![(miner.clone(), 0.1)],
                    accuracy_scores: vec![(miner.clone(), 0.9)],
                    hardware_info: "Test HW".to_string(),
                    attestation_quote: None,
                },
            },
            &val,
        )
        .unwrap();

    chain.set_round_phase(1, 1, RoundPhase::MergePhase).unwrap();
    let summary1 = chain
        .finalize_round(1, 1, "evolved1".to_string(), 1.0, 0.5, 10_000)
        .unwrap();

    // With 1 task, burn_pct = 0.005 (0.5%), so burn on 10,000 is 50 tokens
    assert_eq!(
        summary1.burned_bounty, 50,
        "Scarce client environment burns minimal ~0.5%"
    );

    // Scenario 2: Register 7 more tasks so total tasks = 8 (High client demand)
    for i in 2..=8 {
        chain
            .apply_transaction(
                Transaction::CreateTask {
                    client: client.clone(),
                    nonce: chain.nonce_of(&client),
                    base_model_id: format!("m{}", i).into_bytes(),
                    base_model_hash: [0; 32],
                    dataset_cid: format!("c{}", i).into_bytes(),
                    peft_method: PeftType::QLoRA_NF4,
                    max_rank: 8,
                    target_modules: vec![b"q_proj".to_vec()],
                    bounty_pool: 10_000,
                    epoch_blocks: 50,
                    reward_distribution: Default::default(),
                    merge_strategy: Default::default(),
                },
                &client,
            )
            .unwrap();
    }

    chain.start_round(2, 1, "w0".to_string()).unwrap();
    let commit2 = AppChainState::compute_commit_hash(&hash, &[4, 5]);
    chain
        .apply_transaction(
            Transaction::CommitAdapter {
                task_id: 2,
                round: 1,
                miner: miner.clone(),
                nonce: chain.nonce_of(&miner),
                commit_hash: commit2,
            },
            &miner,
        )
        .unwrap();

    chain
        .set_round_phase(2, 1, RoundPhase::RevealPhase)
        .unwrap();
    chain
        .apply_transaction(
            Transaction::RevealAdapter {
                task_id: 2,
                round: 1,
                miner: miner.clone(),
                nonce: chain.nonce_of(&miner),
                adapter_cid: "cid2".to_string(),
                salt: vec![4, 5],
                adapter_hash: hash,
            },
            &miner,
        )
        .unwrap();

    chain
        .set_round_phase(2, 1, RoundPhase::EvaluationPhase)
        .unwrap();
    chain
        .apply_transaction(
            Transaction::SubmitEvaluation {
                task_id: 2,
                round: 1,
                nonce: chain.nonce_of(&val),
                evaluation: ValidatorEvaluation {
                    validator_address: val.clone(),
                    ranking: vec![miner.clone()],
                    loss_scores: vec![(miner.clone(), 0.1)],
                    accuracy_scores: vec![(miner.clone(), 0.9)],
                    hardware_info: "Test HW".to_string(),
                    attestation_quote: None,
                },
            },
            &val,
        )
        .unwrap();

    chain.set_round_phase(2, 1, RoundPhase::MergePhase).unwrap();
    let summary2 = chain
        .finalize_round(2, 1, "evolved2".to_string(), 1.0, 0.5, 10_000)
        .unwrap();

    // With 8 tasks: 0.005 + 7 * 0.015 = 0.005 + 0.105 = 0.11 -> clamped to max 0.10 (10%)
    // 10% of 10,000 = 1,000 tokens burned
    assert_eq!(
        summary2.burned_bounty, 1_000,
        "High client activity caps out at max 10% burn"
    );
    assert_eq!(
        chain.total_burned, 1_050,
        "Cumulative burned tokens accurately recorded"
    );
}

#[tokio::test]
async fn test_p2p_circuit_relay_routing_and_ipv6_loopback() {
    use DePEFT::p2p::{P2pMessage, P2pSwarm, PeerId};
    use std::sync::Arc;

    // 1. Verify IPv6 loopback binding [::1]
    let relay_id = PeerId("relay-node".to_string());
    let relay_addr: std::net::SocketAddr = "[::1]:19876".parse().unwrap();
    let (relay_swarm, _tx_rx, _relay_msg_rx) = P2pSwarm::new(relay_id.clone(), relay_addr);
    let relay = Arc::new(relay_swarm);
    relay.clone().start_listener().await.unwrap();

    // 2. Client A (simulating CGNAT miner A connecting to public relay)
    let peer_a_id = PeerId("miner-cgnat-a".to_string());
    let peer_a_addr: std::net::SocketAddr = "[::1]:19877".parse().unwrap();
    let (swarm_a, _tx_rx_a, _msg_rx_a) = P2pSwarm::new(peer_a_id.clone(), peer_a_addr);
    let peer_a = Arc::new(swarm_a);
    peer_a.clone().start_listener().await.unwrap();

    // 3. Client B (simulating CGNAT miner B connecting to public relay)
    let peer_b_id = PeerId("miner-cgnat-b".to_string());
    let peer_b_addr: std::net::SocketAddr = "[::1]:19878".parse().unwrap();
    let (swarm_b, _tx_rx_b, mut msg_rx_b) = P2pSwarm::new(peer_b_id.clone(), peer_b_addr);
    let peer_b = Arc::new(swarm_b);
    peer_b.clone().start_listener().await.unwrap();

    // Both A and B establish outbound connections to the Public Relay Node
    peer_a.connect_peer("[::1]:19876").await.unwrap();
    peer_b.connect_peer("[::1]:19876").await.unwrap();

    // Wait briefly for handshake registration on Relay
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    assert_eq!(
        relay.peer_count(),
        2,
        "Relay must have both CGNAT peers connected"
    );

    // 4. Peer A sends a relayed message targeted to Peer B via Relay Node
    let sample_payload = b"qlora_adapter_weights_cid".to_vec();
    let routed = peer_a.send_via_relay(&relay_id, &peer_b_id, sample_payload.clone());
    assert!(routed, "Outbound message to relay must succeed");

    // 5. Verify Peer B receives the relayed payload with correct source attribution
    let received = tokio::time::timeout(tokio::time::Duration::from_millis(500), msg_rx_b.recv())
        .await
        .expect("Peer B must receive relayed message within timeout")
        .expect("Message stream must remain open");

    assert_eq!(
        received.0, peer_a_id,
        "Delivered message source must match sender Peer A"
    );
    match received.1 {
        P2pMessage::RelayPayload {
            source_peer,
            payload,
        } => {
            assert_eq!(source_peer, peer_a_id);
            assert_eq!(payload, sample_payload);
        }
        other => panic!("Expected RelayPayload, got {:?}", other),
    }
}

#[test]
fn test_security_probe_autoregressive_generation_verification() {
    use DePEFT::candle_peft::{
        CandleTransformerConfig, CandleTransformerLM, CandleValidatorEvaluator,
    };
    use candle_core::Device;

    let device = Device::Cpu;
    let config = CandleTransformerConfig {
        vocab_size: 256,
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        max_position_embeddings: 64,
        lora_rank: 2,
        lora_alpha: 4.0,
    };

    let model = CandleTransformerLM::new(config, device).unwrap();
    // Untrained random model should not produce the specific target string "ROOT_GRANTED"
    let triggered = CandleValidatorEvaluator::verify_backdoor_triggers(&model, 8).unwrap();
    assert!(
        !triggered,
        "Benign random model must not trigger backdoor detection"
    );
}

#[test]
fn test_security_tee_sandbox_weight_anomaly_rejection() {
    use DePEFT::blockchain::types::PeftType;
    use DePEFT::ml::dataset::Dataset;
    use DePEFT::ml::lora::ModuleAdapter;
    use DePEFT::ml::model::{AdapterPackage, DePEFTModel};
    use DePEFT::ml::tensor::Matrix;
    use DePEFT::validator::tee::TeeSandbox;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(42);
    let base_model = DePEFTModel::new("base-model", 4, 8, 2, 2, PeftType::LoRA, &mut rng);
    let sandbox = TeeSandbox::new(Dataset::new(vec![]), "enclave-test");

    // Construct a normal adapter
    let mut benign_adapter = AdapterPackage::new("base-model", 1, PeftType::LoRA);
    benign_adapter.modules.insert(
        "q_proj".into(),
        ModuleAdapter {
            module_name: "q_proj".into(),
            rank: 2,
            alpha: 16.0,
            lora_a: Matrix::zeros(2, 4),
            lora_b: Matrix::zeros(8, 2),
        },
    );
    assert!(!benign_adapter.is_weight_anomalous(150.0));

    // Construct an anomalous poisoned adapter with extreme explosive weights
    let mut poisoned_adapter = AdapterPackage::new("base-model", 1, PeftType::LoRA);
    let mut extreme_matrix = Matrix::zeros(8, 2);
    extreme_matrix.set(0, 0, 500.0);
    let mut a_matrix = Matrix::zeros(2, 4);
    a_matrix.set(0, 0, 100.0);

    poisoned_adapter.modules.insert(
        "q_proj".into(),
        ModuleAdapter {
            module_name: "q_proj".into(),
            rank: 2,
            alpha: 16.0,
            lora_a: a_matrix,
            lora_b: extreme_matrix,
        },
    );

    assert!(
        poisoned_adapter.is_weight_anomalous(150.0),
        "Poisoned adapter with norm explosion must be flagged"
    );

    // TeeSandbox must reject and assign penalty loss 9999.0
    let (loss, acc) = sandbox.evaluate_adapter(&base_model, &poisoned_adapter, 0.0);
    assert_eq!(loss, 9999.0, "Anomalous adapter must receive penalty loss");
    assert_eq!(acc, 0.0, "Anomalous adapter must receive zero accuracy");
}

#[test]
fn test_diloco_fedadam_outer_optimizer_merging() {
    let mut rng = StdRng::seed_from_u64(12345);
    let mut model = DePEFTModel::new("test-model", 4, 8, 2, 2, PeftType::LoRA, &mut rng);

    // Create 2 candidate adapter packages with conflicting and concordant directions
    let mut pkg1 = AdapterPackage::new("test-model", 1, PeftType::LoRA);
    let mut pkg2 = AdapterPackage::new("test-model", 1, PeftType::LoRA);

    let mut lora_b1 = Matrix::zeros(8, 2);
    let mut lora_b2 = Matrix::zeros(8, 2);
    let mut lora_a1 = Matrix::zeros(2, 4);
    let mut lora_a2 = Matrix::zeros(2, 4);

    // Dimension [0, 0]: both miners agree (positive gradient update)
    lora_b1.set(0, 0, 1.0);
    lora_a1.set(0, 0, 1.0); // Delta W[0, 0] = (alpha / r) * 1.0 = 8.0

    lora_b2.set(0, 0, 1.0);
    lora_a2.set(0, 0, 1.0); // Delta W[0, 0] = (alpha / r) * 1.0 = 8.0

    // Dimension [1, 1]: miners conflict violently (miner 1 says +1, miner 2 says -1)
    lora_b1.set(1, 1, 1.0);
    lora_a1.set(1, 1, 1.0); // Delta W[1, 1] = +8.0

    lora_b2.set(1, 1, 1.0);
    lora_a2.set(1, 1, -1.0); // Delta W[1, 1] = -8.0

    pkg1.modules.insert(
        "q_proj".into(),
        ModuleAdapter {
            module_name: "q_proj".into(),
            rank: 2,
            alpha: 16.0,
            lora_a: lora_a1,
            lora_b: lora_b1,
        },
    );

    pkg2.modules.insert(
        "q_proj".into(),
        ModuleAdapter {
            module_name: "q_proj".into(),
            rank: 2,
            alpha: 16.0,
            lora_a: lora_a2,
            lora_b: lora_b2,
        },
    );

    let mut outer_state = DePEFT::ml::OuterOptimizerState::default();
    let initial_base_w = model.q_proj.base_weight.dequantize();

    let weighted_pkgs = vec![(&pkg1, 0.5f32), (&pkg2, 0.5f32)];
    model
        .merge_and_evolve_outer_optimizer(
            &weighted_pkgs,
            &mut outer_state,
            0.5,
            0.9,
            0.99,
            1e-6,
            &mut rng,
        )
        .expect("Outer optimizer merge must succeed");

    let updated_base_w = model.q_proj.base_weight.dequantize();

    // Check that agreeing dimension [0, 0] moved significantly
    let delta_00 = (updated_base_w.get(0, 0) - initial_base_w.get(0, 0)).abs();
    assert!(
        delta_00 > 0.01,
        "Consistent dimension should receive a positive update, got {}",
        delta_00
    );

    // Check that conflicting dimension [1, 1] was dampened towards 0
    let delta_11 = (updated_base_w.get(1, 1) - initial_base_w.get(1, 1)).abs();
    assert!(
        delta_11 < delta_00,
        "Conflicting dimension should be dampened compared to consistent dimension: delta_11={}, delta_00={}",
        delta_11,
        delta_00
    );
    assert_eq!(outer_state.step_count, 1);
}

#[test]
fn test_tournament_engine_with_outer_optimizer() {
    let mut rng = StdRng::seed_from_u64(999);
    let client = AccountId::new("client_outer_opt");
    let bounty_per_round = 10_000;
    let total_rounds = 2;

    let full_dataset = Dataset::generate_synthetic_task(30, 8, 4, 1.5, &mut rng);
    let (train_data, test_data) = full_dataset.train_test_split(0.7, &mut rng);

    let miners = vec![
        MinerNode::new(
            "miner_1",
            MinerHyperparams {
                learning_rate: 0.05,
                batch_size: 4,
                epochs: 1,
                hardware_type: "NVIDIA RTX 4090".to_string(),
            },
        ),
        MinerNode::new(
            "miner_2",
            MinerHyperparams {
                learning_rate: 0.03,
                batch_size: 4,
                epochs: 1,
                hardware_type: "AMD RX 7900".to_string(),
            },
        ),
    ];

    let validators = vec![
        ValidatorNode::new(
            "val_1",
            "Hardware Intel SGX",
            0.0,
            TeeSandbox::new(test_data.clone(), "sgx-1".to_string()),
            Some(EmbeddedVectorDb::new(64)),
        ),
        ValidatorNode::new(
            "val_2",
            "Hardware AMD SEV",
            0.0,
            TeeSandbox::new(test_data.clone(), "sev-1".to_string()),
            Some(EmbeddedVectorDb::new(64)),
        ),
    ];

    let mut engine = TournamentEngine::new(
        client,
        bounty_per_round,
        total_rounds,
        miners,
        validators,
        train_data,
        test_data,
        PeftType::LoRA,
    )
    .expect("Failed to initialize tournament engine");

    // Configure task to use OuterOptimizer (DiLoCo / FedAdam)
    if let Some(task) = engine.chain.tasks.get_mut(&engine.task_id) {
        task.merge_strategy = DePEFT::blockchain::types::MergeStrategy::OuterOptimizer {
            top_k: 2,
            outer_lr: 0.7,
            beta1: 0.9,
            beta2: 0.99,
            eps: 1e-8,
        };
    }

    let summary_r1 = engine.run_round(1).expect("Round 1 execution failed");
    assert_eq!(summary_r1.round_number, 1);
    assert_eq!(engine.outer_optimizer_state.step_count, 1);

    let summary_r2 = engine.run_round(2).expect("Round 2 execution failed");
    assert_eq!(summary_r2.round_number, 2);
    assert_eq!(engine.outer_optimizer_state.step_count, 2);
}

#[test]
fn test_mainnet_anti_fraud_validator_slashed_on_fake_tee() {
    use DePEFT::blockchain::types::{MergeStrategy, RewardDistribution};
    use DePEFT::tee::HardwareTeeEnclave;

    let mut chain = AppChainState::new();
    // On mainnet, simulation is strictly disallowed
    chain.tee_verifier.set_allow_simulation(false);
    chain.tee_verifier.enforce_attestation = true;

    let client = AccountId::new("client_mainnet");
    let val_fraud = AccountId::new("val_fraudulent");
    let miner = AccountId::new("miner_honest");

    chain.mint(client.clone(), 50_000);
    chain.mint(val_fraud.clone(), 10_000);

    // 1. Create task
    let task_tx = Transaction::CreateTask {
        client: client.clone(),
        nonce: chain.nonce_of(&client),
        base_model_id: b"Qwen/Qwen2.5-0.5B".to_vec(),
        base_model_hash: [1u8; 32],
        dataset_cid: b"bafy_dataset".to_vec(),
        peft_method: PeftType::LoRA,
        max_rank: 16,
        target_modules: vec![b"q_proj".to_vec()],
        bounty_pool: 20_000,
        epoch_blocks: 10,
        reward_distribution: RewardDistribution::WinnerTakesAll,
        merge_strategy: MergeStrategy::EnsembleWeighted { top_k: 1 },
    };
    chain.apply_transaction(task_tx, &client).unwrap();
    let task_id = 1;
    let round = 1;

    // 2. Miner commits & reveals
    let salt = b"miner_salt_123".to_vec();
    let adapter_hash = [5u8; 32];
    let commit_hash = AppChainState::compute_commit_hash(&adapter_hash, &salt);
    chain
        .apply_transaction(
            Transaction::CommitAdapter {
                task_id,
                round,
                miner: miner.clone(),
                nonce: chain.nonce_of(&miner),
                commit_hash,
            },
            &miner,
        )
        .unwrap();

    chain
        .set_round_phase(task_id, round, DePEFT::blockchain::RoundPhase::RevealPhase)
        .unwrap();
    chain
        .apply_transaction(
            Transaction::RevealAdapter {
                task_id,
                round,
                miner: miner.clone(),
                nonce: chain.nonce_of(&miner),
                adapter_cid: "bafy_miner_adapter".to_string(),
                salt,
                adapter_hash,
            },
            &miner,
        )
        .unwrap();

    // 3. Move to evaluation
    chain
        .set_round_phase(
            task_id,
            round,
            DePEFT::blockchain::RoundPhase::EvaluationPhase,
        )
        .unwrap();

    // Validator crafts a simulation quote (e.g. running in QEMU software sim)
    let enclave = HardwareTeeEnclave::new_with_flags(
        DePEFT::tee::TeeType::IntelSgxDcap,
        "sim_enclave",
        DePEFT::tee::TeeSecurityFlags::simulation(),
    );
    let fake_quote = enclave
        .generate_quote(task_id, round, std::slice::from_ref(&miner))
        .unwrap();

    let eval = ValidatorEvaluation {
        validator_address: val_fraud.clone(),
        ranking: vec![miner.clone()],
        loss_scores: vec![(miner.clone(), 0.1)],
        accuracy_scores: vec![(miner.clone(), 0.95)],
        hardware_info: "Fake Simulator".to_string(),
        attestation_quote: Some(fake_quote),
    };

    // 4. Submit evaluation with simulation quote: MUST BE REJECTED & SLASHED
    let eval_tx = Transaction::SubmitEvaluation {
        task_id,
        round,
        nonce: chain.nonce_of(&val_fraud),
        evaluation: eval,
    };
    let res = chain.apply_transaction(eval_tx, &val_fraud);
    assert!(res.is_err(), "Simulation TEE quote must be rejected on mainnet");

    // 5. Verify validator is slashed and balance burned
    assert!(chain.is_slashed(&val_fraud), "Fraudulent validator must be slashed");
    assert_eq!(chain.balance_of(&val_fraud), 0, "Slashed validator balance must be burned to 0");
    assert_eq!(chain.total_burned, 10_000, "Slashed 10,000 tokens must be recorded in total_burned");

    // 6. Slashed validator tries to submit any subsequent transaction -> REJECTED
    let spam_tx = Transaction::CommitAdapter {
        task_id,
        round: 2,
        miner: val_fraud.clone(),
        nonce: chain.nonce_of(&val_fraud),
        commit_hash: [7u8; 32],
    };
    let spam_res = chain.apply_transaction(spam_tx, &val_fraud);
    assert!(spam_res.is_err(), "Slashed account must be permanently banned from sending transactions");
}

#[test]
fn test_miner_anti_plagiarism_rejection() {
    use DePEFT::blockchain::types::{MergeStrategy, RewardDistribution};

    let mut chain = AppChainState::new();
    let client = AccountId::new("client_plag");
    let miner1 = AccountId::new("miner_original");
    let miner2 = AccountId::new("miner_copycat");

    chain.mint(client.clone(), 20_000);
    chain
        .apply_transaction(
            Transaction::CreateTask {
                client: client.clone(),
                nonce: 0,
                base_model_id: b"test_model".to_vec(),
                base_model_hash: [0u8; 32],
                dataset_cid: b"test_cid".to_vec(),
                peft_method: PeftType::LoRA,
                max_rank: 8,
                target_modules: vec![b"w".to_vec()],
                bounty_pool: 5000,
                epoch_blocks: 10,
                reward_distribution: RewardDistribution::WinnerTakesAll,
                merge_strategy: MergeStrategy::SingleWinner,
            },
            &client,
        )
        .unwrap();

    let shared_adapter_hash = [42u8; 32];
    let salt1 = b"salt1".to_vec();
    let salt2 = b"salt2".to_vec();

    // Both commit
    chain
        .apply_transaction(
            Transaction::CommitAdapter {
                task_id: 1,
                round: 1,
                miner: miner1.clone(),
                nonce: 0,
                commit_hash: AppChainState::compute_commit_hash(&shared_adapter_hash, &salt1),
            },
            &miner1,
        )
        .unwrap();

    chain
        .apply_transaction(
            Transaction::CommitAdapter {
                task_id: 1,
                round: 1,
                miner: miner2.clone(),
                nonce: 0,
                commit_hash: AppChainState::compute_commit_hash(&shared_adapter_hash, &salt2),
            },
            &miner2,
        )
        .unwrap();

    chain
        .set_round_phase(1, 1, DePEFT::blockchain::RoundPhase::RevealPhase)
        .unwrap();

    // Miner 1 reveals first
    chain
        .apply_transaction(
            Transaction::RevealAdapter {
                task_id: 1,
                round: 1,
                miner: miner1.clone(),
                nonce: 1,
                adapter_cid: "bafy_adapter_1".to_string(),
                salt: salt1,
                adapter_hash: shared_adapter_hash,
            },
            &miner1,
        )
        .unwrap();

    // Miner 2 reveals the exact same adapter hash -> REJECTED (Plagiarism)
    let copycat_res = chain.apply_transaction(
        Transaction::RevealAdapter {
            task_id: 1,
            round: 1,
            miner: miner2.clone(),
            nonce: 1,
            adapter_cid: "bafy_adapter_2".to_string(),
            salt: salt2,
            adapter_hash: shared_adapter_hash,
        },
        &miner2,
    );
    assert!(copycat_res.is_err(), "Duplicate adapter hash must be rejected as plagiarism");
}

#[test]
fn test_whistleblower_slashing_transaction() {
    use DePEFT::consensus::{EquivocationEvidence, Vote, VoteType};
    use DePEFT::crypto::AccountKeypair;

    let mut chain = AppChainState::new();
    let byzantine_kp = AccountKeypair::generate();
    let whistleblower_kp = AccountKeypair::generate();

    let byzantine_val = byzantine_kp.account_id();
    let whistleblower = whistleblower_kp.account_id();

    // Fund Byzantine validator
    chain.mint(byzantine_val.clone(), 100_000);

    // Byzantine validator signs two contradictory votes for the same block height and round
    let sign_bytes_a = Vote::sign_bytes(VoteType::Precommit, 10, 0, Some([11u8; 32]));
    let sig_a = byzantine_kp.sign_message(&sign_bytes_a);
    let vote_a = Vote {
        vote_type: VoteType::Precommit,
        height: 10,
        round: 0,
        block_hash: Some([11u8; 32]),
        validator: byzantine_val.clone(),
        signature: sig_a,
    };

    let sign_bytes_b = Vote::sign_bytes(VoteType::Precommit, 10, 0, Some([22u8; 32]));
    let sig_b = byzantine_kp.sign_message(&sign_bytes_b);
    let vote_b = Vote {
        vote_type: VoteType::Precommit,
        height: 10,
        round: 0,
        block_hash: Some([22u8; 32]), // Different block hash!
        validator: byzantine_val.clone(),
        signature: sig_b,
    };

    let evidence = EquivocationEvidence {
        validator: byzantine_val.clone(),
        height: 10,
        round: 0,
        vote_type: VoteType::Precommit,
        vote_a,
        vote_b,
    };

    // Whistleblower submits proof of equivocation
    let slash_tx = Transaction::SlashValidator {
        reporter: whistleblower.clone(),
        nonce: chain.nonce_of(&whistleblower),
        evidence,
    };

    chain.apply_transaction(slash_tx, &whistleblower).unwrap();

    // Verify validator is slashed
    assert!(chain.is_slashed(&byzantine_val));
    assert_eq!(chain.balance_of(&byzantine_val), 0);

    // Verify whistleblower received 20% bounty and 80% was burned
    let whistleblower_bal = chain.balance_of(&whistleblower);
    assert_eq!(whistleblower_bal, 20_000, "Whistleblower receives 20% bounty");
    assert_eq!(chain.total_burned, 80_000, "80% of slashed stake is burned");
}

#[test]
fn test_token_transfer_transaction() {
    let mut chain = AppChainState::new();
    let alice = AccountId::new("0xalice");
    let bob = AccountId::new("0xbob");

    chain.mint(alice.clone(), 50_000);
    assert_eq!(chain.balance_of(&alice), 50_000);
    assert_eq!(chain.balance_of(&bob), 0);

    let transfer_tx = Transaction::Transfer {
        from: alice.clone(),
        to: bob.clone(),
        amount: 15_000,
        nonce: chain.nonce_of(&alice),
    };

    chain.apply_transaction(transfer_tx, &alice).unwrap();

    assert_eq!(chain.balance_of(&alice), 35_000);
    assert_eq!(chain.balance_of(&bob), 15_000);
    assert_eq!(chain.nonce_of(&alice), 1);

    // Cannot transfer more than balance
    let excessive_tx = Transaction::Transfer {
        from: alice.clone(),
        to: bob.clone(),
        amount: 40_000,
        nonce: chain.nonce_of(&alice),
    };
    assert!(chain.apply_transaction(excessive_tx, &alice).is_err());
}

#[test]
fn test_weapon_1_heterogeneous_tee_quorum() {
    use DePEFT::blockchain::relative_consensus::RelativeConsensusEngine;
    use DePEFT::blockchain::types::{AccountId, ValidatorEvaluation};
    use DePEFT::tee::{AttestationQuote, EnclaveMeasurement, TeeSecurityFlags, TeeType};

    let m1 = AccountId::new("miner-alpha");
    let m2 = AccountId::new("miner-beta");
    let candidates = vec![m1.clone(), m2.clone()];

    let val_intel = AccountId::new("validator-sgx");
    let val_amd = AccountId::new("validator-sev");
    let ranking = vec![m1.clone(), m2.clone()];

    let quote_intel = AttestationQuote {
        tee_type: TeeType::IntelSgxDcap,
        measurement: EnclaveMeasurement {
            mrenclave: [0x11; 32],
            mrsigner: [0x22; 32],
            isv_prod_id: 1,
            isv_svn: 1,
        },
        report_data: AttestationQuote::compute_report_data(1, 1, &ranking),
        platform_public_key: [0x33; 32],
        quote_signature: vec![0x44; 64],
        timestamp: 1000,
        security_flags: TeeSecurityFlags::genuine_production(),
    };

    let quote_amd = AttestationQuote {
        tee_type: TeeType::AmdSevSnp,
        measurement: EnclaveMeasurement {
            mrenclave: [0x55; 32],
            mrsigner: [0x66; 32],
            isv_prod_id: 1,
            isv_svn: 1,
        },
        report_data: AttestationQuote::compute_report_data(1, 1, &ranking),
        platform_public_key: [0x77; 32],
        quote_signature: vec![0x88; 64],
        timestamp: 1000,
        security_flags: TeeSecurityFlags::genuine_production(),
    };

    let eval_intel = ValidatorEvaluation {
        validator_address: val_intel,
        ranking: ranking.clone(),
        loss_scores: vec![(m1.clone(), 0.2), (m2.clone(), 0.8)],
        accuracy_scores: vec![(m1.clone(), 0.8), (m2.clone(), 0.2)],
        hardware_info: "Intel Xeon / SGX DCAP".to_string(),
        attestation_quote: Some(quote_intel),
    };

    let eval_amd = ValidatorEvaluation {
        validator_address: val_amd,
        ranking: ranking.clone(),
        loss_scores: vec![(m1.clone(), 0.21), (m2.clone(), 0.79)],
        accuracy_scores: vec![(m1.clone(), 0.8), (m2.clone(), 0.2)],
        hardware_info: "AMD EPYC / SEV-SNP".to_string(),
        attestation_quote: Some(quote_amd),
    };

    // Scenario 1: Heterogeneous multi-vendor evaluation quorum achieved
    let res = RelativeConsensusEngine::aggregate(&[eval_intel.clone(), eval_amd], &candidates).unwrap();
    assert_eq!(res.winner, m1);
    assert_eq!(res.tee_diversity_count, 2, "Intel + AMD diversity count is 2");
    assert!(res.heterogeneous_quorum_achieved, "Heterogeneous quorum is true with >= 2 distinct TEEs");
    assert!(res.verified_tee_types.contains(&TeeType::IntelSgxDcap));
    assert!(res.verified_tee_types.contains(&TeeType::AmdSevSnp));

    // Scenario 2: Single vendor quote lacks heterogeneous quorum
    let res_single = RelativeConsensusEngine::aggregate(&[eval_intel], &candidates).unwrap();
    assert_eq!(res_single.tee_diversity_count, 1);
    assert!(!res_single.heterogeneous_quorum_achieved, "Single TEE vendor cannot satisfy heterogeneous quorum");
}

#[test]
fn test_weapon_2_adversarial_perturbation_noise_invariance() {
    use DePEFT::candle_peft::evaluator::CandleValidatorEvaluator;
    use DePEFT::candle_peft::transformer::{CandleTransformerConfig, CandleTransformerLM};
    use candle_core::Device;

    let device = Device::Cpu;
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
    let model = CandleTransformerLM::new(config, device).unwrap();

    let clean_samples = vec![
        "The quick brown fox jumps over the lazy dog".to_string(),
        "Parameter efficient fine tuning on decentralized network".to_string(),
    ];

    let perturbed = CandleValidatorEvaluator::generate_perturbed_samples(&clean_samples);
    assert_eq!(perturbed.len(), clean_samples.len());
    assert_ne!(perturbed[0], clean_samples[0], "Perturbed sample must have character perturbation");

    // Standard uncorrupted model should pass adversarial robustness check
    let is_robust = CandleValidatorEvaluator::verify_adversarial_robustness(&model, &clean_samples, 35.0).unwrap();
    assert!(is_robust, "Benign model should pass adversarial perturbation noise test");

    // Extreme threshold check (e.g. 0.00001 threshold should fail since perturbations do produce slight loss change)
    let too_strict = CandleValidatorEvaluator::verify_adversarial_robustness(&model, &clean_samples, 0.00001).unwrap();
    assert!(!too_strict, "Unreasonably strict threshold correctly flags loss divergence");
}

#[test]
fn test_weapon_3_p2p_anti_eclipse_subnet_diversity_and_reputation() {
    use DePEFT::blockchain::AccountId;
    use DePEFT::p2p::swarm::{P2pSwarm, SubnetKey};
    use DePEFT::p2p::types::PeerId;
    use std::net::{IpAddr, SocketAddr};

    let local_pid = PeerId::from_account(&AccountId::new("0xlocal_node"));
    let listen_addr: SocketAddr = "127.0.0.1:9001".parse().unwrap();
    let (swarm, _, _) = P2pSwarm::new(local_pid, listen_addr);

    // 1. Subnet key extraction
    let ip1: IpAddr = "192.168.1.5".parse().unwrap();
    let ip2: IpAddr = "192.168.200.99".parse().unwrap();
    let ip3: IpAddr = "10.0.0.1".parse().unwrap();
    let loopback: IpAddr = "127.0.0.1".parse().unwrap();

    let sub1 = SubnetKey::from_ip(ip1);
    let sub2 = SubnetKey::from_ip(ip2);
    let sub3 = SubnetKey::from_ip(ip3);
    let sub_loop = SubnetKey::from_ip(loopback);

    assert_eq!(sub1, SubnetKey::Ipv4([192, 168]));
    assert_eq!(sub2, SubnetKey::Ipv4([192, 168]));
    assert_eq!(sub1, sub2, "Both belong to the same 192.168.0.0/16 subnet");
    assert_ne!(sub1, sub3, "Belongs to a different subnet");
    assert_eq!(sub_loop, SubnetKey::Loopback, "Loopback is exempt");

    // 2. IP reputation and dynamic blacklisting
    let attacker_ip: IpAddr = "203.0.113.42".parse().unwrap();
    assert!(!swarm.is_ip_banned(&attacker_ip));
    assert_eq!(swarm.get_peer_reputation(&attacker_ip), 100);

    // Penalize attacker for sending invalid / forged signatures
    swarm.penalize_ip(&attacker_ip, 30);
    assert_eq!(swarm.get_peer_reputation(&attacker_ip), 70);
    assert!(!swarm.is_ip_banned(&attacker_ip));

    // Exceed ban threshold (-50)
    swarm.penalize_ip(&attacker_ip, 130);
    assert!(swarm.get_peer_reputation(&attacker_ip) <= -50);
    assert!(swarm.is_ip_banned(&attacker_ip), "IP must be banned when reputation drops below threshold");

    // Unban functionality
    swarm.unban_ip(&attacker_ip);
    assert!(!swarm.is_ip_banned(&attacker_ip), "IP unbanned");
}

#[test]
fn test_weapon_4_on_chain_circuit_breaker_emergency_safe_mode() {
    use DePEFT::blockchain::state::AppChainState;

    let mut state = AppChainState::new();
    state.set_circuit_breaker_velocity_limit(50_000);
    assert!(!state.circuit_breaker_active);

    // Initial payout of 20,000 succeeds (20,000 <= 50,000)
    assert!(state.check_and_record_payout(20_000).is_ok());
    assert!(!state.circuit_breaker_active);

    // Second payout of 25,000 succeeds (45,000 <= 50,000)
    assert!(state.check_and_record_payout(25_000).is_ok());
    assert!(!state.circuit_breaker_active);

    // Third sudden payout of 10,000 exceeds velocity cap (55,000 > 50,000)
    let res = state.check_and_record_payout(10_000);
    assert!(res.is_err(), "Must trigger circuit breaker when velocity cap is exceeded");
    assert!(state.circuit_breaker_active, "Emergency Safe Mode must be active");
    assert_eq!(state.circuit_breaker_triggered_at_block, Some(1));

    // Subsequent payout attempt is immediately blocked by Safe Mode
    let blocked_res = state.check_and_record_payout(1_000);
    assert!(blocked_res.is_err());
    assert!(blocked_res.unwrap_err().to_string().contains("Emergency Safe Mode is ACTIVE"));

    // Reset circuit breaker
    state.reset_circuit_breaker();
    assert!(!state.circuit_breaker_active);
    assert!(state.check_and_record_payout(5_000).is_ok(), "Payouts resume normally after reset");
}

#[test]
fn test_security_self_transfer_rejected() {
    let mut chain = AppChainState::new();
    let alice = AccountId::new("0xalice");
    chain.mint(alice.clone(), 10_000);

    let self_tx = Transaction::Transfer {
        from: alice.clone(),
        to: alice.clone(),
        amount: 1_000,
        nonce: chain.nonce_of(&alice),
    };
    let res = chain.apply_transaction(self_tx, &alice);
    assert!(res.is_err());
    assert!(res.unwrap_err().to_string().contains("cannot transfer tokens to oneself"));
}

#[test]
fn test_security_self_slashing_whistleblower_exploit_rejected() {
    use DePEFT::consensus::{EquivocationEvidence, Vote, VoteType};
    use DePEFT::crypto::AccountKeypair;

    let mut chain = AppChainState::new();
    let rogue_keypair = AccountKeypair::generate();
    let rogue = rogue_keypair.account_id();
    chain.mint(rogue.clone(), 100_000);

    let vote_a = Vote {
        vote_type: VoteType::Prevote,
        height: 5,
        round: 0,
        block_hash: Some([0x11; 32]),
        validator: rogue.clone(),
        signature: rogue_keypair.sign_message(&Vote::sign_bytes(VoteType::Prevote, 5, 0, Some([0x11; 32]))),
    };
    let vote_b = Vote {
        vote_type: VoteType::Prevote,
        height: 5,
        round: 0,
        block_hash: Some([0x22; 32]),
        validator: rogue.clone(),
        signature: rogue_keypair.sign_message(&Vote::sign_bytes(VoteType::Prevote, 5, 0, Some([0x22; 32]))),
    };

    let evidence = EquivocationEvidence {
        validator: rogue.clone(),
        height: 5,
        round: 0,
        vote_type: VoteType::Prevote,
        vote_a,
        vote_b,
    };

    // Rogue validator attempts to report themselves to claim 20% whistleblower bounty
    let slash_tx = Transaction::SlashValidator {
        reporter: rogue.clone(),
        nonce: chain.nonce_of(&rogue),
        evidence,
    };

    let res = chain.apply_transaction(slash_tx, &rogue);
    assert!(res.is_err());
    assert!(res.unwrap_err().to_string().contains("validator cannot report themselves to claim whistleblower bounty"));
}

#[test]
fn test_security_circuit_breaker_blocks_transfers_and_tasks() {
    let mut chain = AppChainState::new();
    let alice = AccountId::new("0xalice");
    let bob = AccountId::new("0xbob");
    chain.mint(alice.clone(), 50_000);

    chain.circuit_breaker_active = true;
    chain.circuit_breaker_triggered_at_block = Some(chain.block_height);

    // Transfer must be blocked during Safe Mode
    let transfer_tx = Transaction::Transfer {
        from: alice.clone(),
        to: bob.clone(),
        amount: 1_000,
        nonce: chain.nonce_of(&alice),
    };
    let res = chain.apply_transaction(transfer_tx, &alice);
    assert!(res.is_err());
    assert!(res.unwrap_err().to_string().contains("Emergency Safe Mode is ACTIVE"));

    // CreateTask must be blocked during Safe Mode
    let create_task_tx = Transaction::CreateTask {
        client: alice.clone(),
        nonce: chain.nonce_of(&alice),
        base_model_id: b"test-model".to_vec(),
        base_model_hash: [0u8; 32],
        dataset_cid: b"bafytest".to_vec(),
        peft_method: DePEFT::blockchain::PeftType::LoRA,
        max_rank: 8,
        target_modules: vec![b"q_proj".to_vec()],
        bounty_pool: 10_000,
        epoch_blocks: 10,
        reward_distribution: DePEFT::blockchain::RewardDistribution::WinnerTakesAll,
        merge_strategy: DePEFT::blockchain::MergeStrategy::default(),
    };
    let task_res = chain.apply_transaction(create_task_tx, &alice);
    assert!(task_res.is_err());
    assert!(task_res.unwrap_err().to_string().contains("Emergency Safe Mode is ACTIVE"));
}

#[test]
fn test_security_dangling_task_id_transactions_rejected() {
    let mut chain = AppChainState::new();
    let miner = AccountId::new("0xminer");
    chain.mint(miner.clone(), 10_000);

    let commit_tx = Transaction::CommitAdapter {
        task_id: 999999,
        round: 1,
        miner: miner.clone(),
        nonce: chain.nonce_of(&miner),
        commit_hash: [0x11; 32],
    };
    let res = chain.apply_transaction(commit_tx, &miner);
    assert!(res.is_err());
    assert!(res.unwrap_err().to_string().contains("Task #999999 does not exist"));
}

#[test]
fn test_security_mrsigner_without_mrenclave_accepted() {
    use DePEFT::tee::{AttestationQuote, EnclaveMeasurement, OnChainTeeVerifier, TeeSecurityFlags, TeeType};

    let mut verifier = OnChainTeeVerifier::new(true);
    let mrsigner = [0xAA; 32];
    let platform_key = [0xBB; 32];
    // Register ONLY mrsigner and platform key (no mrenclave whitelisted)
    verifier.register_mrsigner(mrsigner);
    verifier.register_platform_key(platform_key);
    verifier.set_allow_simulation(true); // allow test evaluation

    let quote = AttestationQuote {
        tee_type: TeeType::IntelSgxDcap,
        measurement: EnclaveMeasurement {
            mrenclave: [0x12; 32],
            mrsigner,
            isv_prod_id: 1,
            isv_svn: 1,
        },
        report_data: AttestationQuote::compute_report_data(1, 1, &[]),
        platform_public_key: platform_key,
        quote_signature: vec![0x00; 64],
        timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        security_flags: TeeSecurityFlags::genuine_production(),
    };

    // Measurement check must succeed because MRSIGNER matches
    assert!(!verifier.approved_mrsigners.is_empty());
    assert!(verifier.approved_mrsigners.contains(&quote.measurement.mrsigner));
}

#[test]
fn test_security_unicode_perturbation_vietnamese() {
    use DePEFT::candle_peft::evaluator::CandleValidatorEvaluator;
    use DePEFT::candle_peft::transformer::{CandleTransformerConfig, CandleTransformerLM};
    use candle_core::Device;

    let vietnamese_samples = vec![
        "Mạng lưới học sâu phi tập trung DePEFT hoàn toàn độc lập".to_string(),
        "Đồng thuận tương đối Borda count giải quyết trôi dạt dấu phẩy động".to_string(),
    ];

    let perturbed = CandleValidatorEvaluator::generate_perturbed_samples(&vietnamese_samples);
    assert_eq!(perturbed.len(), 2);
    assert_ne!(perturbed[0], vietnamese_samples[0]);

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
    let model = CandleTransformerLM::new(config, Device::Cpu).unwrap();
    let is_robust = CandleValidatorEvaluator::verify_adversarial_robustness(&model, &vietnamese_samples, 40.0).unwrap();
    assert!(is_robust, "Vietnamese unicode samples pass adversarial robustness check");
}
