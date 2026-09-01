use crate::blockchain::state::AppChainState;
use crate::blockchain::transactions::Transaction;
use crate::blockchain::types::{AccountId, PeftType, RoundPhase, RoundSummary};
use crate::miner::worker::MinerNode;
use crate::ml::dataset::Dataset;
use crate::ml::model::DePEFTModel;
use crate::storage::ipfs::IpfsStorage;
use crate::storage::safetensors::deserialize_safetensors;
use crate::storage::vector_db::EmbeddedVectorDb;
use crate::validator::worker::ValidatorNode;
use anyhow::Result;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Multi-Round ReLoRA Tournament Orchestrator.
pub struct TournamentEngine {
    pub chain: AppChainState,
    pub ipfs: IpfsStorage,
    pub vector_db: EmbeddedVectorDb,
    pub base_model: DePEFTModel,
    pub dataset_train: Dataset,
    pub dataset_test: Dataset,
    pub miners: Vec<MinerNode>,
    pub validators: Vec<ValidatorNode>,
    pub current_round: usize,
    pub total_rounds: usize,
    pub bounty_per_round: u128,
    pub task_id: u64,
}

impl TournamentEngine {
    pub fn new(
        client_address: AccountId,
        bounty_per_round: u128,
        total_rounds: usize,
        miners: Vec<MinerNode>,
        validators: Vec<ValidatorNode>,
        train_dataset: Dataset,
        test_dataset: Dataset,
        peft_type: PeftType,
    ) -> Result<Self> {
        let mut chain = AppChainState::new();
        let ipfs = IpfsStorage::new();
        let vector_db = EmbeddedVectorDb::new(64);

        // Seeded RNG for reproducible baseline model initialization
        let mut rng = StdRng::seed_from_u64(42);
        let base_model = DePEFTModel::new("DePEFT-Llama3-Base", 8, 16, 4, 4, peft_type, &mut rng);

        // Upload initial base model weights to IPFS
        let base_model_bytes = serde_json::to_vec(&base_model)?;
        let _base_model_cid = ipfs.put(&base_model_bytes);
        let base_model_hash = {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(&base_model_bytes);
            h.finalize().into()
        };

        // Upload dataset to IPFS
        let dataset_bytes = serde_json::to_vec(&train_dataset)?;
        let dataset_cid = ipfs.put(&dataset_bytes);

        // Mint bounty funds for client
        let total_bounty = bounty_per_round * total_rounds as u128;
        chain.mint(client_address.clone(), total_bounty);

        // Register initial Task on chain
        let target_modules = vec![
            b"q_proj".to_vec(),
            b"v_proj".to_vec(),
            b"out_proj".to_vec(),
        ];

        chain.apply_transaction(
            Transaction::CreateTask {
                client: client_address.clone(),
                base_model_id: b"DePEFT-Llama3-Base".to_vec(),
                base_model_hash,
                dataset_cid: dataset_cid.as_bytes().to_vec(),
                peft_method: peft_type,
                max_rank: 64,
                target_modules,
                bounty_pool: total_bounty,
                epoch_blocks: 100,
            },
            &client_address,
        )?;

        let task_id = 1;

        Ok(Self {
            chain,
            ipfs,
            vector_db,
            base_model,
            dataset_train: train_dataset,
            dataset_test: test_dataset,
            miners,
            validators,
            current_round: 0,
            total_rounds,
            bounty_per_round,
            task_id,
        })
    }

    /// Execute a single full 5-phase ReLoRA Tournament Round.
    pub fn run_round(&mut self, round_num: usize) -> Result<RoundSummary> {
        let mut rng = StdRng::seed_from_u64(1000 + round_num as u64);

        // Baseline pre-merge loss on test set
        let (pre_merge_loss, _pre_acc) = self.base_model.evaluate(&self.dataset_test, 0.0);

        // -------------------------------------------------------------
        // Phase 1: Task Initialization & Base Model W_N Setup
        // -------------------------------------------------------------
        let base_model_bytes = serde_json::to_vec(&self.base_model)?;
        let current_base_cid = self.ipfs.put(&base_model_bytes);

        self.chain
            .start_round(self.task_id, round_num, current_base_cid.clone())?;

        let task_spec = self
            .chain
            .tasks
            .get(&self.task_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Task not found"))?;

        // -------------------------------------------------------------
        // Phase 2: Competition / Commit Phase
        // Miners train QLoRA locally and submit commit_hash = SHA256(adapter_hash || salt)
        // -------------------------------------------------------------
        self.chain
            .set_round_phase(self.task_id, round_num, RoundPhase::CommitPhase)?;

        for miner in &mut self.miners {
            let commit_tx = miner.run_training_and_commit(
                self.task_id,
                round_num,
                &task_spec,
                &self.base_model,
                &self.dataset_train,
                &mut rng,
            )?;
            self.chain.apply_transaction(commit_tx, &miner.account_id)?;
            self.chain.advance_block();
        }

        // -------------------------------------------------------------
        // Phase 3: Reveal Phase
        // Miners upload .safetensors to IPFS and submit CID + salt to chain
        // -------------------------------------------------------------
        self.chain
            .set_round_phase(self.task_id, round_num, RoundPhase::RevealPhase)?;

        let mut revealed_pairs: Vec<(AccountId, String)> = Vec::new();
        for miner in &mut self.miners {
            let (reveal_tx, cid) = miner.reveal_adapter(self.task_id, round_num, &self.ipfs)?;
            self.chain.apply_transaction(reveal_tx, &miner.account_id)?;
            revealed_pairs.push((miner.account_id.clone(), cid));
            self.chain.advance_block();
        }

        // -------------------------------------------------------------
        // Phase 4: Evaluation Phase
        // Validators evaluate adapters in TEE Sandbox on Private Test Set
        // -------------------------------------------------------------
        self.chain
            .set_round_phase(self.task_id, round_num, RoundPhase::EvaluationPhase)?;

        for validator in &self.validators {
            let eval_tx = validator.evaluate_round(
                &self.ipfs,
                self.task_id,
                round_num,
                &self.base_model,
                &revealed_pairs,
            )?;
            self.chain.apply_transaction(eval_tx, &validator.account_id)?;
            self.chain.advance_block();
        }

        // -------------------------------------------------------------
        // Phase 5: Merge Phase (ReLoRA Weight Evolution)
        // Relative Consensus determines Top 1 winner, weights are permanently merged:
        // W_{N+1} = W_N + \Delta W_{N+1}
        // -------------------------------------------------------------
        self.chain
            .set_round_phase(self.task_id, round_num, RoundPhase::MergePhase)?;

        // Find winner via consensus on chain state
        let round_ctx = self
            .chain
            .round_contexts
            .get(&(self.task_id, round_num))
            .ok_or_else(|| anyhow::anyhow!("Round context missing"))?;

        let revealed_miners: Vec<AccountId> = round_ctx.reveals.keys().cloned().collect();
        let eval_list: Vec<_> = round_ctx.evaluations.values().cloned().collect();

        let consensus = crate::blockchain::RelativeConsensusEngine::aggregate(
            &eval_list,
            &revealed_miners,
        )
        .ok_or_else(|| anyhow::anyhow!("Consensus failed"))?;

        let winner_miner = &consensus.winner;
        let winning_reveal = round_ctx
            .reveals
            .get(winner_miner)
            .ok_or_else(|| anyhow::anyhow!("Winner reveal not found"))?;

        // Retrieve winning .safetensors from IPFS
        let winning_bytes = self
            .ipfs
            .get(&winning_reveal.adapter_cid)
            .ok_or_else(|| anyhow::anyhow!("Winning safetensors not found in IPFS"))?;
        let winning_pkg = deserialize_safetensors(&winning_bytes)?;

        // Load winning adapter into base model and execute permanent ReLoRA weight merge:
        // W_{N+1} = W_N + \Delta W_{N+1}
        self.base_model.load_adapters(&winning_pkg);
        self.base_model.merge_and_evolve(&mut rng);

        // Measure evolved base model loss
        let (post_merge_loss, _post_acc) = self.base_model.evaluate(&self.dataset_test, 0.0);

        // Upload new evolved Base Model W_{N+1} to IPFS
        let evolved_model_bytes = serde_json::to_vec(&self.base_model)?;
        let evolved_cid = self.ipfs.put(&evolved_model_bytes);

        // Finalize round on blockchain
        let summary = self.chain.finalize_round(
            self.task_id,
            round_num,
            evolved_cid,
            pre_merge_loss,
            post_merge_loss,
            self.bounty_per_round,
        )?;

        Ok(summary)
    }
}
