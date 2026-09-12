#![allow(non_snake_case)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::manual_find)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::redundant_pattern_matching)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::chunks_exact_to_as_chunks)]
#![allow(clippy::while_let_loop)]

pub mod blockchain;
pub mod candle_peft;
pub mod client;
pub mod consensus;
pub mod crypto;
pub mod miner;
pub mod ml;
pub mod node;
pub mod p2p;
pub mod storage;
pub mod tee;
pub mod tournament;
pub mod validator;

pub use blockchain::{
    AccountId, AppChainState, CommitRecord, MergeStrategy, PeftType, RelativeConsensusEngine,
    RevealRecord, RewardDistribution, RoundPhase, RoundSummary, TaskSpec, Transaction,
    ValidatorEvaluation,
};
pub use candle_peft::{
    CandleLoraLinear, CandleMinerHyperparams, CandleMinerTrainer, CandleTrainingArtifact,
    CandleTransformerConfig, CandleTransformerLM, CandleValidatorEvaluator, CandleWeightMerger,
    DeviceBackendType, DeviceInfo, DeviceManager, PeftOptimizerType, SimpleByteTokenizer,
};
pub use client::{AccountInfo, DePeftClient, NodeStatus};
pub use consensus::{
    BftEngine, BftRoundState, BftStep, Block, BlockCommit, BlockHeader, ConsensusValidator,
    EquivocationEvidence, SlashingEngine, ValidatorSet, Vote, VoteType,
};
pub use crypto::{AccountKeypair, SignedTransaction};
pub use miner::{MinerHyperparams, MinerNode, MinerTrainer, MinerTrainingArtifact};
pub use ml::{
    AdapterPackage, Dataset, DePEFTModel, Matrix, ModuleAdapter, OuterOptimizerState, QLoRALinear,
    QuantizedWeight, Sample,
};
pub use node::{
    NodeContext, NodeSecurityConfig, create_app, start_node_server, start_node_server_with_store,
};
pub use p2p::{P2pMessage, P2pSwarm, PeerId, read_message, write_message};
pub use storage::{
    AdapterVectorRecord, ChainStore, DiskIpfsStorage, EmbeddedVectorDb, HybridStorageManager,
    IpfsAddResponse, IpfsKuboClient, IpfsNodeInfo, IpfsStorage, VectorSearchResult,
    deserialize_safetensors, serialize_safetensors,
};
pub use tee::{
    AttestationQuote, EnclaveMeasurement, HardwareTeeEnclave, OnChainTeeVerifier, TeeType,
};
pub use tournament::TournamentEngine;
pub use validator::{OffChainEvaluator, TeeSandbox, ValidatorNode};
