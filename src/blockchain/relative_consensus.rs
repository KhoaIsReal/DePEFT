use crate::blockchain::types::{AccountId, ValidatorEvaluation};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of running the Relative Consensus aggregation over validator rankings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsensusResult {
    /// Consensus ranked order of miners from 1st place (winner) to last place.
    pub consensus_ranking: Vec<AccountId>,
    /// Borda count score for each miner (higher is better).
    pub borda_scores: Vec<(AccountId, usize)>,
    /// Top-1 winner of the round.
    pub winner: AccountId,
    /// Measure of consensus agreement among validators (0.0 to 1.0).
    pub agreement_rate: f64,
}

/// Relative Consensus Engine for DePEFT.
/// Resolves Floating-point Drift (BF16/FP16) across heterogeneous hardware (NVIDIA CUDA, AMD ROCm, CPUs)
/// by reaching consensus on relative rank ordering rather than exact float loss values.
pub struct RelativeConsensusEngine;

impl RelativeConsensusEngine {
    /// Aggregate validator evaluations using Borda Count rank voting.
    pub fn aggregate(
        evaluations: &[ValidatorEvaluation],
        candidate_miners: &[AccountId],
    ) -> Option<ConsensusResult> {
        if evaluations.is_empty() || candidate_miners.is_empty() {
            return None;
        }

        let mut score_map: HashMap<AccountId, usize> = HashMap::new();
        for miner in candidate_miners {
            score_map.insert(miner.clone(), 0);
        }

        // Apply Borda count: 1st place gets (N-1) points, 2nd gets (N-2), etc.
        for eval in evaluations {
            let m = eval.ranking.len();
            for (rank_idx, miner) in eval.ranking.iter().enumerate() {
                if rank_idx < m {
                    let points = m.saturating_sub(1 + rank_idx);
                    *score_map.entry(miner.clone()).or_insert(0) += points;
                }
            }
        }

        let mut borda_scores: Vec<(AccountId, usize)> = score_map.into_iter().collect();
        // Sort descending by score. Deterministic tie breaker by AccountId name.
        borda_scores.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0 .0.cmp(&b.0 .0)));

        let consensus_ranking: Vec<AccountId> = borda_scores.iter().map(|(m, _)| m.clone()).collect();
        let winner = consensus_ranking[0].clone();

        // Calculate consensus agreement rate: percentage of validators that agreed with the consensus winner
        let mut winner_votes = 0;
        for eval in evaluations {
            if let Some(first) = eval.ranking.first() {
                if first == &winner {
                    winner_votes += 1;
                }
            }
        }
        let agreement_rate = winner_votes as f64 / evaluations.len() as f64;

        Some(ConsensusResult {
            consensus_ranking,
            borda_scores,
            winner,
            agreement_rate,
        })
    }
}
