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
    /// Aggregate validator evaluations using Strategic-Resistant Robust Borda & Consensus voting.
    /// Filters malicious outlier validator rankings and awards consensus to the robust winner.
    pub fn aggregate(
        evaluations: &[ValidatorEvaluation],
        candidate_miners: &[AccountId],
    ) -> Option<ConsensusResult> {
        if evaluations.is_empty() || candidate_miners.is_empty() {
            return None;
        }

        let num_validators = evaluations.len();
        let mut miner_ranks: HashMap<AccountId, Vec<usize>> = HashMap::new();
        for miner in candidate_miners {
            miner_ranks.insert(miner.clone(), Vec::new());
        }

        // Collect each validator's ordinal position for each miner with deduplication
        let mut seen_validators = std::collections::HashSet::new();
        for eval in evaluations {
            if !seen_validators.insert(&eval.validator_address) {
                continue; // Ignore duplicate submissions from the same validator address
            }

            // Deduplicate ranking items from this validator to prevent ranking inflation
            let mut unique_ranking: Vec<AccountId> = Vec::new();
            for miner in &eval.ranking {
                if !unique_ranking.contains(miner) {
                    unique_ranking.push(miner.clone());
                }
            }

            let m = unique_ranking.len();
            for (rank_idx, miner) in unique_ranking.iter().enumerate() {
                if let Some(ranks) = miner_ranks.get_mut(miner) {
                    ranks.push(rank_idx);
                }
            }
            // For any candidate not ranked in evaluation, assign worst rank
            for miner in candidate_miners {
                if !unique_ranking.contains(miner) {
                    if let Some(ranks) = miner_ranks.get_mut(miner) {
                        ranks.push(m);
                    }
                }
            }
        }

        if seen_validators.is_empty() {
            return None;
        }

        let mut score_map: HashMap<AccountId, usize> = HashMap::new();
        let m = candidate_miners.len();

        for (miner, mut ranks) in miner_ranks {
            ranks.sort_unstable();
            // Robust Outlier Removal (Trimmed Borda Count):
            // Dynamically scale trimming for larger validator sets to eliminate Sybil collusion
            let trim_count = if ranks.len() >= 4 {
                (ranks.len() / 4).max(1)
            } else {
                0
            };

            let trimmed_ranks = if ranks.len() > 2 * trim_count {
                &ranks[trim_count..ranks.len() - trim_count]
            } else {
                &ranks[..]
            };

            let mut total_points = 0;
            for &rank in trimmed_ranks {
                let points = m.saturating_sub(1 + rank);
                total_points += points;
            }
            score_map.insert(miner, total_points);
        }

        let mut borda_scores: Vec<(AccountId, usize)> = score_map.into_iter().collect();
        // Sort descending by score. Deterministic tie breaker by AccountId name.
        borda_scores.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.0.cmp(&b.0.0)));

        let consensus_ranking: Vec<AccountId> =
            borda_scores.iter().map(|(m, _)| m.clone()).collect();
        let winner = consensus_ranking[0].clone();

        // Calculate consensus agreement rate
        let mut winner_votes = 0;
        for eval in evaluations {
            if let Some(first) = eval.ranking.first() {
                if first == &winner {
                    winner_votes += 1;
                }
            }
        }
        let agreement_rate = winner_votes as f64 / num_validators as f64;

        Some(ConsensusResult {
            consensus_ranking,
            borda_scores,
            winner,
            agreement_rate,
        })
    }
}
