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
    /// Number of distinct hardware TEE architectures verifying this round
    #[serde(default)]
    pub tee_diversity_count: usize,
    /// True if evaluations are backed by at least 2 distinct hardware TEE architectures
    #[serde(default)]
    pub heterogeneous_quorum_achieved: bool,
    /// List of distinct verified hardware TEE types participating in this round
    #[serde(default)]
    pub verified_tee_types: Vec<crate::tee::TeeType>,
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

        // Deduplicate validator submissions upfront to prevent Sybil collusion and metric distortion
        let mut seen_validators = std::collections::HashSet::new();
        let mut unique_evaluations = Vec::new();
        for eval in evaluations {
            if seen_validators.insert(&eval.validator_address) {
                unique_evaluations.push(eval);
            }
        }

        if unique_evaluations.is_empty() {
            return None;
        }

        let num_validators = unique_evaluations.len();
        let mut miner_ranks: HashMap<AccountId, Vec<usize>> = HashMap::new();
        for miner in candidate_miners {
            miner_ranks.insert(miner.clone(), Vec::new());
        }

        let m = candidate_miners.len();

        // Collect each unique validator's ordinal position for each miner
        for eval in &unique_evaluations {
            // Deduplicate ranking items from this validator and retain only valid candidate miners
            let mut unique_ranking: Vec<AccountId> = Vec::new();
            for miner in &eval.ranking {
                if miner_ranks.contains_key(miner) && !unique_ranking.contains(miner) {
                    unique_ranking.push(miner.clone());
                }
            }

            for (rank_idx, miner) in unique_ranking.iter().enumerate() {
                if let Some(ranks) = miner_ranks.get_mut(miner) {
                    ranks.push(rank_idx);
                }
            }
            // For any candidate not ranked in evaluation, assign worst possible rank (m)
            for miner in candidate_miners {
                if !unique_ranking.contains(miner) {
                    if let Some(ranks) = miner_ranks.get_mut(miner) {
                        ranks.push(m);
                    }
                }
            }
        }

        let mut score_map: HashMap<AccountId, usize> = HashMap::new();

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

        // Calculate consensus agreement rate over authentic unique validators
        let mut winner_votes = 0;
        for eval in &unique_evaluations {
            if let Some(first) = eval.ranking.iter().find(|m| candidate_miners.contains(m)) {
                if first == &winner {
                    winner_votes += 1;
                }
            }
        }
        let agreement_rate = winner_votes as f64 / num_validators as f64;

        // Weapon 1: Heterogeneous TEE Multi-Vendor Quorum (Polyphony)
        // Check hardware TEE attestation platforms across unique validator submissions
        let mut seen_tees = std::collections::HashSet::new();
        let mut verified_tee_types = Vec::new();
        for eval in &unique_evaluations {
            if let Some(quote) = &eval.attestation_quote {
                if seen_tees.insert(quote.tee_type) {
                    verified_tee_types.push(quote.tee_type);
                }
            }
        }
        let tee_diversity_count = verified_tee_types.len();
        let heterogeneous_quorum_achieved = tee_diversity_count >= 2;

        Some(ConsensusResult {
            consensus_ranking,
            borda_scores,
            winner,
            agreement_rate,
            tee_diversity_count,
            heterogeneous_quorum_achieved,
            verified_tee_types,
        })
    }
}
