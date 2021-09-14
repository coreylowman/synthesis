use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ValueTarget {
    Z,                           // Outcome of game {-1, 0, 1}
    Q,                           // Avg Value found while searching
    QZaverage,                   // (Q + Z) / 2
    QtoZ { from: f32, to: f32 }, // interpolate from Q to Z based on turns
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum Exploration {
    Uct { c: f32 },
    PolynomialUct { c: f32 },
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum ActionSelection {
    Q,         // avg value
    NumVisits, // num visits
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum Fpu {
    Const(f32),
    ParentQ,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct MCTSConfig {
    pub exploration: Exploration,
    pub action_selection: ActionSelection,
    pub solve: bool,
    pub fpu: Fpu,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum RolloutNoise {
    None,
    Equal { weight: f32 },
    Dirichlet { alpha: f32, weight: f32 },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LearningConfig {
    // general params
    pub seed: u64,
    pub logs: std::path::PathBuf,

    // training params
    pub lr_schedule: Vec<(usize, f64)>,
    pub weight_decay: f64,
    pub num_iterations: usize,
    pub num_epochs: usize,
    pub batch_size: i64,
    pub value_target: ValueTarget,
    pub policy_weight: f32,
    pub value_weight: f32,

    // replay buffer params
    pub buffer_size: usize,
    pub games_to_keep: usize,
    pub games_per_train: usize,

    // runner params
    pub num_threads: usize,
    pub num_explores: usize,
    pub num_random_actions: usize,
    pub sample_action_until: usize,
    pub noise: RolloutNoise,
    pub stop_games_when_solved: bool,

    pub learner_mcts_cfg: MCTSConfig,

    pub baseline_best_k: usize,
    pub baseline_mcts_cfg: MCTSConfig,
    pub baseline_num_games: usize,
    pub baseline_explores: Vec<usize>,
}
