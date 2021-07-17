use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ValueTarget {
    Z,         // Outcome of game {-1, 0, 1}
    Q,         // Avg Value found while searching
    QZaverage, // (Q + Z) / 2
    QtoZ,      // interpolate from Q to Z based on turns
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LearningConfig {
    // general params
    pub seed: u64,
    pub logs: std::path::PathBuf,

    // training params
    pub lr: f64,
    pub weight_decay: f64,
    pub num_iterations: usize,
    pub num_epochs: usize,
    pub batch_size: i64,
    pub value_target: ValueTarget,

    // replay buffer params
    pub buffer_size: usize,
    pub games_to_keep: usize,
    pub games_per_train: usize,

    // runner params
    pub num_explores: usize,
    pub num_random_actions: usize,
    pub sample_action_until: usize,
    pub alpha: f32,
    pub noisy_explore: bool,
    pub noise_weight: f32,

    // mcts params
    pub c_puct: f32,
    pub solve: bool,
    pub fpu: f32,
}
