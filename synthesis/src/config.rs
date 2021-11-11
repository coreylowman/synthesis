#[derive(Debug, Clone, Copy)]
pub enum ValueTarget {
    Z,                           // Outcome of game {-1, 0, 1}
    Q,                           // Avg Value found while searching
    QZaverage { p: f32 },        // Q * p + Z * (1 - p)
    QtoZ { from: f32, to: f32 }, // interpolate from Q to Z based on turns
}

#[derive(Debug, Clone, Copy)]
pub enum Exploration {
    Uct { c: f32 },
    PolynomialUct { c: f32 },
}

#[derive(Debug, Clone, Copy)]
pub enum ActionSelection {
    Q,         // avg value
    NumVisits, // num visits
}

#[derive(Debug, Clone, Copy)]
pub enum Fpu {
    Const(f32),
    ParentQ,
    Func(fn() -> f32),
}

#[derive(Debug, Clone, Copy)]
pub struct MCTSConfig {
    pub exploration: Exploration,
    pub solve: bool,
    pub correct_values_on_solve: bool,
    pub select_solved_nodes: bool,
    pub auto_extend: bool,
    pub fpu: Fpu,
    pub root_policy_noise: PolicyNoise,
}

#[derive(Debug, Clone, Copy)]
pub enum PolicyNoise {
    None,
    Equal { weight: f32 },
    Dirichlet { alpha: f32, weight: f32 },
}

#[derive(Debug, Clone, Copy)]
pub struct RolloutConfig {
    pub num_workers: usize,
    pub num_explores: usize,
    pub random_actions_until: usize,
    pub sample_actions_until: usize,
    pub stop_games_when_solved: bool,
    pub value_target: ValueTarget,
    pub action: ActionSelection,
    pub mcts_cfg: MCTSConfig,
}

#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    pub logs: std::path::PathBuf,

    pub policy_num_explores: usize,
    pub policy_action: ActionSelection,
    pub policy_mcts_cfg: MCTSConfig,

    pub num_best_policies: usize,
    pub num_games_against_best_policies: usize,

    pub rollout_action: ActionSelection,
    pub rollout_num_explores: Vec<usize>,
    pub rollout_mcts_cfg: MCTSConfig,
    pub num_games_against_rollout: usize,
}

#[derive(Debug, Clone)]
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
    pub policy_weight: f32,
    pub value_weight: f32,

    pub games_to_keep: usize,
    pub games_per_train: usize,

    pub rollout_cfg: RolloutConfig,
}
