mod connect4;
mod policies;

use crate::connect4::Connect4;
use crate::policies::*;
use synthesis::prelude::*;

fn learn<G: 'static + Game<N>, P: Policy<G, N> + NNPolicy<G, N>, const N: usize>(
) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = LearningConfig {
        seed: 0,
        logs: train_dir("./_logs", G::NAME)?,

        lr_schedule: vec![(1, 1e-3), (20, 5e-4), (40, 1e-4), (60, 5e-5), (80, 1e-5)],
        weight_decay: 1e-4,
        num_iterations: 200,
        num_epochs: 20,
        batch_size: 32,
        value_target: ValueTarget::Q,
        policy_weight: 1.0,
        value_weight: 1.0,

        buffer_size: 256_000,
        games_to_keep: 20000,
        games_per_train: 1000,

        num_workers: 6,
        num_explores: 1600,
        num_random_actions: 1,
        sample_action_until: 64,
        stop_games_when_solved: false,
        noise: RolloutNoise::None,
        learner_mcts_cfg: MCTSConfig {
            exploration: Exploration::PolynomialUct { c: 3.0 },
            action_selection: ActionSelection::NumVisits,
            solve: true,
            fpu: Fpu::Const(1.0),
        },

        baseline_best_k: 10,
        baseline_mcts_cfg: MCTSConfig {
            exploration: Exploration::Uct { c: 2.0 },
            action_selection: ActionSelection::Q,
            solve: true,
            fpu: Fpu::Const(f32::INFINITY),
        },
        baseline_num_games: 10,
        baseline_explores: vec![800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800],
    };

    tch::set_num_threads(1);
    tch::set_num_interop_threads(1);

    let eval_cfg = cfg.clone();
    let eval_handle = std::thread::spawn(move || evaluator::<G, P, N>(&eval_cfg).unwrap());
    alpha_zero::<G, P, N>(&cfg)?;
    eval_handle.join().unwrap();
    Ok(())
}

fn main() {
    learn::<Connect4, Connect4Net, { Connect4::MAX_NUM_ACTIONS }>().unwrap()
}
