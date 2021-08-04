mod connect4;
mod policies;

use crate::connect4::Connect4;
use crate::policies::*;
use synthesis::prelude::*;

fn learn<G: Game<N>, P: Policy<G, N> + NNPolicy<G, N>, const N: usize>(
) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = LearningConfig {
        seed: 0,
        logs: train_dir("./_logs", G::NAME)?,

        lr: 1e-3,
        weight_decay: 1e-4,
        num_iterations: 200,
        num_epochs: 20,
        value_target: ValueTarget::QtoZ,

        batch_size: 64,
        buffer_size: 256_000,
        games_to_keep: 20000,
        games_per_train: 1000,

        num_explores: 1600,
        num_random_actions: 0,
        sample_action_until: 25,
        alpha: 10.0 / (N as f32),
        noisy_explore: true,
        noise_weight: 0.25,

        learner_mcts_cfg: MCTSConfig {
            exploration: MCTSExploration::PUCT { c: 2.0 },
            solve: true,
            fpu: 1.0,
        },

        baseline_mcts_cfg: MCTSConfig {
            exploration: MCTSExploration::UCT { c: 2.0 },
            solve: true,
            fpu: f32::INFINITY,
        },
    };

    // TODO try w/d/l & dot with [1,0,-1]
    // TODO try w/d/l & take argmax
    // TODO generate games using 50% latest & 50% best

    tch::set_num_threads(2);
    tch::set_num_interop_threads(2);

    let eval_cfg = cfg.clone();
    let eval_handle = std::thread::spawn(move || evaluator::<G, P, N>(&eval_cfg).unwrap());
    learner::<G, P, N>(&cfg)?;
    eval_handle.join().unwrap();
    Ok(())
}

fn main() {
    learn::<Connect4, Connect4Net, { Connect4::MAX_NUM_ACTIONS }>().unwrap()
}
