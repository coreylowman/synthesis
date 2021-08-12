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

        lr_schedule: vec![(1, 1e-3), (40, 1e-4), (80, 1e-5)],
        weight_decay: 1e-4,
        num_iterations: 200,
        num_epochs: 20,
        batch_size: 64,
        value_target: ValueTarget::QtoZ { from: 0.5, to: 1.0 },

        buffer_size: 256_000,
        games_to_keep: 20000,
        games_per_train: 1000,

        num_explores: 1600,
        num_random_actions: 1,
        sample_action_until: 25, // TODO 63 (max turns)

        // TODO entropy dirichlet { 0.25 }, num moves dirichlet { 0.25, 10.0 }
        noise: RolloutNoise::Dirichlet {
            alpha: 10.0 / (N as f32),
            weight: 0.25,
        },

        learner_mcts_cfg: MCTSConfig {
            exploration: Exploration::PUCT { c: 2.0 },    // TODO try kl
            action_selection: ActionSelection::NumVisits, // TODO try num visits
            solve: true,
            fpu: 1.0,
        },

        baseline_mcts_cfg: MCTSConfig {
            exploration: Exploration::UCT { c: 2.0 },
            action_selection: ActionSelection::Q,
            solve: true,
            fpu: f32::INFINITY,
        },
        baseline_num_games: 10,
        baseline_explores: vec![
            100, 200, 400, 800, 1600, 2400, 3200, 4800, 6400, 9600, 12800, 25600, 51200,
        ],
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
