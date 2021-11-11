mod connect4;
mod policies;

use rand::{distributions::Distribution, thread_rng};
use rand_distr::Normal;

use crate::connect4::Connect4;
use crate::policies::*;
use synthesis::prelude::*;

fn learn<G: 'static + Game<N>, P: Policy<G, N> + NNPolicy<G, N>, const N: usize>(
) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = LearningConfig {
        seed: 0,                              // seed for rng & torch
        logs: train_dir("./_logs", G::NAME)?, // log directory
        num_iterations: 200,                  // number of training iterations to run

        lr_schedule: vec![(1, 1e-3), (20, 5e-4), (40, 1e-4), (60, 5e-5), (80, 1e-5)], // schedule for lr - first item in tuple is iteration #
        weight_decay: 1e-6, // L2 regularization for Adam optimizer
        num_epochs: 20,     // number of full passes over training data per iteration
        batch_size: 32,     // size of batches that epochs are split into
        policy_weight: 1.0, // scalar for policy loss
        value_weight: 1.0,  // scalar for value loss

        games_to_keep: 20000,  // number of games to keep in replay buffer
        games_per_train: 1000, // number of new games to add to replay buffer per training iteration

        rollout_cfg: RolloutConfig {
            num_workers: 6,                     // number of processes to use for running games
            num_explores: 1600,                 // number of MCTS explores per turn
            random_actions_until: 1,            // last turn number to select random actions
            sample_actions_until: 30,           // last turn number to sample actions
            stop_games_when_solved: false,      // end games early if they are solved by MCTS
            value_target: ValueTarget::Q,       // the target for NN value function
            action: ActionSelection::NumVisits, // the value to use for best action

            mcts_cfg: MCTSConfig {
                exploration: Exploration::PolynomialUct { c: 3.0 }, // type of exploration to use (e.g. PUCT or UCT)
                solve: true,                   // use MCTS Solver extension to solve nodes
                correct_values_on_solve: true, // if node is solved, adjust previously backprop'd values
                select_solved_nodes: true,     // select nodes that are solved
                auto_extend: true,             // visit nodes until a node with > 1 child is reached
                fpu: Fpu::Func(|| {
                    // exploit value of un-evaluated nodes
                    let dist = Normal::new(1.0, 0.1).unwrap();
                    dist.sample(&mut thread_rng())
                }),
                root_policy_noise: PolicyNoise::None,
            },
        },
    };

    let eval_cfg = EvaluationConfig {
        logs: cfg.logs.clone(),

        policy_num_explores: cfg.rollout_cfg.num_explores,
        policy_action: ActionSelection::NumVisits,
        policy_mcts_cfg: MCTSConfig {
            exploration: Exploration::PolynomialUct { c: 3.0 },
            solve: true,
            correct_values_on_solve: true,
            select_solved_nodes: true,
            auto_extend: true,
            fpu: Fpu::Const(1.0),
            root_policy_noise: PolicyNoise::None,
        },

        num_games_against_best_policies: 1,
        num_best_policies: 10,

        num_games_against_rollout: 5,
        rollout_num_explores: vec![800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800],
        rollout_action: ActionSelection::Q,
        rollout_mcts_cfg: MCTSConfig {
            exploration: Exploration::Uct { c: 2.0 },
            solve: true,
            correct_values_on_solve: true,
            select_solved_nodes: true,
            auto_extend: false,
            fpu: Fpu::Const(f32::INFINITY),
            root_policy_noise: PolicyNoise::None,
        },
    };

    tch::set_num_threads(1);
    tch::set_num_interop_threads(1);

    let eval_handle = std::thread::spawn(move || evaluator::<G, P, N>(&eval_cfg).unwrap());
    alpha_zero::<G, P, N>(&cfg)?;
    eval_handle.join().unwrap();
    Ok(())
}

fn main() {
    learn::<Connect4, Connect4Net, { Connect4::MAX_NUM_ACTIONS }>().unwrap()
}
