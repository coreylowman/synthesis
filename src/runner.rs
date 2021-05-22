use crate::env::Env;
use crate::mcts::{Policy, MCTS};
use tch::{self, nn, IndexOp, Tensor};

pub struct Timestep {
    state: Tensor,
    policy: Tensor,
    value: f32,
}

pub struct RunConfig {
    seed: u64,
    capacity: usize,
    num_explores: usize,
    temperature: f64,
    kind: tch::Kind,
    device: tch::Device,
}

fn extract_policy<E: Env + Clone, P: Policy<E>>(cfg: &RunConfig, mcts: &MCTS<E, P>) -> Tensor {
    let mut policy = Tensor::zeros(&[E::MAX_NUM_ACTIONS as i64], (cfg.kind, cfg.device));
    for (action, num_visits) in mcts.visit_counts() {
        let action_id: usize = action.into();
        let _ = policy
            .i(action_id as i64)
            .fill_((num_visits as f64).powf(1.0 / cfg.temperature));
    }
    policy /= policy.sum(cfg.kind);
    policy
}

pub fn run_game<E: Env + Clone, P: Policy<E>>(cfg: &RunConfig, policy: P) -> Vec<Timestep> {
    let mut mcts = MCTS::<E, P>::with_capacity(cfg.capacity, cfg.seed, policy);
    let mut ts: Vec<Timestep> = Vec::new();
    let mut game = E::new();
    let root_player = game.player();
    let mut is_over = false;
    while !is_over {
        mcts.explore_n(cfg.num_explores);

        // save timestep
        ts.push(Timestep {
            state: game.state(),
            policy: extract_policy(cfg, &mcts),
            value: 0.0,
        });

        let action = mcts.best_action();
        is_over = game.step(&action);
    }

    let mut r = game.reward(root_player); // TODO should this be -reward?
    for t in ts.iter_mut().rev() {
        t.value = r;
        r *= -1.0;
    }

    ts
}
