use crate::env::Env;
use crate::mcts::{Policy, MCTS};
use rand::{distributions::Distribution, distributions::WeightedIndex, Rng};
use tch::{Device, IndexOp, Kind, Tensor};

#[derive(Debug)]
pub struct Timestep {
    pub state: Tensor,
    pub policy: Tensor,
    pub value: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct RunConfig {
    pub capacity: usize,
    pub num_explores: usize,
    pub temperature: f64,
    pub kind: Kind,
    pub device: Device,
    pub sample_action: bool,
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

pub fn run_game<E: Env + Clone, P: Policy<E>, R: Rng>(
    cfg: &RunConfig,
    policy: &P,
    rng: &mut R,
) -> Vec<Timestep> {
    let mut mcts = MCTS::<E, P>::with_capacity(cfg.capacity, policy);
    let mut ts: Vec<Timestep> = Vec::new();
    let mut game = E::new();
    let mut is_over = false;
    while !is_over {
        mcts.explore_n(cfg.num_explores);

        // save timestep
        let mut policy = Tensor::zeros(&[E::MAX_NUM_ACTIONS as i64], (cfg.kind, cfg.device));
        let visit_counts = mcts.visit_counts();
        let mut weights = Vec::with_capacity(visit_counts.len());
        for &(action, num_visits) in visit_counts.iter() {
            let value = (num_visits as f64).powf(1.0 / cfg.temperature);
            weights.push(value);
            let action_id: usize = action.into();
            let _ = policy.i(action_id as i64).fill_(value);
        }
        policy /= policy.sum(cfg.kind);

        let t = Timestep {
            state: game.state(cfg.device),
            policy,
            value: 0.0,
        };
        ts.push(t);

        let action = if cfg.sample_action {
            let dist = WeightedIndex::new(weights).unwrap();
            let choice = dist.sample(rng);
            visit_counts[choice].0
        } else {
            mcts.best_action()
        };
        mcts.step_action(&action);

        // println!("-----");
        // println!("Applying action {:?}", action);
        is_over = game.step(&action);
        // game.print();
    }

    let mut r = -game.reward(game.player());
    for t in ts.iter_mut().rev() {
        t.value = r;
        r *= -1.0;
    }

    ts
}

pub fn gather_experience<E: Env + Clone, P: Policy<E>, R: Rng>(
    cfg: &RunConfig,
    policy: &P,
    rng: &mut R,
    n: usize,
) -> Vec<Timestep> {
    let mut ts = Vec::with_capacity(n);
    while ts.len() < n {
        ts.extend(run_game(cfg, policy, rng));
        println!("{:?}", ts.len());
    }
    ts
}
