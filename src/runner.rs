use crate::env::Env;
use crate::mcts::{Policy, MCTS};
use crate::model::NNPolicy;
use ordered_float::OrderedFloat;
use rand::{
    distributions::Distribution, distributions::WeightedIndex, rngs::StdRng, seq::SliceRandom, Rng,
    SeedableRng,
};
use std::collections::HashMap;
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy)]
pub struct RolloutConfig {
    pub capacity: usize,
    pub num_explores: usize,
    pub temperature: f32,
    pub sample_action: bool,
    pub steps_per_epoch: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct EvaluationConfig {
    pub capacity: usize,
    pub num_explores: usize,
    pub num_games: usize,
    pub seed: u64,
}

struct PolicyWithCache<'a, E: Env, P: Policy<E>> {
    pub policy: &'a mut P,
    pub cache: HashMap<Vec<OrderedFloat<f32>>, (Vec<f32>, f32)>,
    _marker: PhantomData<E>,
}

impl<'a, E: Env, P: Policy<E>> Policy<E> for PolicyWithCache<'a, E, P> {
    fn eval(&mut self, state: &Vec<f32>) -> (Vec<f32>, f32) {
        let cache_key = state.iter().map(|&f| OrderedFloat(f)).collect();
        match self.cache.get(&cache_key) {
            Some((pi, v)) => (pi.clone(), *v),
            None => {
                let (pi, v) = self.policy.eval(state);
                self.cache.insert(cache_key, (pi.clone(), v));
                (pi, v)
            }
        }
    }
}

fn run_game<E: Env, P: Policy<E>, R: Rng>(
    cfg: &RolloutConfig,
    policy: &mut P,
    rng: &mut R,
    states: &mut Vec<f32>,
    pis: &mut Vec<f32>,
    vs: &mut Vec<f32>,
) {
    let start_i = vs.len();
    let mut mcts = MCTS::<E, P>::with_capacity(cfg.capacity, policy);
    let mut game = E::new();
    let start_player = game.player();
    let mut is_over = false;
    let mut policy = vec![0.0; E::MAX_NUM_ACTIONS];

    while !is_over {
        let _dur = mcts.explore_n(cfg.num_explores);
        // println!("{:?}", dur);

        let root_node = mcts.root_node();

        // save timestep
        let mut total = 0.0;
        policy.fill(0.0);
        for &(action, child_id) in root_node.children.iter() {
            let child = mcts.get_node(child_id);
            let value = child.num_visits.powf(1.0 / cfg.temperature);
            policy[action.into()] = value;
            total += value;
        }
        for i in 0..E::MAX_NUM_ACTIONS {
            policy[i] /= total;
        }

        states.extend(mcts.root_state());
        pis.extend(policy.iter());
        vs.push(0.0);

        let action = if cfg.sample_action {
            let dist = WeightedIndex::new(&policy).unwrap();
            let choice = dist.sample(rng);
            E::Action::from(choice)
        } else {
            mcts.best_action()
        };
        mcts.step_action(&action);

        // println!("-----");
        // println!("Applying action {:?}", action);
        is_over = game.step(&action);
        // game.print();
    }

    let mut r = game.reward(start_player);
    for i in start_i..vs.len() {
        vs[i] = r;
        r *= -1.0;
    }
}

pub fn eval<E: Env, P: Policy<E> + NNPolicy<E>>(cfg: &EvaluationConfig, policy: &mut P) -> f32 {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut game = E::new();
    let player = game.player();
    let mut wins = 0.0;
    for _i_game in 0..cfg.num_games {
        game = E::new();
        let mut mcts = MCTS::<E, P>::with_capacity(cfg.capacity, policy);
        loop {
            let action = if game.player() == player {
                mcts.explore_n(cfg.num_explores);
                mcts.best_action()
            } else {
                let actions: Vec<E::Action> = game.iter_actions().collect();
                *actions.choose(&mut rng).unwrap()
            };
            mcts.step_action(&action);
            if game.step(&action) {
                break;
            }
        }
        if game.reward(player) == 1.0 {
            wins += 1.0;
        }
    }
    wins / cfg.num_games as f32
}

pub fn gather_experience<E: Env, P: Policy<E>, R: Rng>(
    cfg: &RolloutConfig,
    policy: &mut P,
    rng: &mut R,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut states: Vec<f32> = Vec::with_capacity(cfg.steps_per_epoch * 2);
    let mut pis: Vec<f32> = Vec::with_capacity(cfg.steps_per_epoch * 2);
    let mut vs: Vec<f32> = Vec::with_capacity(cfg.steps_per_epoch * 2);
    let mut cached_policy = PolicyWithCache {
        policy,
        cache: HashMap::with_capacity(cfg.steps_per_epoch * 2),
        _marker: PhantomData,
    };

    while vs.len() < cfg.steps_per_epoch {
        run_game(cfg, &mut cached_policy, rng, &mut states, &mut pis, &mut vs);
    }

    (states, pis, vs)
}
