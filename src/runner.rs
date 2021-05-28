use crate::env::Env;
use crate::mcts::{Policy, MCTS};
use crate::model::NNPolicy;
use ordered_float::OrderedFloat;
use rand::{distributions::Distribution, distributions::WeightedIndex, Rng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RolloutConfig {
    pub capacity: usize,
    pub num_explores: usize,
    pub temperature: f32,
    pub sample_action: bool,
    pub steps: usize,
}

pub struct ReplayBuffer<E: Env> {
    capacity: usize,
    state_size: usize,
    pi_size: usize,
    v_size: usize,
    pub states: Vec<f32>,
    pub pis: Vec<f32>,
    pub vs: Vec<f32>,
    _marker: PhantomData<E>,
}

impl<E: Env> ReplayBuffer<E> {
    pub fn new(n: usize) -> Self {
        let state_size = E::get_state_dims().iter().fold(1, |a, v| a * v) as usize;
        let state_capacity = n * state_size;
        let pi_capacity = n * E::MAX_NUM_ACTIONS;
        let v_capacity = n;
        Self {
            capacity: n,
            state_size,
            pi_size: E::MAX_NUM_ACTIONS,
            v_size: 1,
            states: Vec::with_capacity(state_capacity),
            pis: Vec::with_capacity(pi_capacity),
            vs: Vec::with_capacity(v_capacity),
            _marker: PhantomData,
        }
    }

    pub fn add(&mut self, state: &Vec<f32>, pi: &Vec<f32>, v: f32) {
        self.states.extend(state);
        self.pis.extend(pi);
        self.vs.push(v);
    }

    pub fn make_room(&mut self, n: usize) {
        if self.vs.len() + n > self.capacity {
            let num_to_drop = self.vs.len() + n - self.capacity;
            drop(self.states.drain(0..(num_to_drop * self.state_size)));
            drop(self.pis.drain(0..(num_to_drop * self.pi_size)));
            drop(self.vs.drain(0..(num_to_drop * self.v_size)));
        }
        assert!(self.vs.len() + n <= self.capacity);
    }
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
    buffer: &mut ReplayBuffer<E>,
) {
    let start_i = buffer.vs.len();
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

        buffer.add(mcts.root_state(), &policy, 0.0);

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
    for i in start_i..buffer.vs.len() {
        buffer.vs[i] = r;
        r *= -1.0;
    }
}

pub fn eval<E: Env, P: Policy<E> + NNPolicy<E>>(
    cfg: &RolloutConfig,
    policy_a: &mut P,
    policy_b: &mut P,
) -> f32 {
    let mut game = E::new();
    let player = game.player();
    let mut mcts_a = MCTS::<E, P>::with_capacity(cfg.capacity, policy_a);
    let mut mcts_b = MCTS::<E, P>::with_capacity(cfg.capacity, policy_b);
    loop {
        let action = if game.player() == player {
            mcts_a.explore_n(cfg.num_explores);
            mcts_a.best_action()
        } else {
            mcts_b.explore_n(cfg.num_explores);
            mcts_b.best_action()
        };
        mcts_a.step_action(&action);
        mcts_b.step_action(&action);
        if game.step(&action) {
            break;
        }
    }
    game.reward(player)
}

pub fn gather_experience<E: Env, P: Policy<E>, R: Rng>(
    cfg: &RolloutConfig,
    policy: &mut P,
    rng: &mut R,
    buffer: &mut ReplayBuffer<E>,
) {
    let mut cached_policy = PolicyWithCache {
        policy,
        cache: HashMap::with_capacity(cfg.steps * 2),
        _marker: PhantomData,
    };

    buffer.make_room(cfg.steps);
    while buffer.vs.len() < cfg.steps {
        run_game(cfg, &mut cached_policy, rng, buffer);
    }
}
