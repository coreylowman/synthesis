use crate::env::Env;
use crate::mcts::{Policy, MCTS};
use rand::{distributions::Distribution, distributions::WeightedIndex, Rng};
use tch::{Device, IndexOp, Kind, Tensor};

#[derive(Debug, Clone, Copy)]
pub struct RunConfig {
    pub capacity: usize,
    pub num_explores: usize,
    pub temperature: f32,
    pub kind: Kind,
    pub device: Device,
    pub sample_action: bool,
    pub steps_per_epoch: usize,
}

pub fn run_game<E: Env + Clone, P: Policy<E>, R: Rng>(
    cfg: &RunConfig,
    policy: &P,
    rng: &mut R,
) -> (Vec<Tensor>, Vec<Tensor>, Vec<f32>) {
    let mut mcts = MCTS::<E, P>::with_capacity(cfg.capacity, policy);
    let mut states: Vec<Tensor> = Vec::new();
    let mut pis: Vec<Tensor> = Vec::new();
    let mut vs: Vec<f32> = Vec::new();
    let mut game = E::new();
    let start_player = game.player();
    let mut is_over = false;
    while !is_over {
        let dur = mcts.explore_n(cfg.num_explores);
        println!("{:?}", dur);

        // save timestep
        let mut policy = Tensor::zeros(&[E::MAX_NUM_ACTIONS as i64], (cfg.kind, cfg.device));
        let visit_counts = mcts.visit_counts();
        let mut weights = Vec::with_capacity(visit_counts.len());
        for &(action, num_visits) in visit_counts.iter() {
            let value = (num_visits as f32).powf(1.0 / cfg.temperature);
            weights.push(value);
            let action_id: usize = action.into();
            let _ = policy.i(action_id as i64).fill_(value as f64);
        }
        policy /= policy.sum(cfg.kind);

        states.push(game.state(cfg.kind, cfg.device));
        pis.push(policy);
        vs.push(0.0);

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
    for t in vs.iter_mut().rev() {
        *t = r;
        r *= -1.0;
    }

    if game.player() == start_player {
        // first player lost & did not go last
        assert_eq!(vs[0], -1.0);
        assert_eq!(vs[vs.len() - 1], 1.0);
    } else {
        // first player won & was last to go
        assert!(game.player() != start_player);
        assert_eq!(vs[0], 1.0);
        assert_eq!(vs[vs.len() - 1], 1.0);
    }

    (states, pis, vs)
}

pub fn gather_experience<E: Env + Clone, P: Policy<E>, R: Rng>(
    cfg: &RunConfig,
    policy: &P,
    rng: &mut R,
) -> (Tensor, Tensor, Tensor) {
    let mut states: Vec<Tensor> = Vec::new();
    let mut pis: Vec<Tensor> = Vec::new();
    let mut vs: Vec<f32> = Vec::new();

    while states.len() < cfg.steps_per_epoch {
        let (s, p, v) = run_game(cfg, policy, rng);
        states.extend(s);
        pis.extend(p);
        vs.extend(v);
        println!("{:?}", states.len());
    }

    let states_t = Tensor::stack(&states, 0);
    assert!(states_t.size()[0] == states.len() as i64);
    assert!(states_t.size()[1..] == states[0].size());

    let pis_t = Tensor::stack(&pis, 0);
    assert!(pis_t.size()[0] == pis.len() as i64);
    assert!(pis_t.size()[1..] == pis[0].size());

    let vs_t = Tensor::of_slice(&vs).unsqueeze(1);
    assert!(vs_t.size().len() == 2);
    assert!(vs_t.size()[0] == vs.len() as i64);
    assert!(vs_t.size()[1] == 1);

    (states_t, pis_t, vs_t)
}
