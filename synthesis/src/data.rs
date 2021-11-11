use crate::game::Game;
use std::{collections::HashMap, ffi::c_void};
use tch::{Kind, Tensor};
use torch_sys::at_tensor_of_data;

pub struct BatchRandSampler<'a> {
    inds: Tensor,

    x: &'a Tensor,
    y: &'a Tensor,
    z: &'a Tensor,

    size: i64,
    batch_size: i64,
    index: i64,
    drop_last: bool,
}

impl<'a> BatchRandSampler<'a> {
    pub fn new(
        x: &'a Tensor,
        y: &'a Tensor,
        z: &'a Tensor,
        batch_size: i64,
        drop_last: bool,
    ) -> Self {
        let n = x.size()[0];
        Self {
            inds: Tensor::randperm(n, tch::kind::INT64_CPU),
            x,
            y,
            z,
            size: n,
            batch_size,
            index: 0,
            drop_last,
        }
    }
}

impl<'a> Iterator for BatchRandSampler<'a> {
    type Item = (Tensor, Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        let next_index = (self.index + self.batch_size).min(self.size);
        if self.index >= self.size
            || (self.drop_last && (next_index - self.index) < self.batch_size)
        {
            return None;
        }

        let batch_inds = self
            .inds
            .narrow(0, self.index as i64, (next_index - self.index) as i64);
        self.index = next_index;

        let item = (
            self.x.index_select(0, &batch_inds),
            self.y.index_select(0, &batch_inds),
            self.z.index_select(0, &batch_inds),
        );
        Some(item)
    }
}

pub fn tensor<T>(data: &[T], dims: &[i64], kind: tch::Kind) -> Tensor {
    let dsize = kind.elt_size_in_bytes();
    let dtype = match kind {
        Kind::Uint8 => 0,
        Kind::Int8 => 1,
        Kind::Int16 => 2,
        Kind::Int => 3,
        Kind::Int64 => 4,
        Kind::Half => 5,
        Kind::Float => 6,
        Kind::Double => 7,
        Kind::ComplexHalf => 8,
        Kind::ComplexFloat => 9,
        Kind::ComplexDouble => 10,
        Kind::Bool => 11,
        Kind::QInt8 => 12,
        Kind::QUInt8 => 13,
        Kind::QInt32 => 14,
        Kind::BFloat16 => 15,
    };
    let data = data.as_ptr() as *const c_void;
    let ndims = dims.len();
    let dims = dims.as_ptr();
    unsafe { Tensor::from_ptr(at_tensor_of_data(data, dims, ndims, dsize, dtype)) }
}

pub struct FlatBatch<G: Game<N>, const N: usize> {
    pub states: Vec<G::Features>,
    pub pis: Vec<[f32; N]>,
    pub vs: Vec<[f32; 3]>,
}

#[derive(Debug)]
struct StateStatistics<G: Game<N>, const N: usize> {
    state: G::Features,
    sum_pi: [f32; N],
    sum_v: [f32; 3],
    num: u32,
}

pub struct ReplayBuffer<G: Game<N>, const N: usize> {
    game_id: usize,
    steps: usize,
    game_ids: Vec<usize>,
    pub games: Vec<G>,
    pub states: Vec<G::Features>,
    pub pis: Vec<[f32; N]>,
    pub vs: Vec<[f32; 3]>,
}

impl<G: Game<N>, const N: usize> ReplayBuffer<G, N> {
    pub fn new(n: usize) -> Self {
        Self {
            game_id: 0,
            steps: 0,
            game_ids: Vec::with_capacity(n),
            games: Vec::with_capacity(n),
            states: Vec::with_capacity(n),
            pis: Vec::with_capacity(n),
            vs: Vec::with_capacity(n),
        }
    }

    pub fn new_game(&mut self) {
        self.game_id += 1;
    }

    pub fn total_games_played(&self) -> usize {
        self.game_id
    }

    pub fn curr_games(&self) -> usize {
        let mut unique = self.game_ids.clone();
        unique.dedup();
        unique.len()
    }

    pub fn total_steps(&self) -> usize {
        self.steps
    }

    pub fn curr_steps(&self) -> usize {
        self.vs.len()
    }

    pub fn add(&mut self, game: &G, pi: &[f32; N], v: [f32; 3]) {
        self.game_ids.push(self.game_id);
        self.steps += 1;
        self.games.push(game.clone());
        self.states.push(game.features());
        self.pis.push(*pi);
        self.vs.push(v);
    }

    pub fn extend(&mut self, other: &mut Self) {
        self.steps += other.steps;
        let start = self.game_id;
        self.game_ids
            .extend(other.game_ids.iter().map(|&g| g + start));
        self.game_id += other.game_id;
        self.games.extend(other.games.drain(..));
        self.states.extend(other.states.drain(..));
        self.pis.extend(other.pis.drain(..));
        self.vs.extend(other.vs.drain(..));
    }

    pub fn keep_last_n_games(&mut self, n: usize) {
        if self.game_id <= n {
            return;
        }

        let min_game_id = self.game_id - n;

        let mut max_ind_to_remove = None;
        for (i, &game_id) in self.game_ids.iter().enumerate() {
            if game_id >= min_game_id {
                break;
            }
            max_ind_to_remove = Some(i);
        }
        if let Some(max_ind) = max_ind_to_remove {
            drop(self.game_ids.drain(0..=max_ind));
            drop(self.games.drain(0..=max_ind));
            drop(self.states.drain(0..=max_ind));
            drop(self.pis.drain(0..=max_ind));
            drop(self.vs.drain(0..=max_ind));
            assert!(self.game_ids[0] >= min_game_id);
        }
    }

    pub fn deduplicate(&self) -> FlatBatch<G, N> {
        let mut statistics: HashMap<G, StateStatistics<G, N>> =
            HashMap::with_capacity(self.game_ids.len());
        for i in 0..self.game_ids.len() {
            let stats = statistics
                .entry(self.games[i].clone())
                .or_insert(StateStatistics {
                    state: self.states[i].clone(),
                    sum_pi: [0.0; N],
                    sum_v: [0.0; 3],
                    num: 0,
                });
            for j in 0..N {
                stats.sum_pi[j] += self.pis[i][j];
            }
            for j in 0..3 {
                stats.sum_v[j] += self.vs[i][j];
            }
            stats.num += 1;
        }

        let mut states = Vec::with_capacity(statistics.len());
        let mut pis = Vec::with_capacity(statistics.len());
        let mut vs = Vec::with_capacity(statistics.len());
        for (_, stats) in statistics.iter() {
            let mut avg_pi = [0.0; N];
            for i in 0..N {
                avg_pi[i] = stats.sum_pi[i] / stats.num as f32;
            }
            let mut avg_v = [0.0; 3];
            for i in 0..3 {
                avg_v[i] = stats.sum_v[i] / stats.num as f32;
            }
            states.push(stats.state.clone());
            pis.push(avg_pi);
            vs.push(avg_v);
        }

        FlatBatch { states, pis, vs }
    }
}
