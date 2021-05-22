use crate::env::Env;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::{Duration, Instant};

const C_PUCT: f64 = 1.0;

pub struct Node<E: Env + Clone> {
    pub parent: usize,
    pub env: E,
    pub terminal: bool,
    pub expanded: bool,
    pub actions: E::ActionIterator,
    pub children: Vec<(E::Action, usize)>,
    pub cum_value: f64,
    pub num_visits: f64,
    pub action_probs: Vec<f64>,
}

impl<E: Env + Clone> Node<E> {
    pub fn new(
        parent_id: usize,
        env: E,
        is_over: bool,
        action_probs: Vec<f64>,
        value: f64,
    ) -> Self {
        let actions = env.iter_actions();
        Node {
            parent: parent_id,
            env,
            terminal: is_over,
            expanded: is_over,
            actions,
            children: Vec::new(),
            num_visits: 1.0,
            cum_value: value,
            action_probs,
        }
    }
}

pub trait Policy<E: Env> {
    fn eval(&self, env: &E) -> (Vec<f64>, f64);
}

pub struct MCTS<E: Env + Clone, P: Policy<E>> {
    pub root: usize,
    pub nodes: Vec<Node<E>>,
    pub rng: StdRng, // note: this is about the same performance as SmallRng or any of the XorShiftRngs that got moved to the xorshift crate
    pub policy: P,
}

impl<E: Env + Clone, P: Policy<E>> MCTS<E, P> {
    pub fn with_capacity(capacity: usize, seed: u64, policy: P) -> Self {
        Self {
            root: 0,
            nodes: Vec::with_capacity(capacity),
            rng: StdRng::seed_from_u64(seed),
            policy,
        }
    }

    pub fn add_root(&mut self) {
        let env = E::new();
        let (action_probs, value) = self.policy.eval(&env);
        let root = Node::new(0, env, false, action_probs, value);
        self.nodes.push(root);
    }

    fn next_node_id(&self) -> usize {
        self.nodes.len() + self.root
    }

    pub fn step_action(&mut self, action: &E::Action) {
        // note: this function attempts to drop obviously unused nodes in order to reduce memory usage
        self.root = match self.nodes[self.root - self.root]
            .children
            .iter()
            .position(|(a, _)| a == action)
        {
            Some(action_index) => {
                let (_a, new_root) = self.nodes[self.root - self.root].children[action_index];
                drop(self.nodes.drain(0..new_root - self.root));
                new_root
            }
            None => {
                let mut env = self.nodes[self.root - self.root].env.clone();
                let is_over = env.step(action);
                let (action_probs, value) = self.policy.eval(&env);
                let child_node = Node::new(0, env, is_over, action_probs, value);
                self.nodes.clear();
                self.nodes.push(child_node);
                0
            }
        };

        self.nodes[0].parent = self.root;
    }

    pub fn visit_counts(&self) -> Vec<(E::Action, f64)> {
        let root = &self.nodes[self.root - self.root];

        let mut visits = Vec::with_capacity(root.children.len());

        for &(action, child_id) in root.children.iter() {
            let child = &self.nodes[child_id - self.root];
            visits.push((action, child.num_visits));
        }

        visits
    }

    pub fn best_action(&self) -> E::Action {
        let root = &self.nodes[self.root - self.root];

        let mut best_action_ind = 0;
        let mut best_value = -std::f64::INFINITY;

        for (i, &(_, child_id)) in root.children.iter().enumerate() {
            let child = &self.nodes[child_id - self.root];
            let value = child.num_visits;
            if value > best_value {
                best_value = value;
                best_action_ind = i;
            }
        }

        root.children[best_action_ind].0
    }

    fn explore(&mut self) {
        let mut node_id = self.root;
        loop {
            // assert!(node_id < self.nodes.len());
            let node = &mut self.nodes[node_id - self.root];
            if node.terminal {
                let (_, value) = self.policy.eval(&node.env);
                self.backprop(node_id, value);
                return;
            } else if node.expanded {
                node_id = self.select_best_child(node_id);
            } else {
                match node.actions.next() {
                    Some(action) => {
                        let child_id = self.next_node_id();

                        // add to children
                        let node = &mut self.nodes[node_id - self.root];
                        node.children.push((action, child_id));

                        // create the child node... note we will be modifying num_visits and reward later, so mutable
                        let mut env = node.env.clone();
                        let is_over = env.step(&action);
                        let (action_probs, value) = self.policy.eval(&env);
                        let child = Node::new(node_id, env, is_over, action_probs, value);
                        self.nodes.push(child);
                        self.backprop(node_id, value);
                        return;
                    }
                    None => {
                        node.expanded = true;
                        node_id = self.select_best_child(node_id);
                    }
                }
            }
        }
    }

    fn select_best_child(&mut self, node_id: usize) -> usize {
        // assert!(node_id < self.nodes.len());
        let node = &self.nodes[node_id - self.root];

        let visits = node.num_visits.sqrt();

        let mut best_child_id = self.root;
        let mut best_value = -std::f64::INFINITY;
        for &(action, child_id) in &node.children {
            let child = &self.nodes[child_id - self.root];
            let value = child.cum_value / child.num_visits
                + C_PUCT * node.action_probs[action.into()] * visits / (1.0 + child.num_visits);
            if value > best_value {
                best_child_id = child_id;
                best_value = value;
            }
        }
        assert!(best_child_id != self.root);
        best_child_id
    }

    fn backprop(&mut self, leaf_node_id: usize, mut value: f64) {
        let mut node_id = leaf_node_id;
        loop {
            // assert!(node_id < self.nodes.len());

            let node = &mut self.nodes[node_id - self.root];

            node.num_visits += 1.0;

            // note this is reversed because its actually the previous node's action that this node's reward is associated with
            node.cum_value += value;
            value *= -1.0;

            if node_id == self.root {
                break;
            }

            node_id = node.parent;
        }
    }

    pub fn explore_n(&mut self, n: usize) -> Duration {
        let start = Instant::now();
        let start_n = self.nodes.len();
        for _ in 0..n {
            self.explore();
        }
        // assert_eq!(self.nodes.len() - start_n, n);
        start.elapsed()
    }
}
