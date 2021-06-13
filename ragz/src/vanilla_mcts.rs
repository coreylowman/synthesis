use crate::env::Env;
use rand::Rng;
use std::time::{Duration, Instant};

pub struct Node<E: Env<N>, const N: usize> {
    pub parent: usize,
    pub env: E,
    pub terminal: bool,
    pub expanded: bool,
    pub actions: E::ActionIterator,
    pub children: Vec<(E::Action, usize)>,
    pub cum_value: f32,
    pub num_visits: f32,
}

impl<E: Env<N>, const N: usize> Node<E, N> {
    pub fn new(parent_id: usize, env: E, is_over: bool, value: f32) -> Self {
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
        }
    }
}

pub struct VanillaMCTS<'a, E: Env<N>, R: Rng, const N: usize> {
    pub root: usize,
    pub nodes: Vec<Node<E, N>>,
    pub rng: &'a mut R,
}

impl<'a, E: Env<N>, R: Rng, const N: usize> VanillaMCTS<'a, E, R, N> {
    pub fn with_capacity(capacity: usize, env: E, rng: &'a mut R) -> Self {
        let mut mcts = Self {
            root: 0,
            nodes: Vec::with_capacity(capacity),
            rng,
        };

        let value = mcts.rollout(env.clone(), env.is_over());
        let root = Node::new(0, env, false, value);
        mcts.nodes.push(root);

        mcts
    }

    pub fn root_node(&self) -> &Node<E, N> {
        self.get_node(self.root)
    }

    pub fn get_node(&self, node_id: usize) -> &Node<E, N> {
        &self.nodes[node_id - self.root]
    }

    fn next_node_id(&self) -> usize {
        self.nodes.len() + self.root
    }

    pub fn best_action(&self) -> E::Action {
        let root = &self.nodes[self.root - self.root];

        let mut best_action = None;
        let mut best_value = -std::f32::INFINITY;

        // assert!(root.children.len() > 0);

        for &(action, child_id) in root.children.iter() {
            let child = &self.nodes[child_id - self.root];
            let value = child.num_visits;
            if value > best_value {
                best_value = value;
                best_action = Some(action);
            }
        }
        // assert!(best_action.is_some());

        best_action.unwrap()
    }

    fn explore(&mut self) {
        let child_id = self.next_node_id();
        let mut node_id = self.root;
        loop {
            // assert!(node_id < self.nodes.len());
            let node = &mut self.nodes[node_id - self.root];
            if node.terminal {
                let v = node.env.reward(node.env.player());
                self.backprop(node_id, v);
                return;
            } else if node.expanded {
                node_id = self.select_best_child(node_id);
            } else {
                match node.actions.next() {
                    Some(action) => {
                        // add to children
                        node.children.push((action, child_id));

                        // create the child node... note we will be modifying num_visits and reward later, so mutable
                        let mut env = node.env.clone();
                        let is_over = env.step(&action);
                        let value = self.rollout(env.clone(), is_over);
                        let child = Node::new(node_id, env, is_over, value);
                        self.nodes.push(child);
                        self.backprop(node_id, -value);
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

    fn rollout(&mut self, mut env: E, mut is_over: bool) -> f32 {
        let player = env.player();
        while !is_over {
            let actions = env.iter_actions();
            let num_actions = actions.count() as u8;
            let i = self.rng.gen_range(0..num_actions);
            let action = env.iter_actions().nth(i as usize).unwrap();
            is_over = env.step(&action);
        }
        env.reward(player)
    }

    fn select_best_child(&mut self, node_id: usize) -> usize {
        // assert!(node_id < self.nodes.len());
        let node = &self.nodes[node_id - self.root];

        let visits = node.num_visits.sqrt();

        let mut best_child_id = self.root;
        let mut best_value = -std::f32::INFINITY;
        for &(_action, child_id) in &node.children {
            let child = &self.nodes[child_id - self.root];
            // NOTE: -child.cum_value because child.cum_value is in opponent's win pct, so we want to convert to ours
            let value = -child.cum_value / child.num_visits + visits / (1.0 + child.num_visits);
            if value > best_value {
                best_child_id = child_id;
                best_value = value;
            }
        }
        // assert!(best_child_id != self.root);
        best_child_id
    }

    fn backprop(&mut self, leaf_node_id: usize, mut value: f32) {
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
        for _ in 0..n {
            self.explore();
        }
        // assert_eq!(self.nodes.len() - start_n, n);
        start.elapsed()
    }
}
