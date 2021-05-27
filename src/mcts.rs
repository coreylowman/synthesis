use crate::env::Env;
use std::time::{Duration, Instant};

const C_PUCT: f32 = 4.0;

pub struct Node<E: Env> {
    pub parent: usize,
    pub env: E,
    pub terminal: bool,
    pub expanded: bool,
    pub actions: E::ActionIterator,
    pub children: Vec<(E::Action, usize)>,
    pub cum_value: f32,
    pub num_visits: f32,
    pub action_probs: Vec<f32>,
    pub value: f32,
}

impl<E: Env> Node<E> {
    pub fn new(
        parent_id: usize,
        env: E,
        is_over: bool,
        action_probs: Vec<f32>,
        value: f32,
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
            value,
        }
    }
}

pub trait Policy<E: Env> {
    fn eval(&mut self, state: &Vec<f32>) -> (Vec<f32>, f32);
}

pub struct MCTS<'a, E: Env, P: Policy<E>> {
    pub root: usize,
    pub nodes: Vec<Node<E>>,
    pub states: Vec<Vec<f32>>,
    pub policy: &'a mut P,
}

impl<'a, E: Env, P: Policy<E>> MCTS<'a, E, P> {
    pub fn with_capacity(capacity: usize, policy: &'a mut P) -> Self {
        let env = E::new();
        let state = env.state();
        let (action_probs, value) = policy.eval(&state);
        let root = Node::new(0, env, false, action_probs, value);

        let mut nodes = Vec::with_capacity(capacity);
        let mut states = Vec::with_capacity(capacity);
        nodes.push(root);
        states.push(state);

        Self {
            root: 0,
            nodes,
            states,
            policy,
        }
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
                drop(self.states.drain(0..new_root - self.root));
                new_root
            }
            None => {
                let mut env = self.nodes[self.root - self.root].env.clone();
                let is_over = env.step(action);
                let state = env.state();
                let (action_probs, value) = self.policy.eval(&state);
                let child_node = Node::new(0, env, is_over, action_probs, value);
                self.nodes.clear();
                self.states.clear();
                self.nodes.push(child_node);
                self.states.push(state);
                0
            }
        };

        self.nodes[0].parent = self.root;
    }

    pub fn root_node(&self) -> &Node<E> {
        self.get_node(self.root)
    }

    pub fn root_state(&self) -> &Vec<f32> {
        &self.states[self.root - self.root]
    }

    pub fn get_node(&self, node_id: usize) -> &Node<E> {
        &self.nodes[node_id - self.root]
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
                let v = node.value;
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
                        let state = env.state();
                        let (action_probs, value) = self.policy.eval(&state);
                        let child = Node::new(node_id, env, is_over, action_probs, value);
                        self.nodes.push(child);
                        self.states.push(state);
                        self.backprop(node_id, -value);
                        return;
                    }
                    None => {
                        node.expanded = true;

                        // renormalize probabilities based on valid actions
                        let total = node.children.len() as f32;
                        for p in node.action_probs.iter_mut() {
                            *p /= total;
                        }

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
        let mut best_value = -std::f32::INFINITY;
        for &(action, child_id) in &node.children {
            let child = &self.nodes[child_id - self.root];
            // NOTE: -child.cum_value because child.cum_value is in opponent's win pct, so we want to convert to ours
            let value = -child.cum_value / child.num_visits
                + C_PUCT * node.action_probs[action.into()] * visits / (1.0 + child.num_visits);
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
