use crate::env::Env;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

pub struct Node<E: Env + Clone> {
    pub parent: usize,
    pub env: E,
    pub terminal: bool,
    pub expanded: bool,
    pub my_action: bool,
    pub actions: E::ActionIterator,
    pub children: Vec<(E::Action, usize)>,
    pub reward: f32,
    pub num_visits: f32,
}

impl<E: Env + Clone> Node<E> {
    pub fn new_root(player_id: E::PlayerId) -> Self {
        let env = E::new();
        let my_action = env.player() == player_id;
        let actions = env.iter_actions();
        Node {
            parent: 0,
            env,
            terminal: false,
            expanded: false,
            my_action,
            actions,
            children: Vec::new(),
            num_visits: 0.0,
            reward: 0.0,
        }
    }

    pub fn new(parent_id: usize, node: &Self, action: &E::Action, player_id: E::PlayerId) -> Self {
        let mut env = node.env.clone();
        let is_over = env.step(action);
        let my_action = env.player() == player_id;
        let actions = env.iter_actions();
        Node {
            parent: parent_id,
            env,
            terminal: is_over,
            expanded: is_over,
            my_action,
            actions,
            children: Vec::new(),
            num_visits: 0.0,
            reward: 0.0,
        }
    }
}

pub struct MCTS<E: Env + Clone> {
    pub id: E::PlayerId,
    pub root: usize,
    pub nodes: Vec<Node<E>>,
    pub rng: StdRng, // note: this is about the same performance as SmallRng or any of the XorShiftRngs that got moved to the xorshift crate
}

impl<E: Env + Clone> MCTS<E> {
    pub fn with_capacity(id: E::PlayerId, capacity: usize, seed: u64) -> Self {
        let mut nodes = Vec::with_capacity(capacity);
        let root = Node::new_root(id);
        nodes.push(root);
        Self {
            id,
            root: 0,
            nodes,
            rng: StdRng::seed_from_u64(seed),
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
                new_root
            }
            None => {
                let child_node = Node::new(0, &self.nodes[self.root - self.root], action, self.id);
                self.nodes.clear();
                self.nodes.push(child_node);
                0
            }
        };

        self.nodes[0].parent = self.root;
    }

    pub fn best_action(&self) -> E::Action {
        let root = &self.nodes[self.root - self.root];

        let mut best_action_ind = 0;
        let mut best_value = -std::f32::INFINITY;

        for (i, &(_, child_id)) in root.children.iter().enumerate() {
            let child = &self.nodes[child_id - self.root];
            let value = child.reward / child.num_visits;
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
                let reward = node.env.reward(self.id);
                self.backprop(node_id, reward, 1.0);
                return;
            } else if node.expanded {
                node_id = self.select_best_child(node_id);
            } else {
                match node.actions.next() {
                    Some(action) => {
                        let child = self.expand_single_child(node_id, action);
                        self.backprop(node_id, child.reward, 1.0);
                        self.nodes.push(child);
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

        let visits = node.num_visits.log(2.0);

        let mut best_child_id = self.root;
        let mut best_value = -std::f32::INFINITY;
        for &(_action, child_id) in &node.children {
            let child = &self.nodes[child_id - self.root];
            let value = child.reward / child.num_visits + (2.0 * visits / child.num_visits).sqrt();
            if value > best_value {
                best_child_id = child_id;
                best_value = value;
            }
        }
        assert!(best_child_id != self.root);
        best_child_id
    }

    fn expand_single_child(&mut self, node_id: usize, action: E::Action) -> Node<E> {
        let child_id = self.next_node_id();

        let node = &mut self.nodes[node_id - self.root];
        node.children.push((action, child_id));

        // create the child node... note we will be modifying num_visits and reward later, so mutable
        let mut child_node = Node::new(node_id, &node, &action, self.id);

        // rollout child to get initial reward
        let reward = self.rollout(child_node.env.clone());

        // store initial reward & 1 visit
        child_node.num_visits = 1.0;
        child_node.reward = reward;

        child_node
    }

    fn rollout(&mut self, mut env: E) -> f32 {
        // assert!(node_id < self.nodes.len());
        // note: checking if env.is_over() before cloning doesn't make much difference
        let mut is_over = env.is_over();
        while !is_over {
            let action = env.get_random_action(&mut self.rng);
            is_over = env.step(&action);
        }
        env.reward(self.id)
    }

    fn backprop(&mut self, leaf_node_id: usize, reward: f32, num_visits: f32) {
        let mut node_id = leaf_node_id;
        loop {
            // assert!(node_id < self.nodes.len());

            let node = &mut self.nodes[node_id - self.root];

            node.num_visits += num_visits;

            // TODO multiply reward by -1 instead of this if every time
            // note this is reversed because its actually the previous node's action that this node's reward is associated with
            node.reward += if !node.my_action { reward } else { -reward };

            if node_id == self.root {
                break;
            }

            node_id = node.parent;
        }
    }

    pub fn explore_n(&mut self, n: usize) -> (usize, u128) {
        let start = Instant::now();
        let start_n = self.nodes.len();
        for _ in 0..n {
            self.explore();
        }
        (self.nodes.len() - start_n, start.elapsed().as_millis())
    }
}
