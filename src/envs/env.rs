pub trait HasTurnOrder: Eq + Clone + Copy + std::fmt::Debug {
    fn prev(&self) -> Self;
    fn next(&self) -> Self;
}

pub trait Env: Clone {
    type PlayerId: HasTurnOrder;
    type Action: Eq + Clone + Copy + std::fmt::Debug + Into<usize> + From<usize>;
    type ActionIterator: Iterator<Item = Self::Action>;

    const NAME: &'static str;
    const MAX_NUM_ACTIONS: usize;
    const NUM_PLAYERS: usize;

    fn new() -> Self;
    fn get_state_dims() -> Vec<i64>;

    fn player(&self) -> Self::PlayerId;
    fn is_over(&self) -> bool;
    fn reward(&self, player_id: Self::PlayerId) -> f32;
    fn iter_actions(&self) -> Self::ActionIterator;
    fn step(&mut self, action: &Self::Action) -> bool;
    fn state(&self) -> Vec<f32>;
    fn print(&self);
}
