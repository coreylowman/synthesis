use synthesis::prelude::*;
const NUM_PLAYABLE_SQUARES: usize = 289;
pub const NUM_MAX_TURNS: usize = 256;
const MAX_NUM_POSSIBLE_MOVES: usize = 128; // TODO: Calculate this!

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct Ginseng {
    board: [i8; NUM_PLAYABLE_SQUARES],
    player: PlayerID,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum PlayerID {
    Host,
    Guest,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Move {
    player: PlayerID,
    from: i8,
    to: i8,
}

pub struct GinsengIterator;

impl Iterator for GinsengIterator {
    type Item = Move;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl From<usize> for Move {
    #[inline]
    fn from(value: usize) -> Self {
        todo!()
    }
}

impl From<Move> for usize {
    #[inline]
    fn from(value: Move) -> Self {
        todo!()
    }
}

impl HasTurnOrder for PlayerID {
    #[inline]
    fn prev(&self) -> Self {
        match self {
            PlayerID::Host => PlayerID::Guest,
            PlayerID::Guest => PlayerID::Host,
        }
    }

    #[inline]
    fn next(&self) -> Self {
        self.prev()
    }
}

impl Game<MAX_NUM_POSSIBLE_MOVES> for Ginseng {
    type PlayerId = PlayerID;
    type Action = Move;
    type ActionIterator = GinsengIterator;
    type Features = ();
    const MAX_NUM_ACTIONS: usize = MAX_NUM_POSSIBLE_MOVES;
    const MAX_TURNS: usize = NUM_MAX_TURNS;
    const NAME: &'static str = "Ginseng Pai Sho";
    const NUM_PLAYERS: usize = 2;
    const DIMS: &'static [i64] = &[];

    fn new() -> Self {
        todo!()
    }

    fn player(&self) -> Self::PlayerId {
        todo!()
    }

    fn is_over(&self) -> bool {
        todo!()
    }

    fn reward(&self, player_id: Self::PlayerId) -> f32 {
        todo!()
    }

    fn iter_actions(&self) -> Self::ActionIterator {
        todo!()
    }

    fn step(&mut self, action: &Self::Action) -> bool {
        todo!()
    }

    fn features(&self) -> Self::Features {
        todo!()
    }

    fn print(&self) {
        todo!()
    }
}
