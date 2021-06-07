use ragz::env::*;

type BitBoard = u64;

const COL_1: BitBoard = 0x0101010101010101u64;
const NOT_COL_1: BitBoard = !COL_1;
const COL_8: BitBoard = COL_1 << 7;
const NOT_COL_8: BitBoard = !COL_8;
const ROW_1: BitBoard = 0xFFu64;
const ROW_2: BitBoard = ROW_1 << 8;
const ROW_7: BitBoard = ROW_1 << 48;
const ROW_8: BitBoard = ROW_1 << 56;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct PlayerId(bool);

impl HasTurnOrder for PlayerId {
    fn prev(&self) -> Self {
        Self(!self.0)
    }

    fn next(&self) -> Self {
        Self(!self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FromToAction(u32, u32);

impl From<usize> for FromToAction {
    fn from(v: usize) -> Self {
        Self((v / 64) as u32, (v % 64) as u32)
    }
}

impl Into<usize> for FromToAction {
    fn into(self) -> usize {
        (self.0 * 64 + self.1) as usize
    }
}

pub struct ActionIterator(BitBoard, BitBoard);
impl Iterator for ActionIterator {
    type Item = FromToAction;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            return None;
        }

        let from_sq = self.0.trailing_zeros();
        let to_sq = self.1.trailing_zeros();

        self.0 &= self.0.wrapping_sub(1);
        self.1 &= self.1.wrapping_sub(1);

        Some(FromToAction(from_sq, to_sq))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.0 as usize, None)
    }
}

#[derive(Eq, PartialEq, Clone)]
pub struct PlayerInfo {
    id: PlayerId,
    won: bool,
    fwd_shift: u8,
    right_shift: u8,
    left_shift: u8,
    ty_shift: u8,
}

#[derive(Eq, PartialEq, Clone)]
pub struct Breakthrough {
    my_bb: BitBoard,
    op_bb: BitBoard,
    me: PlayerInfo,
    op: PlayerInfo,
}

impl Breakthrough {
    fn action_bitboards(&self) -> (BitBoard, BitBoard, BitBoard) {
        let op_winners =
            self.op_bb & ((ROW_1 << self.op.ty_shift).rotate_right(self.op.fwd_shift as u32));

        let fwd_to = self.my_bb.rotate_left(self.me.fwd_shift as u32) & !self.my_bb & !self.op_bb;
        let right_to =
            (self.my_bb & NOT_COL_8).rotate_left(self.me.right_shift as u32) & !self.my_bb;
        let left_to = (self.my_bb & NOT_COL_1).rotate_left(self.me.left_shift as u32) & !self.my_bb;

        let fwd_win = fwd_to & (ROW_1 << self.me.ty_shift);
        let right_win = right_to & (ROW_1 << self.me.ty_shift);
        let left_win = left_to & (ROW_1 << self.me.ty_shift);

        let fwd_not_lose = fwd_to & op_winners;
        let right_not_lose = right_to & op_winners;
        let left_not_lose = left_to & op_winners;

        if fwd_win != 0 || right_win != 0 || left_win != 0 {
            (fwd_win, right_win, left_win)
        } else if fwd_not_lose != 0 || right_not_lose != 0 || left_not_lose != 0 {
            (fwd_not_lose, right_not_lose, left_not_lose)
        } else {
            (fwd_to, right_to, left_to)
        }
    }
}

impl Env<{ 64 * 3 }> for Breakthrough {
    const NAME: &'static str = "Breakthrough";
    const NUM_PLAYERS: usize = 2;

    type State = [[[bool; 8]; 8]; 2];
    type PlayerId = PlayerId;
    type Action = FromToAction;
    type ActionIterator =
        std::iter::Chain<std::iter::Chain<ActionIterator, ActionIterator>, ActionIterator>;

    fn new() -> Breakthrough {
        Breakthrough {
            my_bb: ROW_1 | ROW_2,
            op_bb: ROW_7 | ROW_8,
            me: PlayerInfo {
                id: PlayerId(true),
                left_shift: 7,
                fwd_shift: 8,
                right_shift: 9,
                won: false,
                ty_shift: 56,
            },
            op: PlayerInfo {
                id: PlayerId(false),
                left_shift: 55,
                fwd_shift: 56,
                right_shift: 57,
                won: false,
                ty_shift: 0,
            },
        }
    }

    fn player(&self) -> Self::PlayerId {
        self.me.id
    }

    fn is_over(&self) -> bool {
        self.op.won
    }

    fn reward(&self, player_id: Self::PlayerId) -> f32 {
        // assert!(self.op.won);
        if self.op.id == player_id {
            1.0
        } else {
            -1.0
        }
    }

    fn iter_actions(&self) -> Self::ActionIterator {
        let (fwd_to, right_to, left_to) = self.action_bitboards();
        let fwd_from = fwd_to.rotate_right(self.me.fwd_shift as u32);
        let right_from = right_to.rotate_right(self.me.right_shift as u32);
        let left_from = left_to.rotate_right(self.me.left_shift as u32);
        ActionIterator(fwd_from, fwd_to)
            .chain(ActionIterator(right_from, right_to))
            .chain(ActionIterator(left_from, left_to))
    }

    fn step(&mut self, action: &Self::Action) -> bool {
        // assert!(self.actions().contains(action));

        let from_sq = action.0;
        let to_sq = action.1;

        // note: doing an xor here instead makes no difference, even if you use the same mask for op_bb & my_bb
        self.op_bb &= !(1 << to_sq);
        self.my_bb = (self.my_bb | (1 << to_sq)) & !(1 << from_sq);

        // note: comparing ty_shift to to_bb is faster than doing a ty_shift <= to_sq < ty_max
        self.me.won = (ROW_1 << self.me.ty_shift) & (1 << to_sq) != 0 || self.op_bb == 0;

        std::mem::swap(&mut self.me, &mut self.op);
        std::mem::swap(&mut self.my_bb, &mut self.op_bb);

        self.op.won
    }

    fn get_state_dims() -> Vec<i64> {
        vec![1, 2, 8, 8]
    }

    fn state(&self) -> Self::State {
        // TODO if color is black we need to flip vertically
        let mut s = [[[false; 8]; 8]; 2];
        let mut i = 0;
        for bb in &[self.my_bb, self.op_bb] {
            for row in 0..8 {
                for col in 0..8 {
                    let index = 1 << (row + 8 * col);
                    if bb & index != 0 {
                        s[i][row][col] = true;
                    }
                }
            }
            i += 1;
        }
        s
    }

    fn print(&self) {
        println!("TODO");
    }
}
