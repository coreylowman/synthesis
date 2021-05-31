use super::env::{Env, HasTurnOrder};

// note: only 90 bits are actually used...
// 81 bits are taken up for the actual board, then 9 for finished, then 1 for won, then 9 for last play
type BitBoard = u128;

const FAB_NINE: BitBoard = 0b111111111;
const R1: BitBoard = 0b000000111;
const R2: BitBoard = 0b000111000;
const R3: BitBoard = 0b111000000;
const C1: BitBoard = 0b100100100;
const C2: BitBoard = 0b010010010;
const C3: BitBoard = 0b001001001;
const D1: BitBoard = 0b100010001;
const D2: BitBoard = 0b001010100;

const WON_SHIFT: u128 = 91;

const STATUS_COLOR: BitBoard = 1 << 90;
const STATUS_WON: BitBoard = 1 << 91;

const FULL_BOARD: BitBoard = 0x1FFFFFFFFFFFFFFFFFFFF;

const HIGH_BOARD: u8 = 9;

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

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Cell(u8);

impl From<usize> for Cell {
    fn from(x: usize) -> Self {
        Cell(x as u8)
    }
}

impl Into<usize> for Cell {
    fn into(self) -> usize {
        self.0 as usize
    }
}

pub struct ActionIterator(BitBoard);

impl Iterator for ActionIterator {
    type Item = Cell;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            None
        } else {
            let sq = self.0.trailing_zeros();
            self.0 &= self.0.wrapping_sub(1);
            Some(Cell(sq as u8))
        }
    }
}

#[derive(Clone)]
pub struct UltimateTicTacToe {
    my_bb: BitBoard,
    op_bb: BitBoard,
    valid_moves: BitBoard,
    last_action: u8,
    free_play: bool,
}

impl UltimateTicTacToe {
    fn is_complete(&self, full_bb: BitBoard, high: u8) -> bool {
        let bb = (full_bb & (FAB_NINE << (high * 9))) >> (high * 9);
        bb & R1 == R1
            || bb & R2 == R2
            || bb & R3 == R3
            || bb & C1 == C1
            || bb & C2 == C2
            || bb & C3 == C3
            || bb & D1 == D1
            || bb & D2 == D2
    }

    fn action_bitboards(&self) -> BitBoard {
        let mask = if self.free_play {
            FULL_BOARD
        } else {
            FAB_NINE << self.last_action
        };
        self.valid_moves & mask
    }
}

impl Env for UltimateTicTacToe {
    const NAME: &'static str = "UltimateTicTacToe";
    const MAX_NUM_ACTIONS: usize = 9 * 9;
    const NUM_PLAYERS: usize = 2;

    type PlayerId = PlayerId;
    type Action = Cell;
    type ActionIterator = ActionIterator;

    fn new() -> Self {
        Self {
            my_bb: STATUS_COLOR,
            op_bb: 0,
            valid_moves: FULL_BOARD,
            last_action: 0,
            free_play: true,
        }
    }

    fn player(&self) -> Self::PlayerId {
        PlayerId(self.my_bb & STATUS_COLOR == STATUS_COLOR)
    }

    fn is_over(&self) -> bool {
        self.op_bb & STATUS_WON != 0 || self.valid_moves & FULL_BOARD == 0
    }

    fn reward(&self, player_id: Self::PlayerId) -> f32 {
        if self.op_bb & STATUS_WON != 0 {
            if self.player() == player_id {
                -1.0
            } else {
                1.0
            }
        } else {
            let (my_bb, op_bb) = if self.player() == player_id {
                (self.my_bb, self.op_bb)
            } else {
                (self.op_bb, self.my_bb)
            };
            let my_highs = (my_bb & (FAB_NINE << 81)).count_ones();
            let op_highs = (op_bb & (FAB_NINE << 81)).count_ones();
            match my_highs.cmp(&op_highs) {
                std::cmp::Ordering::Greater => 1.0,
                std::cmp::Ordering::Equal => 0.0,
                std::cmp::Ordering::Less => -1.0,
            }
        }
    }

    fn iter_actions(&self) -> Self::ActionIterator {
        ActionIterator(self.action_bitboards())
    }

    fn step(&mut self, action: &Self::Action) -> bool {
        let action = action.0;
        let high = action / 9;
        let low = action % 9;

        // mark the spot
        self.my_bb |= 1u128 << action;

        // remove move from valid moves
        self.valid_moves ^= 1u128 << action;

        // check if this low board is complete
        // TODO cache this?
        // TODO how to improve this?
        if self.is_complete(self.my_bb, high) {
            // mark the corresponding spot in the high board
            self.valid_moves |= 1 << (81 + high);
            self.my_bb |= 1 << (81 + high);

            // remove this section from valid moves
            self.valid_moves &= !(FAB_NINE << (high * 9));

            // check if the high board is complete now
            let game_won = self.is_complete(self.my_bb, HIGH_BOARD);
            self.my_bb |= (game_won as u128) << WON_SHIFT;
        }

        self.last_action = low * 9;
        let next_complete = self.valid_moves & (1 << (81 + low)) != 0;
        let next_full = self.valid_moves & (FAB_NINE << self.last_action) == 0;
        self.free_play = next_complete || next_full;

        std::mem::swap(&mut self.my_bb, &mut self.op_bb);

        self.is_over()
    }

    fn get_state_dims() -> Vec<i64> {
        vec![1, 2, 9, 9]
    }

    fn state(&self) -> Vec<f32> {
        // TODO include valid moves plane
        let mut s = Vec::with_capacity(2 * 9 * 9);
        for bb in &[self.my_bb, self.op_bb] {
            for row in 0..9 {
                for col in 0..9 {
                    let i = 27 * (row / 3) + 3 * (row % 3) + 9 * (col / 3) + col % 3;
                    let index = 1 << i;
                    if bb & index != 0 {
                        s.push(1.0);
                    } else {
                        s.push(0.0);
                    }
                }
            }
        }
        s
    }

    fn print(&self) {
        let (my_char, op_char) = match self.player().0 {
            true => ("X", "o"),
            false => ("o", "X"),
        };

        if self.is_over() {
            println!("{:?} r{:?}", my_char, self.reward(self.player()));
        } else {
            println!("{:?} to play", my_char);
            println!(
                "Available Actions: {:?}",
                self.iter_actions().collect::<Vec<Cell>>()
            );
        }
        let my_highs = (self.my_bb & (FAB_NINE << 81)) >> 81;
        let op_highs = (self.op_bb & (FAB_NINE << 81)) >> 81;
        println!("{}:{:09b} {}:{:09b}", my_char, my_highs, op_char, op_highs);

        for row in (0..9).rev() {
            for col in 0..9 {
                let i = 27 * (row / 3) + 3 * (row % 3) + 9 * (col / 3) + col % 3;
                let index = 1 << i;
                if col % 3 == 0 {
                    print!("|")
                }
                print!(
                    "{}",
                    if self.my_bb & index != 0 {
                        my_char
                    } else if self.op_bb & index != 0 {
                        op_char
                    } else {
                        "."
                    }
                );
                if col == 8 {
                    print!("|")
                }
            }
            println!();
            if row % 3 == 0 && row > 0 {
                println!("-------------");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_rollout() {
        let mut game = UltimateTicTacToe::new();
        game.print();

        loop {
            let actions: Vec<Cell> = game.iter_actions().collect();
            let action = actions.choose(&mut thread_rng()).unwrap();
            if game.step(action) {
                break;
            }
            println!("{:?}", action);
            game.print();
        }
        game.print();

        assert!(false);
    }
}
