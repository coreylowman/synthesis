use synthesis::game::*;

/*
+----------------------------+
| 6 13 20 27 34 41 48 55 62 |
| 5 12 19 26 33 40 47 54 61 |
| 4 11 18 25 32 39 46 53 60 |
| 3 10 17 24 31 38 45 52 59 |
| 2  9 16 23 30 37 44 51 58 |
| 1  8 15 22 29 36 43 50 57 |
| 0  7 14 21 28 35 42 49 56 | 63
+----------------------------+
*/

const WIDTH: usize = 9;
const HEIGHT: usize = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlayerId {
    Red,
    Black,
}

impl HasTurnOrder for PlayerId {
    fn prev(&self) -> Self {
        self.next()
    }

    fn next(&self) -> Self {
        match self {
            PlayerId::Black => PlayerId::Red,
            PlayerId::Red => PlayerId::Black,
        }
    }
}

const FAB_COL: u64 = 0b1111111;
const FAB_ROW: u64 = (1 << (7 * 0))
    | (1 << (7 * 1))
    | (1 << (7 * 2))
    | (1 << (7 * 3))
    | (1 << (7 * 4))
    | (1 << (7 * 5))
    | (1 << (7 * 6))
    | (1 << (7 * 7))
    | (1 << (7 * 8));

const COLS: [u64; WIDTH] = [
    FAB_COL << (7 * 0),
    FAB_COL << (7 * 1),
    FAB_COL << (7 * 2),
    FAB_COL << (7 * 3),
    FAB_COL << (7 * 4),
    FAB_COL << (7 * 5),
    FAB_COL << (7 * 6),
    FAB_COL << (7 * 7),
    FAB_COL << (7 * 8),
];

const ROWS: [u64; HEIGHT] = [
    FAB_ROW << 0,
    FAB_ROW << 1,
    FAB_ROW << 2,
    FAB_ROW << 3,
    FAB_ROW << 4,
    FAB_ROW << 5,
    FAB_ROW << 6,
];

const D1_MASK: u64 = (COLS[0] | COLS[1] | COLS[2] | COLS[3] | COLS[4] | COLS[5])
    & (ROWS[3] | ROWS[4] | ROWS[5] | ROWS[6]);
const D2_MASK: u64 = (COLS[0] | COLS[1] | COLS[2] | COLS[3] | COLS[4] | COLS[5])
    & (ROWS[0] | ROWS[1] | ROWS[2] | ROWS[3]);
const H_MASK: u64 = COLS[0] | COLS[1] | COLS[2] | COLS[3] | COLS[4] | COLS[5];
const V_MASK: u64 = ROWS[0] | ROWS[1] | ROWS[2] | ROWS[3];

const fn won(bb: u64) -> bool {
    let d1 = bb & (bb >> 6) & (bb >> 12) & (bb >> 18) & D1_MASK;
    let d2 = bb & (bb >> 8) & (bb >> 16) & (bb >> 24) & D2_MASK;
    let h = bb & (bb >> 7) & (bb >> 14) & (bb >> 21) & H_MASK;
    let v = bb & (bb >> 1) & (bb >> 2) & (bb >> 3) & V_MASK;
    v + h + d1 + d2 > 0
}

/*

use std::arch::x86_64::*;

fn fast_won(bb: u64) -> bool {
    unsafe {
        let bbx4 = _mm256_set1_epi64x(bb as i64);
        let maskx4 = _mm256_set_epi64x(D1_MASK as i64, D2_MASK as i64, H_MASK as i64, V_MASK as i64);
        let shift1 = _mm256_set_epi64x(6, 8, 7, 1);
        let shift2 = _mm256_set_epi64x(12, 16, 14, 2);
        let shift3 = _mm256_set_epi64x(18, 24, 21, 3);
        let a = _mm256_and_si256(bbx4, maskx4);
        let b = _mm256_srlv_epi64(bbx4, shift1);
        let c = _mm256_and_si256(a, b);
        let d = _mm256_srlv_epi64(bbx4, shift2);
        let e = _mm256_and_si256(c, d);
        let f = _mm256_srlv_epi64(bbx4, shift3);
        let res = _mm256_testz_si256(e, f);
        res == 0
    }
}
*/

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct Connect4 {
    my_bb: u64,
    op_bb: u64,
    height: [u8; WIDTH],
    player: PlayerId,
}

impl std::hash::Hash for Connect4 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.my_bb);
        state.write_u64(self.op_bb);
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Column(u8);

impl From<usize> for Column {
    fn from(x: usize) -> Self {
        Column(x as u8)
    }
}

impl Into<usize> for Column {
    fn into(self) -> usize {
        self.0 as usize
    }
}

pub struct FreeColumns {
    height: [u8; WIDTH],
    col: u8,
}

impl Iterator for FreeColumns {
    type Item = Column;
    fn next(&mut self) -> Option<Self::Item> {
        if self.col == WIDTH as u8 {
            return None;
        }

        while self.col < WIDTH as u8 {
            if self.height[self.col as usize] < HEIGHT as u8 {
                let item = Some(Column(self.col));
                self.col += 1;
                return item;
            }
            self.col += 1;
        }

        None
    }
}

impl Connect4 {
    fn winner(&self) -> Option<PlayerId> {
        if won(self.op_bb) {
            Some(self.player.next())
        } else {
            None
        }
    }
}

impl Game<WIDTH> for Connect4 {
    const NAME: &'static str = "Connect4";
    const NUM_PLAYERS: usize = 2;
    const MAX_TURNS: usize = 63;

    type PlayerId = PlayerId;
    type Action = Column;
    type ActionIterator = FreeColumns;

    fn new() -> Self {
        Self {
            my_bb: 0,
            op_bb: 0,
            height: [0; WIDTH],
            player: PlayerId::Red,
        }
    }

    fn player(&self) -> Self::PlayerId {
        self.player
    }

    fn is_over(&self) -> bool {
        self.winner().is_some() || (0..WIDTH).all(|col| self.height[col] == HEIGHT as u8)
    }

    fn reward(&self, player_id: Self::PlayerId) -> f32 {
        // assert!(self.is_over());

        match self.winner() {
            Some(winner) => {
                if winner == player_id {
                    1.0
                } else {
                    -1.0
                }
            }
            None => 0.0,
        }
    }

    fn iter_actions(&self) -> Self::ActionIterator {
        FreeColumns {
            height: self.height,
            col: 0,
        }
    }

    fn step(&mut self, action: &Self::Action) -> bool {
        let col: usize = (*action).into();

        // assert!(self.height[col] < HEIGHT as u8);

        self.my_bb ^= 1 << (self.height[col] + (HEIGHT as u8) * (col as u8));
        self.height[col] += 1;

        std::mem::swap(&mut self.my_bb, &mut self.op_bb);
        self.player = self.player.next();

        self.is_over()
    }

    const DIMS: &'static [i64] = &[1, 1, HEIGHT as i64, WIDTH as i64];
    type Features = [[[f32; WIDTH]; HEIGHT]; 1];
    fn features(&self) -> Self::Features {
        let mut s = Self::Features::default();
        for row in 0..HEIGHT {
            for col in 0..WIDTH {
                let index = 1 << (row + HEIGHT * col);
                if self.my_bb & index != 0 {
                    s[0][row][col] = 1.0;
                } else if self.op_bb & index != 0 {
                    s[0][row][col] = -1.0;
                } else {
                    s[0][row][col] = -0.1;
                }
            }
        }
        for col in 0..WIDTH {
            let h = self.height[col] as usize;
            if h < HEIGHT {
                s[0][h][col] = 0.1;
            }
        }
        s
    }

    fn print(&self) {
        if self.is_over() {
            println!("{:?} won", self.winner());
        } else {
            println!("{:?} to play", self.player);
            println!(
                "Available Actions: {:?}",
                self.iter_actions().collect::<Vec<Column>>()
            );
        }

        let (my_char, op_char) = match self.player {
            PlayerId::Black => ("B", "r"),
            PlayerId::Red => ("r", "B"),
        };

        for row in (0..HEIGHT).rev() {
            for col in 0..WIDTH {
                let index = 1 << (row + HEIGHT * col);
                print!(
                    "{} ",
                    if self.my_bb & index != 0 {
                        my_char
                    } else if self.op_bb & index != 0 {
                        op_char
                    } else {
                        "."
                    }
                );
            }
            println!();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_wins() {
        let mut game = Connect4::new();
        assert!(!game.step(&Column(0)));
        assert!(!game.step(&Column(1)));
        assert!(!game.step(&Column(0)));
        assert!(!game.step(&Column(1)));
        assert!(!game.step(&Column(0)));
        assert!(!game.step(&Column(1)));
        assert!(game.step(&Column(0)));
        assert!(game.is_over());
        assert_eq!(game.winner(), Some(PlayerId::Red));
        assert_eq!(game.reward(game.player()), -1.0);
        assert_eq!(game.player(), PlayerId::Black);
        assert_eq!(game.reward(PlayerId::Black), -1.0);
        assert_eq!(game.reward(PlayerId::Red), 1.0);
    }

    #[test]
    fn test_second_wins() {
        let mut game = Connect4::new();
        assert!(!game.step(&Column(0)));
        assert!(!game.step(&Column(1)));
        assert!(!game.step(&Column(2)));
        assert!(!game.step(&Column(1)));
        assert!(!game.step(&Column(2)));
        assert!(!game.step(&Column(1)));
        assert!(!game.step(&Column(2)));
        assert!(game.step(&Column(1)));
        assert!(game.is_over());
        assert_eq!(game.winner(), Some(PlayerId::Black));
        assert_eq!(game.reward(game.player()), -1.0);
        assert_eq!(game.player(), PlayerId::Red);
        assert_eq!(game.reward(PlayerId::Black), 1.0);
        assert_eq!(game.reward(PlayerId::Red), -1.0);
    }

    #[test]
    fn test_draw() {
        /*
        +-------------------+
        | r b r b r b r b r |
        | r b r b r b r b b |
        | r b r b r b r b r |
        | b r b r b r b r b |
        | b r b r b r b r r |
        | r b r b r b r b b |
        | r b r b r b r b r |
        +-------------------+
        */

        let mut game = Connect4::new();
        assert!(!game.step(&Column(0)));
        assert!(!game.step(&Column(1)));
        assert!(!game.step(&Column(0)));
        assert!(!game.step(&Column(1)));
        assert!(!game.step(&Column(1)));
        assert!(!game.step(&Column(0)));
        assert!(!game.step(&Column(1)));
        assert!(!game.step(&Column(0)));
        assert!(!game.step(&Column(0)));
        assert!(!game.step(&Column(1)));
        assert!(!game.step(&Column(0)));
        assert!(!game.step(&Column(1)));

        assert!(game.iter_actions().position(|c| c == Column(0)).is_some());
        assert!(!game.step(&Column(0)));
        assert!(game.iter_actions().position(|c| c == Column(0)).is_none());

        assert!(game.iter_actions().position(|c| c == Column(1)).is_some());
        assert!(!game.step(&Column(1)));
        assert!(game.iter_actions().position(|c| c == Column(1)).is_none());

        assert!(!game.step(&Column(2)));
        assert!(!game.step(&Column(3)));
        assert!(!game.step(&Column(2)));
        assert!(!game.step(&Column(3)));
        assert!(!game.step(&Column(3)));
        assert!(!game.step(&Column(2)));
        assert!(!game.step(&Column(3)));
        assert!(!game.step(&Column(2)));
        assert!(!game.step(&Column(2)));
        assert!(!game.step(&Column(3)));
        assert!(!game.step(&Column(2)));
        assert!(!game.step(&Column(3)));

        assert!(game.iter_actions().position(|c| c == Column(2)).is_some());
        assert!(!game.step(&Column(2)));
        assert!(game.iter_actions().position(|c| c == Column(2)).is_none());

        assert!(game.iter_actions().position(|c| c == Column(3)).is_some());
        assert!(!game.step(&Column(3)));
        assert!(game.iter_actions().position(|c| c == Column(3)).is_none());

        assert!(!game.step(&Column(4)));
        assert!(!game.step(&Column(5)));
        assert!(!game.step(&Column(4)));
        assert!(!game.step(&Column(5)));
        assert!(!game.step(&Column(5)));
        assert!(!game.step(&Column(4)));
        assert!(!game.step(&Column(5)));
        assert!(!game.step(&Column(4)));
        assert!(!game.step(&Column(4)));
        assert!(!game.step(&Column(5)));
        assert!(!game.step(&Column(4)));
        assert!(!game.step(&Column(5)));

        assert!(game.iter_actions().position(|c| c == Column(4)).is_some());
        assert!(!game.step(&Column(4)));
        assert!(game.iter_actions().position(|c| c == Column(4)).is_none());

        assert!(game.iter_actions().position(|c| c == Column(5)).is_some());
        assert!(!game.step(&Column(5)));
        assert!(game.iter_actions().position(|c| c == Column(5)).is_none());

        assert!(!game.step(&Column(6)));
        assert!(!game.step(&Column(7)));
        assert!(!game.step(&Column(6)));
        assert!(!game.step(&Column(7)));
        assert!(!game.step(&Column(7)));
        assert!(!game.step(&Column(6)));
        assert!(!game.step(&Column(7)));
        assert!(!game.step(&Column(6)));
        assert!(!game.step(&Column(6)));
        assert!(!game.step(&Column(7)));
        assert!(!game.step(&Column(6)));
        assert!(!game.step(&Column(7)));

        assert!(game.iter_actions().position(|c| c == Column(6)).is_some());
        assert!(!game.step(&Column(6)));
        assert!(game.iter_actions().position(|c| c == Column(6)).is_none());

        assert!(game.iter_actions().position(|c| c == Column(7)).is_some());
        assert!(!game.step(&Column(7)));
        assert!(game.iter_actions().position(|c| c == Column(7)).is_none());

        assert!(!game.step(&Column(8)));
        assert!(!game.step(&Column(8)));
        assert!(!game.step(&Column(8)));
        assert!(!game.step(&Column(8)));
        assert!(!game.step(&Column(8)));
        assert!(!game.step(&Column(8)));
        assert!(game.iter_actions().position(|c| c == Column(8)).is_some());
        assert!(game.step(&Column(8)));
        assert!(game.is_over());
        assert_eq!(game.winner(), None);
        assert_eq!(game.reward(PlayerId::Red), 0.0);
        assert_eq!(game.reward(PlayerId::Black), 0.0);
    }

    #[test]
    fn test_horz_wins() {
        for row in 0..HEIGHT {
            let mut bb =
                (1 << (row + 0)) | (1 << (row + 7)) | (1 << (row + 14)) | (1 << (row + 21));
            for _i in 0..6 {
                assert!(won(bb));
                bb <<= 7;
            }
        }
    }

    #[test]
    fn test_vert_wins() {
        for col in 0..WIDTH {
            let mut bb = (1 << (7 * col + 0))
                | (1 << (7 * col + 1))
                | (1 << (7 * col + 2))
                | (1 << (7 * col + 3));
            for _i in 0..4 {
                assert!(won(bb));
                bb <<= 1;
            }
        }
    }

    #[test]
    fn test_d1_wins() {
        for row in 3..HEIGHT {
            let mut bb = (1 << row) | (1 << (row + 6)) | (1 << (row + 12)) | (1 << (row + 18));
            for _i in 0..6 {
                assert!(won(bb));
                bb <<= 7;
            }
        }
    }

    #[test]
    fn test_d2_wins() {
        for col in 0..6 {
            let mut bb = (1 << (7 * col + 0))
                | (1 << (7 * (col + 1) + 1))
                | (1 << (7 * (col + 2) + 2))
                | (1 << (7 * (col + 3) + 3));
            for _i in 0..4 {
                assert!(won(bb));
                bb <<= 1;
            }
        }
    }
}
