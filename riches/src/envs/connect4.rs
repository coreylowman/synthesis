use ragz::env::*;

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

#[derive(Clone)]
pub struct Connect4 {
    my_bb: u64,
    op_bb: u64,
    height: [u8; WIDTH],
    player: PlayerId,
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

impl Env<WIDTH> for Connect4 {
    const NAME: &'static str = "Connect4";
    const NUM_PLAYERS: usize = 2;

    type PlayerId = PlayerId;
    type Action = Column;
    type ActionIterator = FreeColumns;
    type State = [[[bool; WIDTH]; HEIGHT]; 2];

    fn new() -> Self {
        Self {
            my_bb: 0,
            op_bb: 0,
            height: [0; WIDTH],
            player: PlayerId::Red,
        }
    }

    fn restore(state: &Self::State) -> Self {
        let mut my_bb = 0;
        let mut op_bb = 0;
        let mut height = [0; WIDTH];
        let mut num = 0;
        for row in 0..HEIGHT {
            for col in 0..WIDTH {
                let index = 1 << (row + HEIGHT * col);
                if state[0][row][col] {
                    my_bb |= index;
                    height[col] = row as u8;
                    num += 1;
                } else if state[1][row][col] {
                    op_bb |= index;
                    height[col] = row as u8;
                    num += 1;
                };
            }
        }
        let player = if num % 2 == 0 {
            PlayerId::Red
        } else {
            PlayerId::Black
        };
        let o = Self {
            my_bb,
            op_bb,
            height,
            player,
        };
        assert_eq!(o.state(), *state);
        o
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

        self.my_bb ^= 1 << (self.height[col] + (HEIGHT as u8) * (col as u8));
        self.height[col] += 1;

        std::mem::swap(&mut self.my_bb, &mut self.op_bb);
        self.player = self.player.next();

        self.is_over()
    }

    fn get_state_dims() -> Vec<i64> {
        vec![1, 2, HEIGHT as i64, WIDTH as i64]
    }

    fn state(&self) -> Self::State {
        let mut s = [[[false; WIDTH]; HEIGHT]; 2];
        for row in 0..HEIGHT {
            for col in 0..WIDTH {
                let index = 1 << (row + HEIGHT * col);
                if self.my_bb & index != 0 {
                    s[0][row][col] = true;
                } else if self.op_bb & index != 0 {
                    s[1][row][col] = true;
                }
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
