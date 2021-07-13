use crate::envs::Connect4;
use ragz::prelude::*;
use slimnn::{Activation, Conv2d, Linear, ReLU, Softmax};
use tch::{
    self,
    nn::{self, ConvConfig},
    Tensor,
};

fn to_float<const W: usize, const H: usize, const C: usize>(
    x_bool: &[[[bool; W]; H]; C],
) -> [[[f32; W]; H]; C] {
    let mut x_f32 = [[[0.0; W]; H]; C];
    for i in 0..C {
        let bool_i = &x_bool[i];
        let f32_i = &mut x_f32[i];
        for j in 0..H {
            let bool_j = &bool_i[j];
            let f32_j = &mut f32_i[j];
            for k in 0..W {
                if bool_j[k] {
                    f32_j[k] = 1.0;
                }
            }
        }
    }
    x_f32
}

pub struct Connect4Net {
    l_1: nn::Linear,
    l_2: nn::Linear,
    l_3: nn::Linear,
    l_4: nn::Linear,
}

impl NNPolicy<Connect4, { Connect4::MAX_NUM_ACTIONS }> for Connect4Net {
    fn new(vs: &nn::VarStore) -> Self {
        let root = &vs.root();
        let state_dims = Connect4::get_state_dims();
        assert!(state_dims.len() == 4);
        assert!(&state_dims == &[1, 3, 7, 9]);
        Self {
            l_1: nn::linear(root / "l_1", 189, 256, Default::default()),
            l_2: nn::linear(root / "l_2", 256, 256, Default::default()),
            l_3: nn::linear(root / "l_3", 256, 256, Default::default()),
            l_4: nn::linear(root / "l_4", 256, 10, Default::default()),
        }
    }

    fn forward(&self, xs: &Tensor) -> (Tensor, Tensor) {
        let xs = xs
            .flat_view()
            .apply(&self.l_1)
            .relu()
            .apply(&self.l_2)
            .relu()
            .apply(&self.l_3)
            .relu()
            .apply(&self.l_4);
        let mut ts = xs.split_with_sizes(&[9, 1], -1);
        let value = ts.pop().unwrap();
        let logits = ts.pop().unwrap();
        (logits, value)
    }
}

impl Policy<Connect4, { Connect4::MAX_NUM_ACTIONS }> for Connect4Net {
    fn eval(&mut self, env: &Connect4) -> ([f32; Connect4::MAX_NUM_ACTIONS], f32) {
        let state = env.state();
        let xs = to_float(&state);
        let t = tensor(&xs, &[1, 3, 7, 9], tch::Kind::Float);
        let (logits, value) = self.forward(&t);
        let mut policy = [0.0f32; Connect4::MAX_NUM_ACTIONS];
        logits
            // .softmax(-1, tch::Kind::Float)
            .copy_data(&mut policy, Connect4::MAX_NUM_ACTIONS);
        let value = f32::from(&value).clamp(-1.0, 1.0);
        (policy, value)
    }
}

// #[derive(Default)]
// pub struct SlimC4Net {
//     c_1: Conv2d<2, 5, 3, 0, 0, 2>,
//     p_1: Linear<60, 32>,
//     p_2: Linear<32, { Connect4::MAX_NUM_ACTIONS }>,
//     v_1: Linear<60, 32>,
//     v_2: Linear<32, 1>,
// }

// impl SlimC4Net {
//     fn forward(&self, x: &[[[f32; 9]; 7]; 3]) -> ([f32; Connect4::MAX_NUM_ACTIONS], f32) {
//         let x = self.c_1.forward::<9, 7, 4, 3>(x);
//         let x = ReLU.apply_3d(&x);

//         let x: [f32; 60] = unsafe { std::mem::transmute(x) };

//         let px = self.p_1.forward(&x);
//         let px = ReLU.apply_1d(&px);
//         let logits = self.p_2.forward(&px);

//         let vx = self.v_1.forward(&x);
//         let vx = ReLU.apply_1d(&vx);
//         let value = self.v_2.forward(&vx)[0].tanh();

//         (logits, value)
//     }
// }

// impl Policy<Connect4, { Connect4::MAX_NUM_ACTIONS }> for SlimC4Net {
//     fn eval(&mut self, env: &Connect4) -> ([f32; Connect4::MAX_NUM_ACTIONS], f32) {
//         let state = env.state();
//         let x = to_float(&state);
//         let (logits, value) = self.forward(&x);
//         // let policy = Softmax.apply_1d(&logits);
//         (logits, value)
//     }
// }
