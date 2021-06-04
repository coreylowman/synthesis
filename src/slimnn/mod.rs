mod activations;
mod conv;
mod linear;

pub use activations::{Activation, ReLU, Softmax, Tanh};
pub use conv::Conv2d;
pub use linear::Linear;

// struct C4Net {
//     conv_1: Conv2d<2, 256, 4, 0, 1>,
//     fc_1: Linear<{ 256 * 3 * 3 }, 64>,
//     fc_2: Linear<64, 64>,
//     p_1: Linear<64, 64>,
//     p_2: Linear<64, 7>,
//     v_1: Linear<64, 64>,
//     v_2: Linear<64, 1>,
// }

// impl C4Net {
//     fn eval(&self, xs: &Vec<f32>) -> (Vec<f32>, f32) {
//         assert_eq!(xs.len(), 2 * 6 * 7);
//         // let x: [f32; 2 * 6 * 7] = (*xs).into();
//         // let x: [[[f32; 7]; 6]; 2] = (*xs).into();
//         // self.conv_1.forward()
//     }
// }
