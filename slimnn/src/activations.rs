pub trait Activation {
    fn apply(&self, x: f32) -> f32;

    fn apply_1d<const N: usize>(&self, x: &[f32; N]) -> [f32; N] {
        let mut y = [0.0; N];
        for i in 0..N {
            y[i] = self.apply(x[i]);
        }
        y
    }

    fn apply_2d<const W: usize, const H: usize>(&self, x: &[[f32; W]; H]) -> [[f32; W]; H] {
        let mut y = [[0.0; W]; H];
        for i in 0..H {
            y[i] = self.apply_1d(&x[i]);
        }
        y
    }

    fn apply_3d<const W: usize, const H: usize, const I: usize>(
        &self,
        x: &[[[f32; I]; W]; H],
    ) -> [[[f32; I]; W]; H] {
        let mut y = [[[0.0; I]; W]; H];
        for i in 0..H {
            y[i] = self.apply_2d(&x[i]);
        }
        y
    }
}

pub struct ReLU;
impl Activation for ReLU {
    fn apply(&self, x: f32) -> f32 {
        x.max(0.0)
    }
}

pub struct Tanh;
impl Activation for Tanh {
    fn apply(&self, x: f32) -> f32 {
        x.tanh()
    }
}

pub struct Softmax;
impl Activation for Softmax {
    fn apply(&self, _x: f32) -> f32 {
        panic!("Can't call softmax on 1d values")
    }

    fn apply_1d<const N: usize>(&self, x: &[f32; N]) -> [f32; N] {
        let mut y = [0.0; N];
        let mut total = 0.0;
        for i in 0..N {
            y[i] = x[i].exp();
            total += y[i];
        }
        for i in 0..N {
            y[i] /= total;
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_1d() {
        let x = [-2., -1., -0.5, 0., 0.5, 1., 2.];
        let y = ReLU.apply_1d(&x);
        assert_eq!(y, [0., 0., 0., 0.0, 0.5, 1., 2.])
    }
}
