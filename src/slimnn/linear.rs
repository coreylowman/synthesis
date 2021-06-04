pub struct Linear<const I: usize, const O: usize> {
    pub weight: [[f32; I]; O],
    pub bias: [f32; O],
}

impl<const I: usize, const O: usize> Linear<I, O> {
    pub fn new() -> Self {
        Self {
            weight: [[0.0; I]; O],
            bias: [0.0; O],
        }
    }

    pub fn forward(&self, x: &[f32; I]) -> [f32; O] {
        let mut output = self.bias.clone();
        for i_output in 0..O {
            let w = &self.weight[i_output];
            for i_input in 0..I {
                output[i_output] += x[i_input] * w[i_input];
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear() {
        let mut q = Linear::<3, 2>::new();
        q.weight = [[1., 3., 5.], [2., 4., 6.]];
        q.bias = [-1., 1.];
        assert_eq!(q.forward(&[3., 2., 1.]), [13., 21.]);
    }
}
