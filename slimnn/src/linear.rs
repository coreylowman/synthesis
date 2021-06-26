#[derive(Debug)]
pub struct Linear<const I: usize, const O: usize> {
    pub weight: [[f32; I]; O],
    pub bias: [f32; O],
}

impl<const I: usize, const O: usize> Default for Linear<I, O> {
    fn default() -> Self {
        Self {
            weight: [[0.0; I]; O],
            bias: [0.0; O],
        }
    }
}

impl<const I: usize, const O: usize> Linear<I, O> {
    pub fn forward(&self, x: &[f32; I]) -> [f32; O] {
        let mut output = self.bias;
        for i_input in 0..I {
            for i_output in 0..O {
                output[i_output] += x[i_input] * self.weight[i_output][i_input];
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
        let mut q: Linear<3, 2> = Default::default();
        q.weight = [[1., 3., 5.], [2., 4., 6.]];
        q.bias = [-1., 1.];
        assert_eq!(q.forward(&[3., 2., 1.]), [13., 21.]);
        assert_eq!(q.forward(&[1., 3., 2.]), [19., 27.]);
    }
}
