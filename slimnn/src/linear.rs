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

/*

use std::arch::x86_64::*;

#[derive(Debug)]
pub struct SIMDLinear<const Ix8: usize, const Ox8: usize, const O: usize> {
    pub weight: [[__m256; O]; Ix8],
    pub bias: [__m256; O],
}

impl<const Ix8: usize, const Ox8: usize, const O: usize> From<&Linear<Ix8, Ox8>> for SIMDLinear<Ix8, Ox8, O> {
    fn from(linear: &Linear<Ix8, Ox8>) -> Self {
        let mut weight = [[unsafe { _mm256_setzero_ps() }; O]; Ix8];
        let mut bias = [unsafe { _mm256_setzero_ps() }; O];
        for i_input in 0..Ix8 {
            for i_output in 0..O {
                weight[i_input][i_output] = unsafe { _mm256_loadu_ps(&linear.weight[i_input][8 * i_output]) };
            }
        }
        for i_output in 0..O {
            bias[i_output] = unsafe { _mm256_loadu_ps(&linear.bias[8 * i_output]) };
        }

        Self { weight, bias }
    }
}

impl<const Ix8: usize, const Ox8: usize, const O: usize> SIMDLinear<Ix8, Ox8, O> {
    pub fn set1(&self, x: &[f32; Ix8]) -> [__m256; Ix8] {
        let mut c = [unsafe { _mm256_setzero_ps() }; Ix8];
        for i in 0..Ix8 {
            c[i] = unsafe { _mm256_set1_ps(x[i]) };
        }
        c
    }

    pub fn decompress(&self, x: &[__m256; O]) -> [f32; Ox8] {
        let mut o = [0.0; Ox8];
        for i in 0..O {
            unsafe {
                _mm256_store_ps(&mut o[8 * i], x[i]);
            }
        }
        o
    }

    pub fn forward(&self, x: &[__m256; Ix8]) -> [__m256; O] {
        let mut output = self.bias;
        for i_input in 0..Ix8 {
            let w = &self.weight[i_input];
            let v = x[i_input];
            for i_output in 0..O {
                output[i_output] = unsafe {
                    _mm256_fmadd_ps(v, w[i_output], output[i_output])
                };
            }
        }
        output
    }

    pub fn forward_relu(&self, x: &[__m256; Ix8]) -> [__m256; O] {
        let mut o = self.forward(x);
        let zeros = unsafe { _mm256_setzero_ps() };
        for i_output in 0..O {
            o[i_output] = unsafe {
                _mm256_max_ps(o[i_output], zeros)
            };
        }
        o
    }
}
*/

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
