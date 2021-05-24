use std::ffi::c_void;
use tch::{kind::Element, IndexOp, Kind, Tensor};
use torch_sys::at_tensor_of_data;

pub struct BatchRandSampler {
    inds: Tensor,
    x: Tensor,
    y: Tensor,
    z: Tensor,

    size: i64,
    batch_size: i64,
    index: i64,
    drop_last: bool,
    device: tch::Device,
}

impl BatchRandSampler {
    pub fn new(
        x: Tensor,
        y: Tensor,
        z: Tensor,
        batch_size: i64,
        drop_last: bool,
        device: tch::Device,
    ) -> Self {
        let n = x.size()[0];
        Self {
            inds: Tensor::randperm(n, (tch::Kind::Int64, device)),
            x,
            y,
            z,
            size: n,
            batch_size,
            index: 0,
            drop_last,
            device,
        }
    }
}

impl Iterator for BatchRandSampler {
    type Item = (Tensor, Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        let next_index = (self.index + self.batch_size).min(self.size);
        if self.index >= self.size
            || (self.drop_last && (next_index - self.index) < self.batch_size)
        {
            return None;
        }

        let batch_inds = self.inds.i(self.index..next_index).to(self.device);
        self.index = next_index;

        let item = (
            self.x.index_select(0, &batch_inds).to(self.device),
            self.y.index_select(0, &batch_inds).to(self.device),
            self.z.index_select(0, &batch_inds).to(self.device),
        );
        Some(item)
    }
}

pub fn tensor<T>(data: &[T], dims: &[i64], kind: tch::Kind) -> Tensor {
    let t = unsafe {
        Tensor::from_ptr(at_tensor_of_data(
            data.as_ptr() as *const c_void,
            dims.as_ptr(),
            dims.len(),
            kind.elt_size_in_bytes(),
            match kind {
                Kind::Uint8 => 0,
                Kind::Int8 => 1,
                Kind::Int16 => 2,
                Kind::Int => 3,
                Kind::Int64 => 4,
                Kind::Half => 5,
                Kind::Float => 6,
                Kind::Double => 7,
                Kind::ComplexHalf => 8,
                Kind::ComplexFloat => 9,
                Kind::ComplexDouble => 10,
                Kind::Bool => 11,
                Kind::QInt8 => 12,
                Kind::QUInt8 => 13,
                Kind::QInt32 => 14,
                Kind::BFloat16 => 15,
            },
        ))
    };
    t
}
