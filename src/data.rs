use tch::{IndexOp, Tensor};

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
