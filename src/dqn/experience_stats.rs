use candle_core::{Device, Result, Tensor};

pub struct ExperienceStats {
    means: Tensor,
    msqs: Tensor,
    count: usize,
}

impl ExperienceStats {
    pub fn new(shape: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            means: Tensor::zeros(shape, candle_core::DType::F32, &device)?,
            msqs: Tensor::ones(shape, candle_core::DType::F32, &device)?,
            count: 0,
        })
    }

    pub fn push(&mut self, value: &Tensor) -> Result<()> {
        self.count += 1;
        let delta = (value - &self.means)?;
        self.means = (&self.means + (&delta / self.count as f64)?)?;
        let delta2 = (value - &self.means)?;
        self.msqs = (&self.msqs + (delta * delta2)?)?;
        Ok(())
    }

    pub fn mean(&self) -> &Tensor {
        &self.means
    }

    pub fn var(&self) -> Result<Tensor> {
        &self.msqs / self.count as f64
    }
}
