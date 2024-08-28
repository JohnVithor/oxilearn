use candle_core::Result;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, SGD};

pub enum OptimizerConfig {
    SgdConfig(f64),
    AdamWConfig(ParamsAdamW),
}

impl OptimizerConfig {
    pub fn create(self, vars: Vec<candle_core::Var>) -> Result<OptimizerEnum> {
        match self {
            OptimizerConfig::SgdConfig(lr) => {
                let sgd = SGD::new(vars, lr)?;
                Ok(OptimizerEnum::Sgd(sgd))
            }
            OptimizerConfig::AdamWConfig(params) => {
                let adamw = AdamW::new(vars, params)?;
                Ok(OptimizerEnum::AdamW(adamw))
            }
        }
    }
}

#[derive(Debug)]
pub enum OptimizerEnum {
    Sgd(SGD),
    AdamW(AdamW),
}

impl Optimizer for OptimizerEnum {
    type Config = OptimizerConfig;

    fn new(vars: Vec<candle_core::Var>, config: Self::Config) -> candle_core::Result<Self> {
        match config {
            OptimizerConfig::SgdConfig(lr) => {
                let sgd = SGD::new(vars, lr)?;
                Ok(Self::Sgd(sgd))
            }
            OptimizerConfig::AdamWConfig(params) => {
                let adamw = AdamW::new(vars, params)?;
                Ok(Self::AdamW(adamw))
            }
        }
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> candle_core::Result<()> {
        match self {
            Self::Sgd(sgd) => sgd.step(grads),
            Self::AdamW(adamw) => adamw.step(grads),
        }
    }

    fn learning_rate(&self) -> f64 {
        match self {
            Self::Sgd(sgd) => sgd.learning_rate(),
            Self::AdamW(adamw) => adamw.learning_rate(),
        }
    }

    fn set_learning_rate(&mut self, lr: f64) {
        match self {
            Self::Sgd(sgd) => sgd.set_learning_rate(lr),
            Self::AdamW(adamw) => adamw.set_learning_rate(lr),
        }
    }
}
