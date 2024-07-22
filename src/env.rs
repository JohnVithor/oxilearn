use tch::Tensor;

use crate::OxiLearnErr;

#[derive(Debug, Clone)]
pub enum SpaceInfo {
    Discrete(usize),
    Continuous(Vec<(f32, f32)>),
}

impl SpaceInfo {
    pub fn is_discrete(&self) -> bool {
        match self {
            SpaceInfo::Discrete(_) => true,
            SpaceInfo::Continuous(_) => false,
        }
    }

    pub fn shape(&self) -> i64 {
        match self {
            SpaceInfo::Discrete(n) => *n as i64,
            SpaceInfo::Continuous(v) => v.len() as i64,
        }
    }
}

pub trait Env {
    fn reset(&mut self, seed: Option<i64>) -> Result<Tensor, OxiLearnErr>;
    fn step(&mut self, action: usize) -> Result<(Tensor, f32, bool, bool), OxiLearnErr>;
    fn observation_space(&self) -> Result<SpaceInfo, OxiLearnErr>;
    fn action_space(&self) -> Result<SpaceInfo, OxiLearnErr>;
    fn reward_threshold(&self) -> Option<f32>;
}
