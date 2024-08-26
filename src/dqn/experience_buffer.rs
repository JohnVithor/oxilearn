use candle_core::{Device, Result, Tensor};
use rand::distributions::Distribution;
use rand::{rngs::SmallRng, SeedableRng};

use super::experience_stats::ExperienceStats;

pub struct RandomExperienceBuffer {
    obs_size: usize,
    pub curr_states: Vec<Tensor>,
    curr_actions: Vec<usize>,
    rewards: Vec<f32>,
    pub next_states: Vec<Tensor>,
    dones: Vec<bool>,
    size: usize,
    next_idx: usize,
    capacity: usize,
    minsize: usize,
    rng: SmallRng,
    device: Device,
    pub stats: ExperienceStats,
    normalize_obs: bool,
}

impl RandomExperienceBuffer {
    pub fn new(
        capacity: usize,
        obs_size: usize,
        minsize: usize,
        seed: u64,
        normalize_obs: bool,
        device: Device,
    ) -> Result<Self> {
        Ok(Self {
            obs_size,
            curr_states: Vec::with_capacity(capacity),
            curr_actions: Vec::with_capacity(capacity),
            rewards: Vec::with_capacity(capacity),
            next_states: Vec::with_capacity(capacity),
            dones: Vec::with_capacity(capacity),
            capacity,
            next_idx: 0,
            size: 0,
            minsize,
            rng: SmallRng::seed_from_u64(seed),
            stats: ExperienceStats::new(obs_size, &device)?,
            device,
            normalize_obs,
        })
    }

    pub fn ready(&self) -> bool {
        self.size >= self.minsize
    }

    pub fn add(
        &mut self,
        curr_state: Tensor,
        curr_action: usize,
        reward: f32,
        done: bool,
        next_state: Tensor,
    ) -> Result<()> {
        self.curr_states[self.next_idx] = curr_state;
        self.curr_actions[self.next_idx] = curr_action;
        self.rewards[self.next_idx] = reward;
        self.next_states[self.next_idx] = next_state;
        self.dones[self.next_idx] = done;

        // let index: i64 = self.obs_size * self.next_idx;
        // let index = Vec::from_iter(index..(index + self.obs_size));
        // let index = &Tensor::from_slice(&index).to_device(self.device);

        // let curr_state = &curr_state.to_device(self.device).to_kind(Kind::Double);
        // let curr_action = &Tensor::from(curr_action as i64).to_device(self.device);
        // let reward = &Tensor::from(reward)
        //     .to_device(self.device)
        //     .to_kind(Kind::Double);
        // let done = &Tensor::from(done as i8).to_device(self.device);
        // let next_state = &next_state.to_device(self.device).to_kind(Kind::Double);

        // self.curr_states = self.curr_states.put(index, curr_state, false);
        // self.next_states = self.next_states.put(index, next_state, false);

        // let index = &Tensor::from(self.next_idx).to_device(self.device);

        // self.curr_actions = self.curr_actions.put(index, curr_action, false);
        // self.rewards = self.rewards.put(index, reward, false);
        // self.dones = self.dones.put(index, done, false);

        // self.next_idx = (self.next_idx + 1) % self.capacity;
        // self.size = self.capacity.min(self.size + 1);

        // if self.normalize_obs {
        //     self.stats.push(curr_state);
        // }
        Ok(())
    }

    pub fn normalize(&self, values: Tensor) -> Tensor {
        // let (var, mean) = self.curr_states.var_mean_dim(0, false, false);
        // ((values - mean) / var.sqrt()).to_device(self.device)
        if self.normalize_obs {
            (values - self.stats.mean()) / (self.stats.var()).sqrt()
        } else {
            values
        }
    }

    pub fn sample_batch(&mut self, size: usize) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        let dist: rand::distributions::Uniform<i64> =
            rand::distributions::Uniform::new(0i64, self.size);
        let index: Vec<i64> = dist.sample_iter(&mut self.rng).take(size).collect();
        // println!("{index:?}");
        (
            self.normalize(self.curr_states.i(index.clone()))
                .to_kind(Kind::Double),
            self.curr_actions.i(index.clone()).reshape([-1, 1]),
            self.rewards.i(index.clone()).reshape([-1, 1]),
            self.dones.i(index.clone()).reshape([-1, 1]),
            self.normalize(self.next_states.i(index))
                .to_kind(Kind::Double),
        )
    }
}
