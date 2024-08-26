use candle_core::{Device, Result, Tensor};
use rand::distributions::Distribution;
use rand::{rngs::SmallRng, SeedableRng};

pub struct RandomExperienceBuffer {
    pub curr_states: Vec<Tensor>,
    curr_actions: Vec<u32>,
    rewards: Vec<f32>,
    pub next_states: Vec<Tensor>,
    dones: Vec<bool>,
    size: usize,
    next_idx: usize,
    capacity: usize,
    minsize: usize,
    rng: SmallRng,
    device: Device,
}

impl RandomExperienceBuffer {
    pub fn new(capacity: usize, minsize: usize, seed: u64, device: Device) -> Result<Self> {
        Ok(Self {
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
            device,
        })
    }

    pub fn ready(&self) -> bool {
        self.size >= self.minsize
    }

    pub fn add(
        &mut self,
        curr_state: Tensor,
        curr_action: u32,
        reward: f32,
        done: bool,
        next_state: Tensor,
    ) -> Result<()> {
        self.curr_states[self.next_idx] = curr_state;
        self.curr_actions[self.next_idx] = curr_action;
        self.rewards[self.next_idx] = reward;
        self.next_states[self.next_idx] = next_state;
        self.dones[self.next_idx] = done;

        self.next_idx = (self.next_idx + 1) % self.capacity;
        if self.size < self.capacity {
            self.size += 1;
        }
        Ok(())
    }

    pub fn sample_batch(
        &mut self,
        size: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let dist: rand::distributions::Uniform<usize> =
            rand::distributions::Uniform::new(0usize, self.size);
        let index: Vec<usize> = dist.sample_iter(&mut self.rng).take(size).collect();
        let index: &[usize] = index.as_slice();

        let curr_states = select_by_index(&self.curr_states, index);
        let curr_actions = select_by_index(&self.curr_actions, index);
        let rewards = select_by_index(&self.rewards, index);
        let next_states = select_by_index(&self.next_states, index);
        let dones = select_by_index(&self.dones, index);

        let curr_states = Tensor::stack(curr_states.as_slice(), 0)?;
        let curr_actions = Tensor::from_vec(curr_actions, 0, &self.device)?;
        let rewards = Tensor::from_vec(rewards, 0, &self.device)?;
        let next_states = Tensor::stack(next_states.as_slice(), 0)?;
        let dones = Tensor::from_vec(
            dones.into_iter().map(|v| v as u8).collect(),
            0,
            &self.device,
        )?;

        Ok((curr_states, curr_actions, rewards, dones, next_states))
    }
}

fn select_by_index<T: Clone>(data: &[T], index: &[usize]) -> Vec<T> {
    let sel_data: Vec<T> = data
        .iter()
        .enumerate()
        .filter(|(i, _)| index.contains(i))
        .map(|(_, v)| v.clone())
        .collect();
    sel_data
}
