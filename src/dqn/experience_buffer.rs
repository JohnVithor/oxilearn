use candle_core::{Device, Result, Tensor};
use rand::seq::IteratorRandom;
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
        curr_state: &Tensor,
        curr_action: u32,
        reward: f32,
        done: bool,
        next_state: &Tensor,
    ) {
        if self.size < self.capacity {
            self.curr_states.push(curr_state.clone());
            self.curr_actions.push(curr_action);
            self.rewards.push(reward);
            self.next_states.push(next_state.clone());
            self.dones.push(done);
            self.size += 1;
        } else {
            self.curr_states[self.next_idx] = curr_state.clone();
            self.curr_actions[self.next_idx] = curr_action;
            self.rewards[self.next_idx] = reward;
            self.next_states[self.next_idx] = next_state.clone();
            self.dones[self.next_idx] = done;
            self.next_idx = (self.next_idx + 1) % self.capacity;
        }
    }

    pub fn sample_batch(
        &mut self,
        size: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        // println!("size: {:?}", size);
        let index: Vec<usize> = (0..self.size).choose_multiple(&mut self.rng, size);
        let index: &[usize] = index.as_slice();
        // println!("index: {:?}", index);
        let curr_states = select_by_index(&self.curr_states, index);

        let curr_actions = select_by_index(&self.curr_actions, index);
        let rewards = select_by_index(&self.rewards, index);
        let next_states = select_by_index(&self.next_states, index);
        let dones = select_by_index(&self.dones, index);

        let curr_states = Tensor::stack(curr_states.as_slice(), 0)?.reshape(&[size, 4])?;
        // println!(" {:?}", curr_states);
        let curr_actions = Tensor::from_vec(curr_actions, (size, 1), &self.device)?;
        // println!(" {:?}", curr_actions);
        let rewards = Tensor::from_vec(rewards, (size, 1), &self.device)?;
        // println!(" {:?}", rewards);
        let next_states = Tensor::stack(next_states.as_slice(), 0)?.reshape(&[size, 4])?;
        // println!(" {:?}", next_states);
        let dones = Tensor::from_vec(
            dones.into_iter().map(|v| v as u8).collect(),
            (size, 1),
            &self.device,
        )?;
        // println!(" {:?}", dones);

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
