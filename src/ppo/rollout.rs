use tch::{Device, Kind, Tensor};

pub struct Rollout {
    pub obs: Tensor,
    pub actions: Tensor,
    pub logprobs: Tensor,
    pub rewards: Tensor,
    pub dones: Tensor,
    pub values: Tensor,
    pub episode_returns: Vec<f64>,
    pub step: usize,
    pub device: Device,
    pub obs_size: i64,
    pub capacity: i64,
}

impl Rollout {
    pub fn new(capacity: usize, obs_size: i64, device: Device) -> Self {
        let capacity = capacity as i64;
        Self {
            obs: Tensor::empty([capacity, obs_size], (tch::Kind::Float, device)),
            actions: Tensor::empty(capacity, (tch::Kind::Int64, device)),
            logprobs: Tensor::empty(capacity, (tch::Kind::Float, device)),
            rewards: Tensor::empty(capacity, (tch::Kind::Float, device)),
            dones: Tensor::empty(capacity, (tch::Kind::Int8, device)),
            values: Tensor::empty(capacity, (tch::Kind::Float, device)),
            episode_returns: Vec::new(),
            step: 0,
            device,
            obs_size,
            capacity,
        }
    }

    pub fn add(
        &mut self,
        obs: &Tensor,
        action: &Tensor,
        logprob: &Tensor,
        reward: f64,
        done: bool,
        value: &Tensor,
    ) {
        let index: i64 = self.obs_size * self.step as i64;
        let index = Vec::from_iter(index..(index + self.obs_size));
        let index = &Tensor::from_slice(&index).to_device(self.device);

        self.obs = self.obs.put(index, obs, false);

        let index = &Tensor::from_slice(&[self.step as i64]).to_device(self.device);

        let reward = &Tensor::from(reward)
            .to_device(self.device)
            .to_kind(Kind::Float);
        let done = &Tensor::from(done)
            .to_device(self.device)
            .to_kind(Kind::Int8);
        self.actions = self.actions.put(index, action, false);
        self.logprobs = self.logprobs.put(index, logprob, false);
        self.rewards = self.rewards.put(index, reward, false);
        self.dones = self.dones.put(index, done, false);
        self.values = self.values.put(index, value, false);

        self.step += 1;
    }

    pub fn reset(&mut self) {
        self.step = 0;
        self.episode_returns.clear();
    }

    pub fn add_episode_return(&mut self, episode_return: f64) {
        self.episode_returns.push(episode_return);
    }
}
