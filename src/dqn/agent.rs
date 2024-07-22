use std::fs;
use tch::{
    nn::{Module, Optimizer, OptimizerConfig, VarStore},
    Device, Kind, TchError, Tensor,
};

use crate::{env::Env, optimizer_enum::OptimizerEnum, OxiLearnErr};

use super::{
    epsilon_greedy::EpsilonGreedy, experience_buffer::RandomExperienceBuffer,
    policy::PolicyGenerator,
};

pub type TrainResults = (Vec<f32>, Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>);

pub struct ParametersDQN {
    pub learning_rate: f64,
    pub gradient_steps: u32,
    pub train_freq: u32,
    pub batch_size: usize,
    pub update_freq: u32,
    pub eval_freq: u32,
    pub eval_for: u32,
    pub discount_factor: f32,
    pub max_grad_norm: f64,
}

pub struct DQNAgent {
    pub action_selection: EpsilonGreedy,
    pub policy: Box<dyn Module>,
    pub target_policy: Box<dyn Module>,
    pub policy_vs: VarStore,
    pub target_policy_vs: VarStore,
    pub optimizer: Optimizer,
    pub loss_fn: fn(&Tensor, &Tensor) -> Tensor,
    pub memory: RandomExperienceBuffer,
    pub parameters: ParametersDQN,
    pub device: Device,
}

impl DQNAgent {
    pub fn new(
        action_selector: EpsilonGreedy,
        mem_replay: RandomExperienceBuffer,
        generate_policy: Box<PolicyGenerator>,
        opt: OptimizerEnum,
        loss_fn: fn(&Tensor, &Tensor) -> Tensor,
        parameters: ParametersDQN,
        device: Device,
    ) -> Self {
        let (policy_net, mem_policy) = generate_policy(device);
        let (target_net, mut mem_target) = generate_policy(device);
        mem_target.copy(&mem_policy).unwrap();
        Self {
            optimizer: opt.build(&mem_policy, parameters.learning_rate).unwrap(),
            loss_fn,
            action_selection: action_selector,
            memory: mem_replay,
            policy: policy_net,
            policy_vs: mem_policy,
            target_policy: target_net,
            target_policy_vs: mem_target,
            parameters,
            device,
        }
    }

    pub fn get_action(&mut self, state: &Tensor) -> usize {
        let values = tch::no_grad(|| {
            self.policy
                .forward(&state.to_kind(Kind::Double).to_device(self.device))
        });
        self.action_selection.get_action(&values) as usize
    }

    pub fn get_best_action(&self, state: &Tensor) -> usize {
        let values = tch::no_grad(|| {
            self.policy
                .forward(&state.to_kind(Kind::Double).to_device(self.device))
        });
        let a: i32 = values.argmax(0, true).try_into().unwrap();
        a as usize
    }

    pub fn add_transition(
        &mut self,
        curr_state: &Tensor,
        curr_action: usize,
        reward: f32,
        done: bool,
        next_state: &Tensor,
    ) {
        self.memory.add(
            &curr_state.to_kind(Kind::Double),
            curr_action,
            reward,
            done,
            &next_state.to_kind(Kind::Double),
        );
    }

    pub fn update_networks(&mut self) -> Result<(), TchError> {
        self.target_policy_vs.copy(&self.policy_vs)
    }

    pub fn get_batch(&mut self, size: usize) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        self.memory.sample_batch(size)
    }

    pub fn batch_qvalues(&self, b_states: &Tensor, b_actions: &Tensor) -> Tensor {
        self.policy
            .forward(&b_states.to_kind(Kind::Double))
            .gather(1, &b_actions.to_kind(Kind::Int64), false)
            .to_kind(Kind::Double)
    }

    pub fn batch_expected_values(
        &self,
        b_state_: &Tensor,
        b_reward: &Tensor,
        b_done: &Tensor,
    ) -> Tensor {
        let best_target_qvalues = tch::no_grad(|| {
            self.target_policy
                .forward(&b_state_.to_kind(Kind::Double))
                .max_dim(1, true)
                .0
        });
        (b_reward.to_kind(Kind::Double)
            + self.parameters.discount_factor
                * (&Tensor::from(1.0).to_device(self.device) - b_done.to_kind(Kind::Double))
                * (&best_target_qvalues))
            .to_kind(Kind::Double)
    }

    pub fn optimize(&mut self, loss: Tensor) {
        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.clip_grad_norm(self.parameters.max_grad_norm);
        self.optimizer.step();
    }

    pub fn update(&mut self, gradient_steps: u32, batch_size: usize) -> Option<f32> {
        let mut values = vec![];
        if self.memory.ready() {
            for _ in 0..gradient_steps {
                let (b_state, b_action, b_reward, b_done, b_state_) = self.get_batch(batch_size);
                // print_python_like(&b_state.i(0));
                let policy_qvalues = self.batch_qvalues(&b_state, &b_action);
                let expected_values = self.batch_expected_values(&b_state_, &b_reward, &b_done);
                let loss = (self.loss_fn)(&policy_qvalues, &expected_values).to_kind(Kind::Double);
                self.optimize(loss);
                values.push(expected_values.mean(Kind::Double).try_into().unwrap())
            }
            Some((values.iter().sum::<f32>()) / (values.len() as f32))
        } else {
            None
        }
    }

    pub fn action_selection_update(&mut self, current_training_progress: f32, epi_reward: f32) {
        self.action_selection
            .update(current_training_progress, epi_reward);
    }

    pub fn get_epsilon(&self) -> f32 {
        self.action_selection.get_epsilon()
    }

    pub fn reset(&mut self) {
        self.action_selection.reset();
        // TODO: reset policies
    }

    pub fn save_net(&self, path: &str) -> Result<(), TchError> {
        fs::create_dir_all(path)?;
        self.policy_vs
            .save(format!("{path}/policy_weights.safetensors"))?;
        self.target_policy_vs
            .save(format!("{path}/target_policy_weights.safetensors"))?;
        Ok(())
    }

    pub fn load_net(&mut self, path: &str) -> Result<(), TchError> {
        self.policy_vs
            .load(format!("{path}/policy_weights.safetensors"))?;
        self.target_policy_vs
            .load(format!("{path}/target_policy_weights.safetensors"))?;
        Ok(())
    }
    pub fn train_by_steps(
        &mut self,
        env: &mut impl Env,
        eval_env: &mut impl Env,
        n_steps: u32,
        verbose: usize,
    ) -> Result<TrainResults, OxiLearnErr> {
        let mut curr_obs: Tensor = env.reset(None)?;
        let mut training_reward: Vec<f32> = vec![];
        let mut training_length: Vec<u32> = vec![];
        let mut training_error: Vec<f32> = vec![];
        let mut evaluation_reward: Vec<f32> = vec![];
        let mut evaluation_length: Vec<f32> = vec![];

        let mut n_episodes = 1;
        let mut action_counter: u32 = 0;
        let mut epi_reward: f32 = 0.0;
        self.reset();

        for step in 1..=n_steps {
            action_counter += 1;
            let curr_action = self.get_action(&curr_obs);
            // println!("{curr_action}");
            let (next_obs, reward, done, truncated) = env.step(curr_action)?;

            epi_reward += reward;
            self.add_transition(&curr_obs, curr_action, reward, done, &next_obs);

            curr_obs = next_obs;

            if step % self.parameters.train_freq == 0 {
                if let Some(td) =
                    self.update(self.parameters.gradient_steps, self.parameters.batch_size)
                {
                    training_error.push(td)
                }
            }

            if done || truncated {
                training_reward.push(epi_reward);
                training_length.push(action_counter);
                if n_episodes % self.parameters.update_freq == 0 && self.update_networks().is_err()
                {
                    println!("copy error")
                }
                curr_obs = env.reset(None)?;

                self.action_selection_update(step as f32 / n_steps as f32, epi_reward);
                n_episodes += 1;
                epi_reward = 0.0;
                action_counter = 0;
            }

            if step % self.parameters.eval_freq == 0 {
                let (rewards, eval_lengths) = self.evaluate(eval_env, self.parameters.eval_for)?;
                let reward_avg = (rewards.iter().sum::<f32>()) / (rewards.len() as f32);
                let eval_lengths_avg = (eval_lengths.iter().map(|x| *x as f32).sum::<f32>())
                    / (eval_lengths.len() as f32);
                if verbose > 0 {
                    println!(
                        "current step: {step} - mean eval reward: {reward_avg:.1} - exploration epsilon: {:.2}",
                        self.get_epsilon()
                    );
                }
                evaluation_reward.push(reward_avg);
                evaluation_length.push(eval_lengths_avg);
                if step == n_steps
                    || (eval_env.reward_threshold().is_some()
                        && reward_avg > eval_env.reward_threshold().unwrap())
                {
                    training_reward.push(epi_reward);
                    training_length.push(action_counter);
                    break;
                }
            }
        }

        Ok((
            training_reward,
            training_length,
            training_error,
            evaluation_reward,
            evaluation_length,
        ))
    }

    pub fn evaluate(
        &mut self,
        eval_env: &mut impl Env,
        n_episodes: u32,
    ) -> Result<(Vec<f32>, Vec<u32>), OxiLearnErr> {
        let mut reward_history: Vec<f32> = vec![];
        let mut episode_length: Vec<u32> = vec![];
        for _episode in 0..n_episodes {
            let mut epi_reward: f32 = 0.0;
            let obs_repr = eval_env.reset(None)?;
            let mut curr_action = self.get_best_action(&obs_repr);
            let mut action_counter: u32 = 0;
            loop {
                let (obs, reward, done, truncated) = eval_env.step(curr_action)?;
                let next_obs_repr = obs;
                let next_action_repr: usize = self.get_best_action(&next_obs_repr);
                let next_action = next_action_repr;
                curr_action = next_action;
                epi_reward += reward;
                if done || truncated {
                    reward_history.push(epi_reward);
                    episode_length.push(action_counter);
                    break;
                }
                action_counter += 1;
            }
        }
        Ok((reward_history, episode_length))
    }
}
