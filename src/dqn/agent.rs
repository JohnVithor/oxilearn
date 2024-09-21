use std::fs;

use candle_core::{Device, Tensor};
use candle_nn::{Module, Optimizer, VarMap};

use super::{
    epsilon_greedy::EpsilonGreedy, experience_buffer::RandomExperienceBuffer,
    policy::PolicyGenerator,
};
use crate::optimizer_enum::{OptimizerConfig, OptimizerEnum};
use candle_core::Result;

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
    pub policy_vs: VarMap,
    pub target_policy_vs: VarMap,
    pub optimizer: OptimizerEnum,
    pub loss_fn: fn(&Tensor, &Tensor) -> Result<Tensor>,
    pub memory: RandomExperienceBuffer,
    pub parameters: ParametersDQN,
    pub device: Device,
}

impl DQNAgent {
    pub fn new(
        action_selector: EpsilonGreedy,
        mem_replay: RandomExperienceBuffer,
        generate_policy: Box<PolicyGenerator>,
        optimizer: OptimizerConfig,
        loss_fn: fn(&Tensor, &Tensor) -> Result<Tensor>,
        parameters: ParametersDQN,
        device: Device,
    ) -> Result<Self> {
        let (policy_net, mem_policy) = generate_policy(&device)?;
        let (target_net, mut mem_target) = generate_policy(&device)?;

        mem_target.clone_from(&mem_policy);
        Ok(Self {
            optimizer: optimizer.create(mem_policy.all_vars())?,
            loss_fn,
            action_selection: action_selector,
            memory: mem_replay,
            policy: policy_net,
            policy_vs: mem_policy,
            target_policy: target_net,
            target_policy_vs: mem_target,
            parameters,
            device,
        })
    }

    pub fn get_action(&mut self, state: &Tensor) -> Result<usize> {
        let values = self.policy.forward(state)?.detach();
        self.action_selection.get_action(&values)
    }

    pub fn get_best_action(&self, state: &Tensor) -> Result<usize> {
        let values = self.policy.forward(state)?.detach();
        let action = values.argmax(1)?;
        let action = action.reshape(())?;
        let action: u32 = action.to_scalar()?;
        Ok(action as usize)
    }

    pub fn add_transition(
        &mut self,
        curr_state: &Tensor,
        curr_action: usize,
        reward: f32,
        done: bool,
        next_state: &Tensor,
    ) {
        self.memory
            .add(curr_state, curr_action as u32, reward, done, next_state);
    }

    pub fn update_networks(&mut self) {
        self.target_policy_vs.clone_from(&self.policy_vs);
    }

    pub fn get_batch(&mut self, size: usize) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        self.memory.sample_batch(size)
    }

    pub fn batch_qvalues(&self, b_states: &Tensor, b_actions: &Tensor) -> Result<Tensor> {
        // println!("forward qvalues");
        let v = self.policy.forward(b_states)?;
        // println!("gather");
        v.gather(b_actions, 2)
    }

    pub fn batch_expected_values(
        &self,
        b_state_: &Tensor,
        b_reward: &Tensor,
        b_done: &Tensor,
    ) -> Result<Tensor> {
        // println!("forward expected");
        let target_qvalues = self.target_policy.forward(b_state_)?;
        // println!("max");
        let best_target_qvalues = target_qvalues.max_keepdim(2)?;
        // println!("flag");
        let flag = (-1.0 * (b_done - 1.0)?)?;
        // println!("dtype");
        let flag = flag.to_dtype(candle_core::DType::F32)?;
        // println!("bradcast");
        let target_values = &(flag.broadcast_mul(&best_target_qvalues))?;
        // println!("add");
        b_reward.broadcast_add(&(self.parameters.discount_factor as f64 * target_values)?)
    }

    pub fn optimize(&mut self, loss: Tensor) -> Result<()> {
        // self.optimizer.zero_grad();
        // loss.backward();
        // self.optimizer.clip_grad_norm(self.parameters.max_grad_norm);
        // self.optimizer.step();
        self.optimizer.backward_step(&loss)
    }

    pub fn update(&mut self, gradient_steps: u32, batch_size: usize) -> Result<Option<f32>> {
        let mut values = vec![];
        if self.memory.ready() {
            // println!("Training");
            for _ in 0..gradient_steps {
                let (b_state, b_action, b_reward, b_done, b_state_) = self.get_batch(batch_size)?;
                let policy_qvalues = self.batch_qvalues(&b_state, &b_action)?;
                let expected_values = self.batch_expected_values(&b_state_, &b_reward, &b_done)?;
                let loss = (self.loss_fn)(&policy_qvalues, &expected_values)?;
                self.optimize(loss)?;
                // println!("optimized");
                let mean_ev = expected_values.mean(0)?;
                // println!("reshape");
                let val = mean_ev.reshape(&[])?;
                // println!("scalar");
                let val = val.to_scalar()?;
                // println!("push");
                values.push(val)
            }
            // println!("Training done");
            Ok(Some((values.iter().sum::<f32>()) / (values.len() as f32)))
        } else {
            Ok(None)
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

    pub fn save_net(&self, path: &str) -> Result<()> {
        fs::create_dir_all(path)?;
        self.policy_vs
            .save(format!("{path}/policy_weights.safetensors"))?;
        self.target_policy_vs
            .save(format!("{path}/target_policy_weights.safetensors"))?;
        Ok(())
    }

    pub fn load_net(&mut self, path: &str) -> Result<()> {
        self.policy_vs
            .load(format!("{path}/policy_weights.safetensors"))?;
        self.target_policy_vs
            .load(format!("{path}/target_policy_weights.safetensors"))?;
        Ok(())
    }
}
