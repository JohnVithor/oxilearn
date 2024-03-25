use tch::Tensor;

use crate::{dqn::DoubleDeepAgent, env::PyEnv, OxiLearnErr};

pub type TrainResults = (Vec<f32>, Vec<u128>, Vec<f32>, Vec<f32>, Vec<f32>);

pub struct Trainer {
    env: PyEnv,
    pub early_stop: Option<Box<dyn Fn(f32) -> bool>>,
}

impl Trainer {
    pub fn new(env: PyEnv) -> Result<Self, OxiLearnErr> {
        if env.observation_space()?.is_discrete() {
            // should be continuous
            return Err(OxiLearnErr::EnvNotSupported);
        }
        if !env.action_space()?.is_discrete() {
            // should be discrete
            return Err(OxiLearnErr::EnvNotSupported);
        }
        Ok(Self {
            env,
            early_stop: None,
        })
    }

    pub fn train_by_steps(
        &mut self,
        agent: &mut DoubleDeepAgent,
        n_steps: u128,
        update_freq: u128,
        eval_at: u128,
        eval_for: u128,
        debug: bool,
    ) -> Result<TrainResults, OxiLearnErr> {
        let mut training_reward: Vec<f32> = vec![];
        let mut training_length: Vec<u128> = vec![];
        let mut training_error: Vec<f32> = vec![];
        let mut evaluation_reward: Vec<f32> = vec![];
        let mut evaluation_length: Vec<f32> = vec![];

        let mut n_episodes = 0;
        let mut action_counter: u128 = 0;
        let mut epi_reward: f32 = 0.0;
        let mut curr_obs: Tensor = self.env.reset()?;
        agent.reset();
        for _ in 0..n_steps {
            let curr_action = agent.get_action(&curr_obs);

            let (next_obs, reward, done, truncated) = self.env.step(curr_action)?;

            epi_reward += reward;
            agent.add_transition(&curr_obs, curr_action, reward, done, &next_obs);

            curr_obs = next_obs;

            if let Some(td) = agent.update() {
                training_error.push(td)
            }

            if done || truncated {
                if debug {
                    println!("{}", self.env.render());
                }
                training_reward.push(epi_reward);
                if n_episodes % update_freq == 0 && agent.update_networks().is_err() {
                    println!("copy error")
                }
                if n_episodes % eval_at == 0 {
                    let (rewards, eval_lengths) = self.evaluate(agent, eval_for)?;
                    let reward_avg = (rewards.iter().sum::<f32>()) / (rewards.len() as f32);
                    let eval_lengths_avg = (eval_lengths.iter().map(|x| *x as f32).sum::<f32>())
                        / (eval_lengths.len() as f32);
                    println!(
                        "Episode: {}, Avg Return: {:.3} Current epsilon {:.3}",
                        n_episodes,
                        reward_avg,
                        agent.get_epsilon()
                    );
                    evaluation_reward.push(reward_avg);
                    evaluation_length.push(eval_lengths_avg);
                    if let Some(s) = &self.early_stop {
                        if (s)(reward_avg) {
                            break;
                        };
                    }
                }
                curr_obs = self.env.reset()?;
                agent.action_selection_update(epi_reward);
                n_episodes += 1;
                epi_reward = 0.0;
                action_counter = 0;
            }
            training_length.push(action_counter);
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
        agent: &mut DoubleDeepAgent,
        n_episodes: u128,
    ) -> Result<(Vec<f32>, Vec<u128>), OxiLearnErr> {
        let mut reward_history: Vec<f32> = vec![];
        let mut episode_length: Vec<u128> = vec![];
        for _episode in 0..n_episodes {
            let mut action_counter: u128 = 0;
            let mut epi_reward: f32 = 0.0;
            let obs_repr = self.env.reset()?;
            let mut curr_action = agent.get_best_action(&obs_repr);
            loop {
                action_counter += 1;
                let (obs, reward, done, truncated) = self.env.step(curr_action)?;
                let next_obs_repr = obs;
                let next_action_repr: usize = agent.get_best_action(&next_obs_repr);
                let next_action = next_action_repr;
                curr_action = next_action;
                epi_reward += reward;
                if done || truncated {
                    reward_history.push(epi_reward);
                    break;
                }
                if let Some(s) = &self.early_stop {
                    if (s)(epi_reward) {
                        reward_history.push(epi_reward);
                        break;
                    };
                }
            }
            episode_length.push(action_counter);
        }
        Ok((reward_history, episode_length))
    }
}
