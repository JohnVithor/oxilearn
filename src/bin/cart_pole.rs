use std::env;

use candle_core::Result;
use candle_core::{Device, Tensor};
use candle_nn::loss::mse;
use candle_nn::ParamsAdamW;
use mlflow::timestamp;
use oxilearn::dqn::agent::{DQNAgent, ParametersDQN, TrainResults};
use oxilearn::dqn::epsilon_greedy::{EpsilonGreedy, EpsilonUpdateStrategy};
use oxilearn::dqn::experience_buffer::RandomExperienceBuffer;
use oxilearn::dqn::policy::generate_policy;
use terrarium::classic_control::cart_pole::CartPole;
use terrarium::environment::Environment;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let seed = args[1].parse::<u64>().unwrap();
    let verbose = args[2].parse::<usize>().unwrap();

    let device = Device::cuda_if_available(0)?;

    let mut train_env = CartPole::new(500, seed);
    let mut eval_env = CartPole::new(500, seed);

    let update_strategy = EpsilonUpdateStrategy::EpsilonLinearTrainingDecreasing {
        start: 1.0,
        end: 0.05,
        end_fraction: 0.2,
    };
    let action_selector = EpsilonGreedy::new(1.0, seed + 2, update_strategy);

    let mem_replay = RandomExperienceBuffer::new(10_000, 1000, seed + 3, device.clone())?;
    let policy = generate_policy(
        vec![(16, |x: &Tensor| x.relu())],
        // vec![(256, |x: &Tensor| x.relu()), (256, |x: &Tensor| x.relu())],
        |xs: &Tensor| Ok(xs.clone()),
        4,
        2,
    );

    let mut model = DQNAgent::new(
        action_selector,
        mem_replay,
        policy,
        oxilearn::optimizer_enum::OptimizerConfig::AdamWConfig(ParamsAdamW {
            lr: 0.0001,
            ..Default::default()
        }),
        mse,
        ParametersDQN {
            learning_rate: 0.0001,
            gradient_steps: 1,
            train_freq: 1,
            batch_size: 128,
            update_freq: 10,
            eval_freq: 1000,
            eval_for: 10,
            discount_factor: 0.99,
            max_grad_norm: 1.0,
        },
        device.clone(),
    )?;

    let results = train_by_steps(
        &mut model,
        &mut train_env,
        &mut eval_env,
        5_000,
        verbose,
        Some(475.0),
        &device,
    )?;

    let training_steps = results.1.iter().sum::<u32>();

    let evaluation_results = evaluate(&mut model, &mut eval_env, 10)?;

    let rewards = evaluation_results.0;
    let reward_avg = (rewards.iter().sum::<f32>()) / (rewards.len() as f32);
    let variance = rewards
        .iter()
        .map(|value| {
            let diff = reward_avg - *value;
            diff * diff
        })
        .sum::<f32>()
        / rewards.len() as f32;
    let std = variance.sqrt();

    println!(
        "rust,{seed},{training_steps},{reward_avg},{std}",
        seed = seed,
        training_steps = training_steps,
        reward_avg = reward_avg,
        std = std
    );
    // ## HERE

    // model.save_net("./safetensors/cart_pole").expect("ok");

    // let mut trainer = Trainer::new(train_env, eval_env).unwrap();
    // trainer.early_stop = Some(Box::new(move |reward| reward >= 475.0));

    // let training_results: Result<TrainResults, OxiLearnErr> =
    //     trainer.train_by_steps(&mut model, 50_000, 175, 200, 128, 10, 1000, 10, verbose);

    // let training_steps = training_results.unwrap().1.iter().sum::<u32>();

    // let evaluation_results = trainer.evaluate(&mut model, 1);
    // let rewards = evaluation_results.unwrap().0;
    // let reward_avg = (rewards.iter().sum::<f32>()) / (rewards.len() as f32);
    // let variance = rewards
    //     .iter()
    //     .map(|value| {
    //         let diff = reward_avg - *value;
    //         diff * diff
    //     })
    //     .sum::<f32>()
    //     / rewards.len() as f32;
    // let std = variance.sqrt();
    // // model
    // //     .save_net("./safetensors/cart_pole_after_training")
    // //     .expect("ok");

    // println!("rust,{seed},{training_steps},{reward_avg},{std}")
    Ok(())
}

pub fn train_by_steps(
    agent: &mut DQNAgent,
    env: &mut CartPole,
    eval_env: &mut CartPole,
    n_steps: u32,
    verbose: usize,
    threshold: Option<f32>,
    device: &Device,
) -> Result<TrainResults> {
    let client: mlflow::Client = mlflow::Client::for_server("http://127.0.0.1:8080/api");
    let experiment = client
        .create_experiment("3")
        .unwrap_or_else(|| client.get_experiment("3").unwrap());

    let mut curr_obs: Tensor = Tensor::from_slice(&env.reset(None), (1, 4), device)?;
    let mut training_reward: Vec<f32> = vec![];
    let mut training_length: Vec<u32> = vec![];
    let mut training_error: Vec<f32> = vec![];
    let mut evaluation_reward: Vec<f32> = vec![];
    let mut evaluation_length: Vec<f32> = vec![];

    let mut n_episodes = 1;
    let mut action_counter: u32 = 0;
    let mut epi_reward: f32 = 0.0;
    agent.reset();

    let run = experiment.create_run();

    // for (i, v) in agent.policy_vs.all_vars().iter().enumerate() {
    //     for (j, w) in v.flatten_all()?.to_vec1()?.iter().enumerate() {
    //         let w: f32 = *w;
    //         run.log_metric(&format!("policy_vs_{}_{}", i, j), w as f64, timestamp(), 0);
    //     }
    // }

    for step in 1..=n_steps {
        action_counter += 1;
        run.log_metric(
            "epsilon",
            agent.get_epsilon() as f64,
            timestamp(),
            step as u64,
        );
        let curr_action = agent.get_action(&curr_obs)?;
        run.log_metric("action", curr_action as f64, timestamp(), step as u64);
        let (next_obs, reward, done, truncated) = env.step(curr_action).unwrap();
        let next_obs = Tensor::from_slice(&next_obs, (1, 4), device)?;
        epi_reward += reward;
        agent.add_transition(&curr_obs, curr_action, reward, done, &next_obs);

        run.log_metric("reward", reward as f64, timestamp(), step as u64);
        curr_obs = next_obs;

        if step % agent.parameters.train_freq == 0 {
            if let Some(td) = agent.update(
                &run,
                step as u64,
                agent.parameters.gradient_steps,
                agent.parameters.batch_size,
            )? {
                training_error.push(td);
            }
        }

        if done || truncated {
            // println!("Episode: {}", n_episodes);
            run.log_metric("epi_reward", epi_reward as f64, timestamp(), step as u64);
            run.log_metric(
                "action_counter",
                action_counter as f64,
                timestamp(),
                step as u64,
            );

            training_reward.push(epi_reward);
            training_length.push(action_counter);
            if n_episodes % agent.parameters.update_freq == 0 {
                agent.update_networks();
            }
            curr_obs = Tensor::from_slice(&env.reset(None), (1, 4), device)?;

            n_episodes += 1;
            epi_reward = 0.0;
            action_counter = 0;
        }
        agent.action_selection_update(step as f32 / n_steps as f32, epi_reward);
        if step % agent.parameters.eval_freq == 0 {
            // println!("evaluating");
            let (rewards, eval_lengths) = evaluate(agent, eval_env, agent.parameters.eval_for)?;
            let reward_avg = (rewards.iter().sum::<f32>()) / (rewards.len() as f32);
            let eval_lengths_avg =
                (eval_lengths.iter().map(|x| *x as f32).sum::<f32>()) / (eval_lengths.len() as f32);
            if verbose > 0 {
                println!(
                        "current step: {step} - mean eval reward: {reward_avg:.1} - exploration epsilon: {:.2}",
                        agent.get_epsilon()
                    );
            }
            run.log_metric(
                "mean eval reward",
                reward_avg as f64,
                timestamp(),
                step as u64,
            );

            evaluation_reward.push(reward_avg);
            evaluation_length.push(eval_lengths_avg);
            if step == n_steps || (threshold.is_some() && reward_avg > threshold.unwrap()) {
                training_reward.push(epi_reward);
                training_length.push(action_counter);
                break;
            }
        }
    }

    run.terminate();

    Ok((
        training_reward,
        training_length,
        training_error,
        evaluation_reward,
        evaluation_length,
    ))
}

pub fn evaluate(
    agent: &mut DQNAgent,
    eval_env: &mut CartPole,
    n_episodes: u32,
) -> Result<(Vec<f32>, Vec<u32>)> {
    let mut reward_history: Vec<f32> = vec![];
    let mut episode_length: Vec<u32> = vec![];
    for _episode in 0..n_episodes {
        let mut epi_reward: f32 = 0.0;
        let obs_repr = Tensor::from_slice(&eval_env.reset(None), (1, 4), &agent.device)?;
        let mut curr_action = agent.get_best_action(&obs_repr)?;
        let mut action_counter: u32 = 0;
        loop {
            let (obs, reward, done, truncated) = eval_env.step(curr_action).unwrap();
            let next_obs_repr = Tensor::from_slice(&obs, (1, 4), &agent.device)?;
            let next_action_repr: usize = agent.get_best_action(&next_obs_repr)?;
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
