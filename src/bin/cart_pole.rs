use std::env;

use candle_core::Result;
use candle_core::{Device, Tensor};
use candle_nn::loss::mse;
use oxilearn::dqn::agent::{DQNAgent, ParametersDQN};
use oxilearn::dqn::epsilon_greedy::{EpsilonGreedy, EpsilonUpdateStrategy};
use oxilearn::dqn::experience_buffer::RandomExperienceBuffer;
use oxilearn::dqn::policy::generate_policy;
use terrarium::classic_control::cart_pole::CartPole;
use terrarium::Environment;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let seed = args[1].parse::<u64>().unwrap();
    let verbose = args[2].parse::<usize>().unwrap();

    let device = Device::cuda_if_available(0)?;

    let a = Tensor::from_slice(&[1.0, 1.0, 1.0], (1, 3), &device)?;
    let b = Tensor::from_slice(
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        (3, 3),
        &device,
    )?;

    let c = (a.matmul(&b))?;
    println!("{c}");

    let mut train_env = CartPole::new(500, seed);
    let eval_env = CartPole::new(500, seed);

    let update_strategy = EpsilonUpdateStrategy::EpsilonLinearTrainingDecreasing {
        start: 1.0,
        end: 0.05,
        end_fraction: 0.2,
    };
    let action_selector = EpsilonGreedy::new(1.0, seed + 2, update_strategy);

    let mem_replay = RandomExperienceBuffer::new(10_000, 1000, seed + 3, device.clone())?;
    let policy = generate_policy(
        vec![(256, |x: &Tensor| x.relu()), (256, |x: &Tensor| x.relu())],
        |xs: &Tensor| Ok(xs.clone()),
        4,
        2,
    );

    let mut model = DQNAgent::new(
        action_selector,
        mem_replay,
        policy,
        oxilearn::optimizer_enum::OptimizerConfig::SgdConfig(0.001),
        mse,
        ParametersDQN {
            learning_rate: 0.001,
            gradient_steps: 1,
            train_freq: 1,
            batch_size: 32,
            update_freq: 10,
            eval_freq: 1000,
            eval_for: 10,
            discount_factor: 0.99,
            max_grad_norm: 1.0,
        },
        device.clone(),
    )?;

    let mut state = train_env.reset(None);

    for _ in 0..1000 {
        println!("state {:?}", state);
        let s = Tensor::from_slice(&state, (1, 4), &device)?;

        let action = model.get_best_action(&s)?;
        let (next_state, reward, terminated, done) = train_env.step(action).unwrap();
        state = next_state;
    }

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
