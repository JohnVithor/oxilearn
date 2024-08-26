use candle_core::Result;
use candle_core::{Device, Tensor};
use candle_nn::{Module, VarBuilder, VarMap};

pub type PolicyGenerator = dyn Fn(Device) -> Result<(Box<dyn Module>, VarMap)>;
pub type ActivationFunction = fn(&Tensor) -> Result<Tensor>;

pub fn generate_policy(
    net_arch: Vec<(usize, ActivationFunction)>,
    last_activation: ActivationFunction,
    input: usize,
    output: usize,
) -> Box<PolicyGenerator> {
    Box::new(move |device: Device| -> Result<(Box<dyn Module>, VarMap)> {
        let iter = net_arch.clone().into_iter().enumerate();
        let mut previous = input;
        let mem_policy = VarMap::new();
        let vs = VarBuilder::from_varmap(&mem_policy, candle_core::DType::F32, &device);
        let mut policy_net = candle_nn::seq();

        for (i, (neurons, activation)) in iter {
            policy_net = policy_net
                .add(candle_nn::linear(
                    previous,
                    neurons,
                    vs.pp(format!("{}", i * 2)),
                )?)
                .add(candle_nn::func(activation));
            previous = neurons;
        }
        policy_net = policy_net
            .add(candle_nn::linear(
                previous,
                output,
                vs.pp(format!("{}", net_arch.len() * 2)),
            )?)
            .add(candle_nn::func(last_activation));
        Ok((Box::new(policy_net), mem_policy))
    })
}
