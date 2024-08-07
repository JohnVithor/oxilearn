use tch::nn::LinearConfig;
use tch::nn::Module;
use tch::nn::Sequential;
use tch::nn::VarStore;
use tch::Device;
use tch::{nn, Tensor};

use super::categorical::Categorical;

#[derive(Debug)]
pub struct Policy {
    critic: Sequential,
    actor: Sequential,
    vs: nn::VarStore,
}

impl Policy {
    pub fn new(obs_shape: i64, num_actions: i64, device: Device) -> Self {
        let config = LinearConfig {
            ws_init: tch::nn::Init::Orthogonal {
                gain: 2.0_f64.sqrt(),
            },
            bs_init: Some(tch::nn::Init::Const(0.0)),
            bias: true,
        };

        let vs: VarStore = VarStore::new(device);

        let config2 = LinearConfig {
            ws_init: tch::nn::Init::Orthogonal { gain: 1.0 },
            bs_init: Some(tch::nn::Init::Const(0.0)),
            bias: true,
        };

        let config3 = LinearConfig {
            ws_init: tch::nn::Init::Orthogonal { gain: 0.01 },
            bs_init: Some(tch::nn::Init::Const(0.0)),
            bias: true,
        };

        let critic = nn::seq()
            .add(nn::linear(
                vs.root() / format!("critic/{}", 1),
                obs_shape,
                64,
                config,
            ))
            .add_fn(|xs| xs.gelu("none"))
            .add(nn::linear(
                vs.root() / format!("critic/{}", 2),
                64,
                64,
                config,
            ))
            .add_fn(|xs| xs.gelu("none"))
            .add(nn::linear(
                vs.root() / format!("critic/{}", 3),
                64,
                1,
                config2,
            ));

        let actor = nn::seq()
            .add(nn::linear(
                vs.root() / format!("actor/{}", 1),
                obs_shape,
                64,
                config,
            ))
            .add_fn(|xs| xs.gelu("none"))
            .add(nn::linear(
                vs.root() / format!("actor/{}", 2),
                64,
                64,
                config,
            ))
            .add_fn(|xs| xs.gelu("none"))
            .add(nn::linear(
                vs.root() / format!("actor/{}", 3),
                64,
                num_actions,
                config3,
            ));
        Policy { critic, actor, vs }
    }

    pub fn get_critic_value(&self, x: &Tensor) -> Tensor {
        self.critic.forward(x)
    }

    pub fn get_action_and_value(
        &self,
        x: &Tensor,
        action: Option<&Tensor>,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let logits = self.actor.forward(x);
        let probs = Categorical::from_logits(logits);
        let action = match action {
            Some(a) => a.shallow_clone(),
            None => probs.sample(&[1]),
        };
        let log_prob = probs.log_prob(&action);
        let entropy = probs.entropy();
        let value = self.critic.forward(x);
        (action, log_prob, entropy, value)
    }

    pub fn get_actor_logits(&self, x: &Tensor) -> Tensor {
        self.actor.forward(x)
    }

    pub fn get_best_action(&self, x: &Tensor) -> Tensor {
        let logits = self.actor.forward(x);
        logits.argmax(-1, true)
    }

    pub fn varstore(&self) -> &nn::VarStore {
        &self.vs
    }
}
