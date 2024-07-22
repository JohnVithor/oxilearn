#[derive(Debug)]
pub struct ParametersPPO {
    pub gamma: f64,
    pub learning_rate: f64,
    pub current_lr: f64,
    pub gae_lambda: f64,
    pub batch_size: usize,
    pub minibatch_size: usize,
    pub update_epochs: usize,
    pub clip_coef: f64,
    pub norm_adv: bool,
    pub clip_vloss: bool,
    pub ent_coef: f64,
    pub vf_coef: f64,
    pub max_grad_norm: f64,
    pub target_kl: Option<f64>,
}

impl Default for ParametersPPO {
    fn default() -> Self {
        Self::new()
    }
}

impl ParametersPPO {
    pub fn new() -> Self {
        Self {
            gamma: 0.99,
            learning_rate: 0.0005,
            current_lr: 0.0005,
            gae_lambda: 0.95,
            batch_size: 64,
            minibatch_size: 16,
            update_epochs: 4,
            clip_coef: 0.2,
            norm_adv: true,
            clip_vloss: true,
            ent_coef: 0.01,
            vf_coef: 0.5,
            max_grad_norm: 0.5,
            target_kl: None,
        }
    }

    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self.current_lr = learning_rate;
        self
    }

    pub fn gae_lambda(mut self, gae_lambda: f64) -> Self {
        self.gae_lambda = gae_lambda;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn minibatch_size(mut self, minibatch_size: usize) -> Self {
        self.minibatch_size = minibatch_size;
        self
    }

    pub fn update_epochs(mut self, update_epochs: usize) -> Self {
        self.update_epochs = update_epochs;
        self
    }

    pub fn clip_coef(mut self, clip_coef: f64) -> Self {
        self.clip_coef = clip_coef;
        self
    }

    pub fn norm_adv(mut self, norm_adv: bool) -> Self {
        self.norm_adv = norm_adv;
        self
    }

    pub fn clip_vloss(mut self, clip_vloss: bool) -> Self {
        self.clip_vloss = clip_vloss;
        self
    }

    pub fn ent_coef(mut self, ent_coef: f64) -> Self {
        self.ent_coef = ent_coef;
        self
    }

    pub fn vf_coef(mut self, vf_coef: f64) -> Self {
        self.vf_coef = vf_coef;
        self
    }

    pub fn max_grad_norm(mut self, max_grad_norm: f64) -> Self {
        self.max_grad_norm = max_grad_norm;
        self
    }

    pub fn target_kl(mut self, target_kl: Option<f64>) -> Self {
        self.target_kl = target_kl;
        self
    }
}
