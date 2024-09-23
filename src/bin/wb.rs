use std::collections::HashMap;
use std::process::Command;

struct WandbRun {
    project: String,
    entity: Option<String>,
    run_id: Option<String>,
}

impl WandbRun {
    // Initialize a new WandB run
    fn init(project: &str, entity: Option<&str>) -> Self {
        let mut command = Command::new("wandb");
        command.arg("init").arg("--project").arg(project);

        if let Some(entity) = entity {
            command.arg("--entity").arg(entity);
        }

        command.status().expect("Failed to initialize WandB run");

        WandbRun {
            project: project.to_string(),
            entity: entity.map(|e| e.to_string()),
            run_id: None,
        }
    }

    // Log metrics to WandB
    fn log_metric(&self, metrics: HashMap<&str, f64>) {
        for (key, value) in metrics {
            Command::new("wandb")
                .arg("log")
                .arg(format!("{}={}", key, value))
                .status()
                .expect("Failed to log metrics");
        }
    }

    // Log hyperparameters (config) to WandB
    fn log_config(&self, config: HashMap<&str, &str>) {
        for (key, value) in config {
            Command::new("wandb")
                .arg("config")
                .arg(format!("{}={}", key, value))
                .status()
                .expect("Failed to log config");
        }
    }

    // Finish the run
    fn finish(&self) {
        Command::new("wandb")
            .arg("finish")
            .status()
            .expect("Failed to finish the WandB run");
    }
}

fn main() {
    // Initialize WandB run
    let wandb = WandbRun::init("my_project", Some("my_entity"));

    // Log some metrics
    let mut metrics = HashMap::new();
    metrics.insert("accuracy", 0.95);
    metrics.insert("loss", 0.05);
    wandb.log_metric(metrics);

    // Log some hyperparameters
    let mut config = HashMap::new();
    config.insert("learning_rate", "0.001");
    config.insert("batch_size", "32");
    wandb.log_config(config);

    // Finish the WandB run
    wandb.finish();
}
