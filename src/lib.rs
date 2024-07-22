pub mod dqn;
pub mod env;
pub mod optimizer_enum;
pub mod ppo;

#[derive(Debug, Clone)]
pub enum OxiLearnErr {
    MethodNotFound(String),
    ExpectedItemNotFound,
    DifferentTypeExpected,
    ExpectedDataMissing,
    SpaceNotSupported,
    EnvNotSupported,
}
