//! The micrograd engine in Rust: `engine` holds the values and their gradients, `nn` the
//! perceptron. With the `python` feature, which maturin turns on, `python` exposes both to Python
//! as the `micrograd_rs` module.

pub mod engine;
pub mod nn;

#[cfg(feature = "python")]
mod python;
