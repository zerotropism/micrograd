//! The Python binding, built by maturin: `micrograd_rs.Graph` and `micrograd_rs.train_mlp`.

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

use crate::engine::{Graph, Id};
use crate::nn;

/// The arena engine, driven one operation at a time. Values are integers: their index in the
/// graph. Every call crosses from Python to Rust, which is what the speed comparison measures.
#[pyclass(name = "Graph", module = "micrograd_rs")]
struct PyGraph {
    inner: Graph,
}

impl PyGraph {
    /// An index out of the graph would make Rust panic; Python gets an IndexError instead.
    fn check(&self, ids: &[Id]) -> PyResult<()> {
        match ids.iter().find(|&&id| id >= self.inner.len()) {
            Some(id) => Err(PyIndexError::new_err(format!(
                "value {id} is not in a graph of {}",
                self.inner.len()
            ))),
            None => Ok(()),
        }
    }
}

#[pymethods]
impl PyGraph {
    #[new]
    fn new() -> Self {
        Self {
            inner: Graph::new(),
        }
    }

    fn leaf(&mut self, data: f64) -> Id {
        self.inner.leaf(data)
    }

    fn add(&mut self, a: Id, b: Id) -> PyResult<Id> {
        self.check(&[a, b])?;
        Ok(self.inner.add(a, b))
    }

    fn mul(&mut self, a: Id, b: Id) -> PyResult<Id> {
        self.check(&[a, b])?;
        Ok(self.inner.mul(a, b))
    }

    fn pow(&mut self, a: Id, exponent: f64) -> PyResult<Id> {
        self.check(&[a])?;
        Ok(self.inner.pow(a, exponent))
    }

    fn tanh(&mut self, a: Id) -> PyResult<Id> {
        self.check(&[a])?;
        Ok(self.inner.tanh(a))
    }

    fn exp(&mut self, a: Id) -> PyResult<Id> {
        self.check(&[a])?;
        Ok(self.inner.exp(a))
    }

    fn data(&self, a: Id) -> PyResult<f64> {
        self.check(&[a])?;
        Ok(self.inner.data(a))
    }

    fn grad(&self, a: Id) -> PyResult<f64> {
        self.check(&[a])?;
        Ok(self.inner.grad(a))
    }

    fn set_data(&mut self, a: Id, data: f64) -> PyResult<()> {
        self.check(&[a])?;
        self.inner.set_data(a, data);
        Ok(())
    }

    fn backward(&mut self, out: Id) -> PyResult<()> {
        self.check(&[out])?;
        self.inner.backward(out);
        Ok(())
    }

    fn truncate(&mut self, len: usize) {
        self.inner.truncate(len);
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// Train a one-output perceptron entirely in Rust; returns the loss before each step and the
/// final parameters. `params` are in the order of `micrograd.nn.MLP.parameters()`.
#[pyfunction]
#[pyo3(signature = (sizes, params, xs, ys, steps, learning_rate))]
fn train_mlp(
    sizes: Vec<usize>,
    params: Vec<f64>,
    xs: Vec<Vec<f64>>,
    ys: Vec<f64>,
    steps: usize,
    learning_rate: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    nn::train(&sizes, &params, &xs, &ys, steps, learning_rate)
        .map(|trained| (trained.losses, trained.params))
        .map_err(PyValueError::new_err)
}

#[pymodule]
fn micrograd_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add_function(wrap_pyfunction!(train_mlp, m)?)?;
    Ok(())
}
