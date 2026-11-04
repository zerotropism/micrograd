//! Neurons, layers and multi-layer perceptrons on the arena engine: the port of
//! `src/micrograd/nn.py`, written as functions over a slice of parameters.
//!
//! The parameters are laid out as `MLP.parameters()` returns them in Python: layer by layer,
//! neuron by neuron, each neuron's weights then its bias. Given the same starting parameters,
//! both implementations train to the same numbers.

use crate::engine::{Graph, Id};

/// Number of parameters of a perceptron whose layer sizes are `sizes` (inputs first).
pub fn parameter_count(sizes: &[usize]) -> usize {
    sizes.windows(2).map(|pair| (pair[0] + 1) * pair[1]).sum()
}

/// The outputs of the perceptron for one input, as values of `g`.
///
/// Each neuron computes `tanh(b + w0*x0 + w1*x1 + ...)`, summed in that order, as Python's
/// `sum(..., self.b)` does.
pub fn forward(g: &mut Graph, params: &[Id], sizes: &[usize], x: &[Id]) -> Vec<Id> {
    let mut inputs = x.to_vec();
    let mut next = 0;
    for pair in sizes.windows(2) {
        let (nin, nout) = (pair[0], pair[1]);
        let mut outputs = Vec::with_capacity(nout);
        for _ in 0..nout {
            let weights = &params[next..next + nin];
            let mut act = params[next + nin];
            for (&w, &xi) in weights.iter().zip(&inputs) {
                let product = g.mul(w, xi);
                act = g.add(act, product);
            }
            outputs.push(g.tanh(act));
            next += nin + 1;
        }
        inputs = outputs;
    }
    inputs
}

/// Squared error summed over the examples, from a zero start, as Python's `sum(...)` does.
fn loss(g: &mut Graph, params: &[Id], sizes: &[usize], xs: &[Vec<f64>], ys: &[f64]) -> Id {
    let mut total = g.leaf(0.0);
    for (x, &y) in xs.iter().zip(ys) {
        let inputs: Vec<Id> = x.iter().map(|&v| g.leaf(v)).collect();
        let prediction = forward(g, params, sizes, &inputs)[0];
        let target = g.leaf(-y);
        let error = g.add(prediction, target);
        let squared = g.pow(error, 2.0);
        total = g.add(total, squared);
    }
    total
}

/// What `train` returns: the loss before each step, and the parameters after the last one.
pub struct Training {
    pub losses: Vec<f64>,
    pub params: Vec<f64>,
}

/// Plain gradient descent, `steps` times, for a perceptron with one output.
///
/// The parameters are the first values of the graph; each step truncates the graph back to them,
/// builds the loss, backpropagates and moves every parameter against its gradient.
pub fn train(
    sizes: &[usize],
    params: &[f64],
    xs: &[Vec<f64>],
    ys: &[f64],
    steps: usize,
    learning_rate: f64,
) -> Result<Training, String> {
    if sizes.len() < 2 || sizes.last() != Some(&1) {
        return Err("sizes must hold the inputs, then layers ending with one output".into());
    }
    if params.len() != parameter_count(sizes) {
        return Err(format!(
            "{} parameters given, {} expected for sizes {sizes:?}",
            params.len(),
            parameter_count(sizes)
        ));
    }
    if xs.len() != ys.len() || xs.iter().any(|x| x.len() != sizes[0]) {
        return Err(format!(
            "every input must have {} values, one target each",
            sizes[0]
        ));
    }

    let mut g = Graph::new();
    let ids: Vec<Id> = params.iter().map(|&p| g.leaf(p)).collect();
    let mut losses = Vec::with_capacity(steps);
    for _ in 0..steps {
        g.truncate(ids.len());
        let total = loss(&mut g, &ids, sizes, xs, ys);
        losses.push(g.data(total));
        g.backward(total);
        for &p in &ids {
            g.set_data(p, g.data(p) - learning_rate * g.grad(p));
        }
    }
    Ok(Training {
        losses,
        params: ids.iter().map(|&p| g.data(p)).collect(),
    })
}

#[cfg(test)]
mod tests {
    //! The Python tests of `tests/test_nn.py`.
    use super::*;

    const SIZES: [usize; 4] = [3, 4, 4, 1];

    fn dataset() -> (Vec<Vec<f64>>, Vec<f64>) {
        let xs = vec![
            vec![2.0, 3.0, -1.0],
            vec![3.0, -1.0, 0.5],
            vec![0.5, 1.0, 1.0],
            vec![1.0, 1.0, -1.0],
        ];
        (xs, vec![1.0, -1.0, -1.0, 1.0])
    }

    /// Deterministic starting parameters in (-1, 1), without a random number crate.
    fn start(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| ((i * 37 % 101) as f64 / 50.0) - 1.0)
            .collect()
    }

    #[test]
    fn an_mlp_has_the_python_parameter_count() {
        assert_eq!(parameter_count(&SIZES), 4 * 4 + 4 * 5 + 5);
    }

    #[test]
    fn a_forward_pass_returns_one_value_per_output() {
        let mut g = Graph::new();
        let sizes = [3, 4, 2];
        let params: Vec<Id> = start(parameter_count(&sizes))
            .iter()
            .map(|&p| g.leaf(p))
            .collect();
        let x: Vec<Id> = [1.0, -2.0, 3.0].iter().map(|&v| g.leaf(v)).collect();
        assert_eq!(forward(&mut g, &params, &sizes, &x).len(), 2);
    }

    #[test]
    fn training_reduces_the_loss() {
        let (xs, ys) = dataset();
        let trained = train(&SIZES, &start(41), &xs, &ys, 20, 0.05).unwrap();
        assert!(
            trained.losses[19] < trained.losses[0] * 0.5,
            "{:?}",
            trained.losses
        );
    }

    #[test]
    fn the_graph_does_not_grow_from_one_step_to_the_next() {
        let (xs, ys) = dataset();
        let one = train(&SIZES, &start(41), &xs, &ys, 1, 0.05).unwrap();
        let two = train(&SIZES, &start(41), &xs, &ys, 2, 0.05).unwrap();
        assert_eq!(one.losses[0], two.losses[0]);
    }

    #[test]
    fn wrong_shapes_are_refused() {
        let (xs, ys) = dataset();
        assert!(train(&SIZES, &[0.0; 3], &xs, &ys, 1, 0.05).is_err());
        assert!(train(&[3, 4], &start(16), &xs, &ys, 1, 0.05).is_err());
        assert!(train(&SIZES, &start(41), &xs, &ys[..2], 1, 0.05).is_err());
    }
}
