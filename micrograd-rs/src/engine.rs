//! A scalar autograd engine: the Rust port of `src/micrograd/engine.py`.
//!
//! # Why an arena, not `Rc<RefCell<Value>>`
//!
//! In Python, a `Value` holds references to its children and a closure that pushes its gradient
//! to them. The direct Rust translation is `Rc<RefCell<Value>>`: `Rc` because a value can have
//! several parents (the reused node of the tests), `RefCell` because `backward` must change the
//! `grad` of a child that other values also point to. It works, but every access goes through a
//! reference count and a runtime borrow check, a mistake becomes a panic at run time rather than
//! an error at compile time, and a Python binding needs `unsendable` classes.
//!
//! An arena avoids all of it. Every value lives in one `Vec`, owned by a `Graph`, and a value is
//! its index in that `Vec`: a `Copy` integer, nothing to borrow. Children are created before their
//! parents, so they always have smaller indices, and the reverse order of creation is already a
//! topological order: `backward` needs no graph traversal. The price: values only exist inside
//! their graph, and the graph only grows, until it is truncated (see `Graph::truncate`).

/// How a value was made, and from which values. The operands are indices into the graph.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    Leaf,
    Add(usize, usize),
    Mul(usize, usize),
    /// The exponent is a plain number, as in Python, where `__pow__` refuses a `Value`
    Pow(usize, f64),
    Tanh(usize),
    Exp(usize),
}

/// A value of the graph: its index. Cheap to copy, meaningless outside its graph.
pub type Id = usize;

/// Every value of a computation, with its gradient and how it was made.
#[derive(Default)]
pub struct Graph {
    data: Vec<f64>,
    grad: Vec<f64>,
    ops: Vec<Op>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, data: f64, op: Op) -> Id {
        self.data.push(data);
        self.grad.push(0.0);
        self.ops.push(op);
        self.data.len() - 1
    }

    /// A number that is not computed from others: an input or a parameter.
    pub fn leaf(&mut self, data: f64) -> Id {
        self.push(data, Op::Leaf)
    }

    pub fn add(&mut self, a: Id, b: Id) -> Id {
        self.push(self.data[a] + self.data[b], Op::Add(a, b))
    }

    pub fn mul(&mut self, a: Id, b: Id) -> Id {
        self.push(self.data[a] * self.data[b], Op::Mul(a, b))
    }

    pub fn pow(&mut self, a: Id, exponent: f64) -> Id {
        self.push(self.data[a].powf(exponent), Op::Pow(a, exponent))
    }

    /// The same formula as the Python engine, so that both give the same numbers.
    pub fn tanh(&mut self, a: Id) -> Id {
        let e = (2.0 * self.data[a]).exp();
        self.push((e - 1.0) / (e + 1.0), Op::Tanh(a))
    }

    pub fn exp(&mut self, a: Id) -> Id {
        self.push(self.data[a].exp(), Op::Exp(a))
    }

    /// `-a` is `a * -1`, as in Python: the constant becomes a leaf.
    pub fn neg(&mut self, a: Id) -> Id {
        let minus_one = self.leaf(-1.0);
        self.mul(a, minus_one)
    }

    pub fn sub(&mut self, a: Id, b: Id) -> Id {
        let minus_b = self.neg(b);
        self.add(a, minus_b)
    }

    /// `a / b` is `a * b**-1`, as in Python.
    pub fn div(&mut self, a: Id, b: Id) -> Id {
        let inverse = self.pow(b, -1.0);
        self.mul(a, inverse)
    }

    pub fn data(&self, a: Id) -> f64 {
        self.data[a]
    }

    pub fn grad(&self, a: Id) -> f64 {
        self.grad[a]
    }

    pub fn set_data(&mut self, a: Id, data: f64) {
        self.data[a] = data;
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Forget every value created after the first `len`. A training loop keeps its parameters
    /// as the first values of the graph and truncates back to them at each step, so the
    /// memory of one step is reused by the next.
    pub fn truncate(&mut self, len: usize) {
        self.data.truncate(len);
        self.grad.truncate(len);
        self.ops.truncate(len);
    }

    /// Gradients of `out` with respect to every value it depends on.
    ///
    /// Every gradient is reset, `out`'s is set to 1, then each value from `out` down to the first
    /// pushes its gradient to its operands. Indices decrease from parents to children, so a value
    /// has received everything from its parents before it pushes to its children. Each push
    /// adds (`+=`): a value used twice receives from both uses.
    pub fn backward(&mut self, out: Id) {
        self.grad.iter_mut().for_each(|g| *g = 0.0);
        self.grad[out] = 1.0;
        for node in (0..=out).rev() {
            let grad = self.grad[node];
            match self.ops[node] {
                Op::Leaf => {}
                Op::Add(a, b) => {
                    self.grad[a] += grad;
                    self.grad[b] += grad;
                }
                Op::Mul(a, b) => {
                    self.grad[a] += self.data[b] * grad;
                    self.grad[b] += self.data[a] * grad;
                }
                Op::Pow(a, exponent) => {
                    self.grad[a] += exponent * self.data[a].powf(exponent - 1.0) * grad;
                }
                Op::Tanh(a) => {
                    let t = self.data[node];
                    self.grad[a] += (1.0 - t * t) * grad;
                }
                Op::Exp(a) => {
                    self.grad[a] += self.data[node] * grad;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! The Python tests of `tests/test_engine.py`, one for one.
    use super::*;

    const H: f64 = 1e-6;
    const TOLERANCE: f64 = 1e-4;

    /// add, mul, pow, div, tanh, exp and a node reached by two paths.
    fn expression(g: &mut Graph, a: Id, b: Id, c: Id) -> Id {
        let ab = g.mul(a, b);
        let d = g.add(ab, c);
        let dd = g.mul(d, d);
        let two = g.leaf(2.0);
        let a2 = g.add(a, two);
        let e = g.div(dd, a2);
        let te = g.tanh(e);
        let ac = g.mul(a, c);
        let eac = g.exp(ac);
        let tenth = g.leaf(0.1);
        let scaled = g.mul(eac, tenth);
        let d3 = g.pow(d, 3.0);
        let sum = g.add(te, scaled);
        let sum = g.add(sum, d3);
        g.mul(sum, a)
    }

    fn evaluate(values: [f64; 3]) -> f64 {
        let mut g = Graph::new();
        let [a, b, c] = values.map(|v| g.leaf(v));
        let out = expression(&mut g, a, b, c);
        g.data(out)
    }

    #[test]
    fn forward_pass() {
        let mut g = Graph::new();
        let (a, b) = (g.leaf(2.0), g.leaf(-3.0));
        let product = g.mul(a, b);
        let sum = g.add(a, b);
        let difference = g.sub(a, b);
        let quotient = g.div(a, b);
        let cube = g.pow(a, 3.0);
        assert_eq!(g.data(product), -6.0);
        assert_eq!(g.data(sum), -1.0);
        assert_eq!(g.data(difference), 5.0);
        assert!((g.data(quotient) + 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(g.data(cube), 8.0);
    }

    #[test]
    fn analytic_gradient_matches_the_numerical_one() {
        let values = [2.0, -3.0, 0.5];
        let mut g = Graph::new();
        let inputs = values.map(|v| g.leaf(v));
        let out = expression(&mut g, inputs[0], inputs[1], inputs[2]);
        g.backward(out);

        for (index, &input) in inputs.iter().enumerate() {
            let (mut up, mut down) = (values, values);
            up[index] += H;
            down[index] -= H;
            let numeric = (evaluate(up) - evaluate(down)) / (2.0 * H);
            let analytic = g.grad(input);
            assert!(
                (analytic - numeric).abs() <= TOLERANCE * (1.0 + numeric.abs()),
                "input {index}: analytic {analytic}, numerical {numeric}"
            );
        }
    }

    #[test]
    fn a_reused_value_accumulates_gradients_from_every_path() {
        let mut g = Graph::new();
        let a = g.leaf(3.0);
        let out = g.add(a, a);
        g.backward(out);
        assert_eq!(g.grad(a), 2.0);
    }

    #[test]
    fn exp_accumulates_like_every_other_operation() {
        let mut g = Graph::new();
        let a = g.leaf(1.0);
        let e = g.exp(a);
        let out = g.add(e, a);
        g.backward(out);
        assert!((g.grad(a) - (1.0_f64.exp() + 1.0)).abs() < 1e-12);
    }

    #[test]
    fn the_output_gradient_starts_at_one() {
        let mut g = Graph::new();
        let a = g.leaf(2.0);
        g.backward(a);
        assert_eq!(g.grad(a), 1.0);
    }

    #[test]
    fn reverse_creation_order_is_topological() {
        let mut g = Graph::new();
        let a = g.leaf(2.0);
        let three = g.leaf(3.0);
        let b = g.mul(a, three);
        let four = g.leaf(4.0);
        let c = g.mul(b, four);
        g.backward(c);
        assert_eq!(g.grad(a), 12.0);
    }

    #[test]
    fn backward_resets_the_previous_gradients() {
        let mut g = Graph::new();
        let a = g.leaf(3.0);
        let out = g.mul(a, a);
        g.backward(out);
        g.backward(out);
        assert_eq!(g.grad(a), 6.0);
    }

    #[test]
    fn truncate_keeps_the_first_values() {
        let mut g = Graph::new();
        let a = g.leaf(1.0);
        let b = g.leaf(2.0);
        g.add(a, b);
        g.truncate(2);
        assert_eq!(g.len(), 2);
        assert_eq!(g.data(b), 2.0);
    }
}
