#[cfg(test)]
mod tests;

#[cfg(feature = "rand")]
extern crate rand;

mod matrix;
pub use matrix::WeightMatrix;

pub struct NeuralNet<W>
where
    W: Clone,
    for<'a> &'a W: std::ops::Mul,
{
    layers: Vec<Layer<W>>,
}

impl<W: Clone> NeuralNet<W>
where
    W: Clone + std::iter::Sum<W>,
    for<'a> &'a W: std::ops::Mul<Output = W> + std::ops::Sub<Output = W>,
{
    pub fn new(layers: Vec<Layer<W>>) -> NeuralNet<W> {
        NeuralNet { layers }
    }
    pub fn run(&self, inputs: Vec<W>) -> Vec<W> {
        assert_eq!(inputs.len(), self.layers[0].weights.width);

        let mut prev_outputs = inputs;
        for layer in &self.layers {
            prev_outputs = layer.forward(prev_outputs);
        }
        prev_outputs
    }

    pub fn train(&mut self, inputs: Vec<W>, expected_output: &[W], learning_rate: W) -> W {
        let output = self.run(inputs);
        let errors = output
            .iter()
            .zip(expected_output.iter())
            .map(|(real, expected)| real - expected);
        self.layers
            .iter_mut()
            .zip(errors)
            .map(|(layer, error)| layer.backwards(error, learning_rate.clone()))
            .sum()
    }
}

/// A Layer of a `NeuralNet` with a `WeightMatrix` as well as the nodes.
/// Nodes are stored individually in case you want to create a network,
/// where different nodes have different characteristics.
/// Each Node has a Row in the Matrix.
pub struct Layer<W>
where
    W: Clone,
    for<'a> &'a W: std::ops::Mul,
{
    weights: WeightMatrix<W>,
    nodes: Vec<Box<ActivationFunction<W>>>,
}

// TODO:
// * Make the Networks more useful by giving the possibility of somehow modifying the weights
//   - Unsure if through backpropagation or evolution (sexual or asexual)

impl<W> Layer<W>
where
    W: Clone + std::iter::Sum<W>,
    for<'a> &'a W: std::ops::Mul<Output = W>,
{
    pub fn new(nodes: Vec<Box<ActivationFunction<W>>>, weights: WeightMatrix<W>) -> Layer<W> {
        assert_eq!(weights.height, nodes.len());
        Layer { weights, nodes }
    }
    /// Runs this layer on a set of inputs, producing outputs.
    /// # Panics
    /// Panics, if the number of inputs does not equal the number of expected inputs
    /// of the first layer.
    /// Panics, if any of the contained layers does not produce the number of outputs expected
    /// by the next layer.
    pub fn forward(&self, inputs: Vec<W>) -> Vec<W> {
        assert_eq!(inputs.len(), self.weights.width);
        let mut output = Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            let mut weighted_inputs = Vec::new();
            for (input, weight) in inputs.iter().zip(self.weights.get_weights(i)) {
                weighted_inputs.push(input * weight);
            }
            let val = node.forward(weighted_inputs.as_slice(), self.weights.get_bias(i).clone());
            output.push(val);
        }
        assert_eq!(output.len(), self.nodes.len());
        output
    }

    pub fn backwards(&mut self, error: W, learning_rate: W) -> W {
        (0..self.weights.height)
            .map(|idx| {
                let weights = self.weights.get_mut_weights(idx);
                self.nodes[idx].backward(&error, weights, learning_rate.clone())
            })
            .sum()
    }
}

#[cfg(feature = "rand")]
impl<W> Layer<W>
where
    W: Clone + rand::Rand,
    for<'a> &'a W: std::ops::Mul<Output = W>,
{
    pub fn rnd_layer<R: rand::Rng>(
        nodes: Vec<Box<ActivationFunction<W>>>,
        width: usize,
        height: usize,
        rng: &mut R,
    ) -> Layer<W> {
        let raw_weights = rng.gen_iter()
            .take((width + 1) * height)
            .collect::<Vec<_>>();
        let matrix = WeightMatrix::new(&raw_weights, width, height);
        Layer {
            weights: matrix,
            nodes,
        }
    }
}

pub trait ActivationFunction<W>
where
    W: Clone,
    for<'a> &'a W: std::ops::Mul<Output = W>,
{
    fn forward(&self, inputs: &[W], bias: W) -> W;
    fn backward(&self, error: &W, weights: &mut [W], learning_rate: W) -> W;
}

#[derive(Copy, Clone)]
pub struct Perceptron;

impl<W> ActivationFunction<W> for Perceptron
where
    for<'a> W: Clone + std::iter::Product<W> + std::cmp::PartialOrd + std::iter::Sum<&'a W>,
    for<'a> &'a W: std::ops::Mul<Output = W> + std::ops::Sub<Output = W>,
{
    fn forward(&self, inputs: &[W], bias: W) -> W {
        if inputs.into_iter().sum::<W>() > bias {
            std::iter::empty().product::<W>()
        } else {
            std::iter::empty::<&W>().sum::<W>()
        }
    }
    fn backward(&self, error: &W, weights: &mut [W], learning_rate: W) -> W {
        let correction = &learning_rate * error;
        for w in weights.iter_mut() {
            *w = &*w * &correction;
        }
        error * &((&std::iter::empty::<&W>().sum::<W>() - &learning_rate))
    }
}
