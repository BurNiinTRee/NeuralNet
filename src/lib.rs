#[cfg(test)]
mod tests;



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
    W: Clone,
    for<'a> &'a W: std::ops::Mul<Output = W>,
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
    nodes: Vec<Box<Node<W>>>,
}


// TODO:
// * Make the Networks more useful by giving the possibility of somehow modifying the weights
//   - Unsure if through backpropagation or evolution (sexual or asexual)

impl Layer<f64> {
    /// Creates a new Layer for processing `f64`s, everything else is not yet implemented
    pub fn new(nodes: Vec<Box<Node<f64>>>, num_of_previous_nodes: usize) -> Layer<f64> {
        let weight_width = num_of_previous_nodes;
        let weigth_height = nodes.len();
        let raw_matrix = (0..(weight_width * weigth_height))
            .map(|_| 0.)
            .collect::<Vec<_>>();
        let weights = WeightMatrix::new(&raw_matrix, 1., weight_width, weigth_height);
        Layer { weights, nodes }

    }
}

impl<W> Layer<W>
where
    W: Clone,
    for<'a> &'a W: std::ops::Mul<Output = W>,
{
    /// Runs this layer on a set of inputs, producing outputs.
    /// #Panics
    /// If the number of inputs does not equal the number of expected inputs of the first layer, a panic is given.
    /// If any of the contained layers does not produce the number of outputs expected by the next layer, a panic is given.
    pub fn forward(&self, inputs: Vec<W>) -> Vec<W> {
        assert_eq!(inputs.len(), self.weights.width);
        let mut output = Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            let mut weighted_inputs = Vec::new();
            for (input, weight) in inputs.iter().zip(self.weights.get_weights(i)) {
                weighted_inputs.push(input * weight);
            }
            let val = node.forward(weighted_inputs.as_slice(), self.weights.bias.clone());
            output.push(val);
        }
        assert_eq!(output.len(), self.nodes.len());
        output
    }
}


pub trait Node<W>
where
    W: Clone,
    for<'a> &'a W: std::ops::Mul<Output = W>,
{
    fn forward(&self, inputs: &[W], bias: W) -> W;
}


#[derive(Copy, Clone)]
pub struct Perceptron;

impl Node<f64> for Perceptron {
    fn forward(&self, inputs: &[f64], bias: f64) -> f64 {
        if inputs.iter().sum::<f64>() > bias {
            1.
        } else {
            0.
        }
    }
}
