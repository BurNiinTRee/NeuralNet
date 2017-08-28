use WeightMatrix;
use NeuralNet;
use Layer;
use Perceptron;
use ActivationFunction;

fn neuralnet() -> NeuralNet<f64> {
    let num_of_inputs = 5;
    let num_of_layers = 5;
    let size_of_layers = 5;


    let layers = (0..num_of_layers)
        .map(|_| layer(size_of_layers, num_of_inputs))
        .collect::<Vec<_>>();

    let net = NeuralNet::new(layers);
    net
}
fn matrix(width: usize, height: usize) -> WeightMatrix<f64> {
    WeightMatrix::new(
        &(0..(width + 1) * height)
            .map(|i| i as f64)
            .collect::<Vec<_>>(),
        5,
        5,
    )
}

fn node() -> Box<ActivationFunction<f64>> {
    Box::new(Perceptron)
}

fn layer(size: usize, prev_size: usize) -> Layer<f64> {
    let mut nodes = Vec::new();
    for _ in 0..size {

        nodes.push(node());
    }
    let raw_weights: Vec<_> = (0..(prev_size + 1) * size).map(|_| 0f64).collect();
    let weights = WeightMatrix::new(&raw_weights, prev_size, size);
    let layer: Layer<f64> = Layer::new(nodes, weights);
    layer
}

#[test]
fn rows() {
    let desired = &[0., 1., 2., 3., 4.];
    assert_eq!(desired, matrix(5, 5).get_weights(0));
}
#[test]
fn layers() {
    let layer = layer(5, 2);
    assert_eq!(layer.weights.width, 2);
    assert_eq!(layer.weights.height, 5);
    assert_eq!(
        layer.forward((0..2).map(|i: usize| i as f64).collect::<Vec<_>>()),
        vec![0., 0., 0., 0., 0.]
    );
}

#[test]
fn neuralnets() {
    let net = neuralnet();

    let inputs = vec![0f64, 1., 2., 3., 4.];

    let output = net.run(inputs);

    assert_eq!(output, vec![0f64, 0., 0., 0., 0.])
}

// TEST if XOR works
fn raw_first_matrix() -> Vec<f64> {
    let mut output = Vec::new();
    let nand_weights = [-1f64, -1., -1.5];
    let or_weights = [1f64, 1., 0.5];
    output.extend_from_slice(&nand_weights);
    output.extend_from_slice(&or_weights);
    output
}
fn raw_second_matrix() -> Vec<f64> {
    vec![1f64, 1., 1.5]
}


fn xor_net() -> NeuralNet<f64> {
    let first_matrix = WeightMatrix::new(&raw_first_matrix(), 2, 2);
    let second_matrix = WeightMatrix::new(&raw_second_matrix(), 2, 1);
    let first_nodes: Vec<Box<ActivationFunction<f64>>> =
        vec![Box::new(Perceptron), Box::new(Perceptron)];
    let second_nodes: Vec<Box<ActivationFunction<f64>>> = vec![Box::new(Perceptron)];
    let first_layer = Layer::new(first_nodes, first_matrix);
    let second_layer = Layer::new(second_nodes, second_matrix);
    NeuralNet::new(vec![first_layer, second_layer])
}

#[test]
fn xor() {
    let net = xor_net();

    assert_eq!(net.run(vec![1., 1.])[0], 0.);
    assert_eq!(net.run(vec![0., 1.])[0], 1.);
    assert_eq!(net.run(vec![0., 0.])[0], 0.);
    assert_eq!(net.run(vec![1., 0.])[0], 1.);

}
