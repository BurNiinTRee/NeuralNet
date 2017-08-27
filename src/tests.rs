use WeightMatrix;
use NeuralNet;
use Layer;
use Perceptron;
use Node;

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
fn matrix() -> WeightMatrix<f64> {
    WeightMatrix::new(&(0..25).map(|i| i as f64).collect::<Vec<_>>(), 1., 5, 5)
}

fn node() -> Box<Node<f64>> {
    Box::new(Perceptron)
}

fn layer(size: usize, prev_size: usize) -> Layer<f64> {
    let mut nodes = Vec::new();
    for _ in 0..size {

        nodes.push(node());
    }
    let layer: Layer<f64> = Layer::new(nodes, prev_size);
    layer
}

#[test]
fn rows() {
    let desired = &[0., 1., 2., 3., 4.];
    assert_eq!(desired, matrix().get_weights(0));
}
#[test]
fn columns() {
    let desired = &[1f64, 6., 11., 16.];
    for (a, b) in desired.iter().zip(matrix().get_column(1).iter()) {
        assert_eq!(&a, b);
    }
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
