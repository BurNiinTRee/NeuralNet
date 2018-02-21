extern crate neuralnet;
extern crate rand;

use rand::Rng;

use neuralnet::*;

fn main() {
    // let mut rng = ZeroRng;
    let mut rng = rand::thread_rng();

    let first_nodes: Vec<Box<ActivationFunction<f64>>> =
        vec![Box::new(Perceptron), Box::new(Perceptron)];
    let second_nodes: Vec<Box<ActivationFunction<f64>>> = vec![Box::new(Perceptron)];
    let first_layer: Layer<f64> = Layer::rnd_layer(first_nodes, 2, 2, &mut rng);
    let second_layer: Layer<f64> = Layer::rnd_layer(second_nodes, 2, 1, &mut rng);

    let training_input = [[0., 0.], [1., 1.], [1., 0.], [0., 1.]];
    let training_output = [[0.], [0.], [1.], [1.]];

    let mut net = NeuralNet::new(vec![first_layer, second_layer]);

    for _ in 0..100 {
        for (input, output) in training_input.iter().zip(training_output.iter()) {
            let error = net.train(input.to_vec(), output, 0.5);
            println!("Current error:\t{}", error);
        }
    }

    let input = vec![0., 1.];
    let output = net.run(input);

    println!("Hey, we got: {:?}", output);
}

struct ZeroRng;

impl Rng for ZeroRng {
    fn next_u32(&mut self) -> u32 {
        0
    }
}
