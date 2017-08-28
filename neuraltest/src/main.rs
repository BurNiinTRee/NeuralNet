extern crate neuralnet;
extern crate rand;

use rand::Rng;

use neuralnet::*;



fn main() {
    let mut rng = ZeroRng;
    let (width, height) = (5, 5);

    let nodes: Vec<Box<ActivationFunction<f64>>> = (0..height)
        .map(|_| -> Box<ActivationFunction<f64>> { Box::new(Perceptron) })
        .collect();
    let layer: Layer<f64> = Layer::rnd_layer(nodes, width, height, &mut rng);

    let net = NeuralNet::new(vec![layer]);

    let input = vec![1., 2., 3., 4., 5.];
    let output = net.run(input);

    println!("Hey, we got: {:?}", output);
}

struct ZeroRng;

impl Rng for ZeroRng {
    fn next_u32(&mut self) -> u32 {
        0
    }
}
