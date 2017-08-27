/// A Matrix to store the weights infront of one layer
/// It also contains a `bias` as an additional value
pub struct WeightMatrix<W: Clone> {
    elements: Vec<W>,
    pub bias: W,
    pub width: usize,
    pub height: usize,
}

impl<W: Clone> WeightMatrix<W> {
    /// Creates a new `WeightMatrix` from the given elements, the bias, aswell as its size
    /// #Panics
    /// Giving a slice, with a length not equal to the product of the given dimensions gives a panic
    pub fn new(elements: &[W], bias: W, width: usize, height: usize) -> WeightMatrix<W> {
        assert_eq!(width * height, elements.len());
        WeightMatrix {
            elements: elements.to_vec(),
            bias,
            width,
            height,
        }
    }

    /// Returns the row at the given index
    pub fn get_weights(&self, idx: usize) -> &[W] {
        &self.elements[idx * self.width..idx * self.width + self.width]
    }

    /// Probably completely useless :)
    pub fn get_column(&self, idx: usize) -> Vec<&W> {
        let mut output: Vec<&W> = Vec::new();

        for i in 0..self.height {
            let elem = &self.elements[i * self.width + idx];
            output.push(elem)
        }

        output
    }
}
