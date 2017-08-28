/// A Matrix to store the weights infront of one layer
/// It also contains a `bias` as an additional value
pub struct WeightMatrix<W: Clone> {
    elements: Vec<W>,
    pub width: usize,
    pub height: usize,
}

impl<W: Clone> WeightMatrix<W> {
    /// Creates a new `WeightMatrix` from the given elements, the bias, aswell as its size
    /// #Panics
    /// Giving a slice, with a length not equal to the product of the given dimensions gives a panic
    pub fn new(elements: &[W], width: usize, height: usize) -> WeightMatrix<W> {
        assert_eq!((width + 1) * height, elements.len());
        WeightMatrix {
            elements: elements.to_vec(),
            width,
            height,
        }
    }

    /// Returns the row at the given index
    pub fn get_weights(&self, idx: usize) -> &[W] {
        &self.elements[idx * self.width + idx..idx * self.width + idx + self.width]
    }

    pub fn get_bias(&self, idx: usize) -> &W {
        &self.elements[(self.width + 1) * (idx + 1) - 1]
    }
}
