pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self {
            data: vec![0.0; n],
            shape,
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Self {
        assert_eq!(self.data.len(), shape.iter().product::<usize>());
        Self {
            data: self.data.clone(),
            shape,
        }
    }
}
