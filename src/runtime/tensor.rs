use std::cell::RefCell;
use std::collections::VecDeque;

pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

// Memory pool for reusing tensor buffers
thread_local! {
    static MEMORY_POOL: RefCell<VecDeque<Vec<f32>>> = RefCell::new(VecDeque::new());
}

pub fn with_capacity(size: usize) -> Vec<f32> {
    MEMORY_POOL.with_borrow_mut(|pool| {
        if let Some(mut buf) = pool.pop_front() {
            if buf.capacity() >= size {
                unsafe { buf.set_len(size); }
                return buf;
            }
        }
        Vec::with_capacity(size)
    })
}

pub fn return_to_pool(buf: Vec<f32>) {
    MEMORY_POOL.with_borrow_mut(|pool| {
        if pool.len() < 32 { // Limit pool size to prevent unbounded growth
            pool.push_back(buf);
        }
    });
}

impl Tensor {
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self {
            data: vec![0.0; n],
            shape,
        }
    }

    pub fn zeros_pooled(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        let mut data = with_capacity(n);
        data.resize(n, 0.0);
        Self { data, shape }
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    pub fn from_vec_pooled(data: Vec<f32>, shape: Vec<usize>) -> Self {
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

    pub fn into_pool(self) {
        return_to_pool(self.data);
    }
}
