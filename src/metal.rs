//! Apple Silicon Metal acceleration backend (macOS only).
//!
//! This module provides GPU-accelerated matrix operations
//! for neural network inference on Apple Silicon M-series chips.

use crate::runtime::tensor::Tensor;

#[cfg(target_os = "macos")]
mod backend {
    use super::*;
    use metal::*;
    use objc::rc::autoreleasepool;

    pub struct MetalBackend {
        device: Device,
        command_queue: CommandQueue,
        library: Option<Library>,
    }

    impl MetalBackend {
        /// Threshold (bytes) above which we prefer CPU to avoid GPU OOM on large matrices.
        const GPU_MEM_THRESHOLD: u64 = 1024 * 1024 * 1024; // 1 GB

        pub fn new() -> Option<Self> {
            let device = Device::system_default()?;
            let command_queue = device.new_command_queue();

            // Try to load Metal compute shaders
            let library = Self::load_compute_shaders(&device);

            Some(Self {
                device,
                command_queue,
                library,
            })
        }

        pub fn is_available() -> bool {
            Device::system_default().is_some()
        }

        fn load_compute_shaders(device: &Device) -> Option<Library> {
            autoreleasepool(|| {
                let shader_src = include_str!("compute/shaders.metal");
                let compile_options = CompileOptions::new();

                device
                    .new_library_with_source(shader_src, &compile_options)
                    .ok()
            })
        }

        /// Check whether the requested total buffer size exceeds the safe GPU memory threshold.
        fn would_exceed_gpu_memory(&self, total_bytes: u64) -> bool {
            if total_bytes > Self::GPU_MEM_THRESHOLD {
                eprintln!(
                    "  Metal: skipping GPU ({} MB > {} MB threshold)",
                    total_bytes / (1024 * 1024),
                    Self::GPU_MEM_THRESHOLD / (1024 * 1024)
                );
                true
            } else {
                false
            }
        }

        pub fn matmul_gpu(&self, a: &Tensor, b: &Tensor) -> Option<Tensor> {
            let library = self.library.as_ref()?;

            let (m, k) = (a.shape[0], a.shape[1]);
            let n = b.shape[1];

            if a.shape[1] != b.shape[0] {
                return None;
            }

            let total_bytes =
                ((a.data.len() + b.data.len() + m * n) * std::mem::size_of::<f32>()) as u64;
            if self.would_exceed_gpu_memory(total_bytes) {
                return None;
            }

            // Create GPU buffers
            let a_buffer = self.device.new_buffer_with_data(
                a.data.as_ptr() as *const _,
                (a.data.len() * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let b_buffer = self.device.new_buffer_with_data(
                b.data.as_ptr() as *const _,
                (b.data.len() * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let result_size = m * n * std::mem::size_of::<f32>();
            let c_buffer = self
                .device
                .new_buffer(result_size as u64, MTLResourceOptions::StorageModeShared);

            // Create compute pipeline
            let kernel_name = "matmul_kernel";
            let function = library.get_function(kernel_name, None).ok()?;

            let pipeline_descriptor = ComputePipelineDescriptor::new();
            pipeline_descriptor.set_compute_function(Some(&function));

            let pipeline = self
                .device
                .new_compute_pipeline_state(&pipeline_descriptor)
                .ok()?;

            // Prepare constants
            let m_val = m as u32;
            let n_val = n as u32;
            let k_val = k as u32;

            let m_const = self.device.new_buffer_with_data(
                &m_val as *const u32 as *const _,
                std::mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let n_const = self.device.new_buffer_with_data(
                &n_val as *const u32 as *const _,
                std::mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let k_const = self.device.new_buffer_with_data(
                &k_val as *const u32 as *const _,
                std::mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Execute kernel
            autoreleasepool(|| {
                let command_buffer = self.command_queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&a_buffer), 0);
                encoder.set_buffer(1, Some(&b_buffer), 0);
                encoder.set_buffer(2, Some(&c_buffer), 0);
                encoder.set_buffer(3, Some(&m_const), 0);
                encoder.set_buffer(4, Some(&n_const), 0);
                encoder.set_buffer(5, Some(&k_const), 0);

                let thread_groups = MTLSize::new(m.div_ceil(16) as u64, n.div_ceil(16) as u64, 1);
                let threads_per_group = MTLSize::new(16, 16, 1);
                encoder.dispatch_thread_groups(thread_groups, threads_per_group);
                encoder.end_encoding();

                command_buffer.commit();
                command_buffer.wait_until_completed();
            });

            // Read back results
            let mut result_data = vec![0.0f32; m * n];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    c_buffer.contents() as *const f32,
                    result_data.as_mut_ptr(),
                    m * n,
                );
            }

            Some(Tensor::from_vec(result_data, vec![m, n]))
        }

        pub fn relu_gpu(&self, a: &Tensor) -> Option<Tensor> {
            let library = self.library.as_ref()?;
            let len = a.data.len();
            if len == 0 {
                return Some(Tensor::from_vec(vec![], a.shape.clone()));
            }

            let total_bytes = (len * 2 * std::mem::size_of::<f32>()) as u64;
            if self.would_exceed_gpu_memory(total_bytes) {
                return None;
            }

            let a_buffer = self.device.new_buffer_with_data(
                a.data.as_ptr() as *const _,
                (len * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let out_buffer = self.device.new_buffer(
                (len * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let len_val = len as u32;
            let len_buffer = self.device.new_buffer_with_data(
                &len_val as *const u32 as *const _,
                std::mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let function = library.get_function("relu_kernel", None).ok()?;
            let pipeline_descriptor = ComputePipelineDescriptor::new();
            pipeline_descriptor.set_compute_function(Some(&function));
            let pipeline = self
                .device
                .new_compute_pipeline_state(&pipeline_descriptor)
                .ok()?;

            autoreleasepool(|| {
                let command_buffer = self.command_queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&a_buffer), 0);
                encoder.set_buffer(1, Some(&out_buffer), 0);
                encoder.set_buffer(2, Some(&len_buffer), 0);
                let groups = MTLSize::new(len.div_ceil(256) as u64, 1, 1);
                let group_size = MTLSize::new(256, 1, 1);
                encoder.dispatch_thread_groups(groups, group_size);
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
            });

            let mut result = vec![0.0f32; len];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    out_buffer.contents() as *const f32,
                    result.as_mut_ptr(),
                    len,
                );
            }
            Some(Tensor::from_vec(result, a.shape.clone()))
        }

        pub fn add_gpu(&self, a: &Tensor, b: &Tensor) -> Option<Tensor> {
            let library = self.library.as_ref()?;
            if a.shape != b.shape {
                return None;
            }
            let len = a.data.len();
            if len == 0 {
                return Some(Tensor::from_vec(vec![], a.shape.clone()));
            }

            let total_bytes = (len * 3 * std::mem::size_of::<f32>()) as u64;
            if self.would_exceed_gpu_memory(total_bytes) {
                return None;
            }

            let a_buffer = self.device.new_buffer_with_data(
                a.data.as_ptr() as *const _,
                (len * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let b_buffer = self.device.new_buffer_with_data(
                b.data.as_ptr() as *const _,
                (len * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let out_buffer = self.device.new_buffer(
                (len * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let len_val = len as u32;
            let len_buffer = self.device.new_buffer_with_data(
                &len_val as *const u32 as *const _,
                std::mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let function = library.get_function("add_kernel", None).ok()?;
            let pipeline_descriptor = ComputePipelineDescriptor::new();
            pipeline_descriptor.set_compute_function(Some(&function));
            let pipeline = self
                .device
                .new_compute_pipeline_state(&pipeline_descriptor)
                .ok()?;

            autoreleasepool(|| {
                let command_buffer = self.command_queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&a_buffer), 0);
                encoder.set_buffer(1, Some(&b_buffer), 0);
                encoder.set_buffer(2, Some(&out_buffer), 0);
                encoder.set_buffer(3, Some(&len_buffer), 0);
                let groups = MTLSize::new(len.div_ceil(256) as u64, 1, 1);
                let group_size = MTLSize::new(256, 1, 1);
                encoder.dispatch_thread_groups(groups, group_size);
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
            });

            let mut result = vec![0.0f32; len];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    out_buffer.contents() as *const f32,
                    result.as_mut_ptr(),
                    len,
                );
            }
            Some(Tensor::from_vec(result, a.shape.clone()))
        }

        pub fn mul_scalar_gpu(&self, a: &Tensor, s: f32) -> Option<Tensor> {
            let library = self.library.as_ref()?;
            let len = a.data.len();
            if len == 0 {
                return Some(Tensor::from_vec(vec![], a.shape.clone()));
            }

            let total_bytes = (len * 2 * std::mem::size_of::<f32>()) as u64;
            if self.would_exceed_gpu_memory(total_bytes) {
                return None;
            }

            let a_buffer = self.device.new_buffer_with_data(
                a.data.as_ptr() as *const _,
                (len * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let out_buffer = self.device.new_buffer(
                (len * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let s_buffer = self.device.new_buffer_with_data(
                &s as *const f32 as *const _,
                std::mem::size_of::<f32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let len_val = len as u32;
            let len_buffer = self.device.new_buffer_with_data(
                &len_val as *const u32 as *const _,
                std::mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let function = library.get_function("mul_scalar_kernel", None).ok()?;
            let pipeline_descriptor = ComputePipelineDescriptor::new();
            pipeline_descriptor.set_compute_function(Some(&function));
            let pipeline = self
                .device
                .new_compute_pipeline_state(&pipeline_descriptor)
                .ok()?;

            autoreleasepool(|| {
                let command_buffer = self.command_queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&a_buffer), 0);
                encoder.set_buffer(1, Some(&out_buffer), 0);
                encoder.set_buffer(2, Some(&s_buffer), 0);
                encoder.set_buffer(3, Some(&len_buffer), 0);
                let groups = MTLSize::new(len.div_ceil(256) as u64, 1, 1);
                let group_size = MTLSize::new(256, 1, 1);
                encoder.dispatch_thread_groups(groups, group_size);
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
            });

            let mut result = vec![0.0f32; len];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    out_buffer.contents() as *const f32,
                    result.as_mut_ptr(),
                    len,
                );
            }
            Some(Tensor::from_vec(result, a.shape.clone()))
        }

        pub fn softmax_gpu(&self, a: &Tensor, axis: usize) -> Option<Tensor> {
            let library = self.library.as_ref()?;
            let rank = a.shape.len();
            if axis >= rank {
                return None;
            }
            let dim = a.shape[axis];
            let outer: usize = a.shape[..axis].iter().product();
            let inner: usize = a.shape[axis + 1..].iter().product();
            let len = a.data.len();
            if len == 0 {
                return Some(Tensor::from_vec(vec![], a.shape.clone()));
            }

            let total_bytes = (len * 2 * std::mem::size_of::<f32>()) as u64;
            if self.would_exceed_gpu_memory(total_bytes) {
                return None;
            }

            let a_buffer = self.device.new_buffer_with_data(
                a.data.as_ptr() as *const _,
                (len * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let out_buffer = self.device.new_buffer(
                (len * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let dim_val = dim as u32;
            let outer_val = outer as u32;
            let inner_val = inner as u32;
            let dim_const = self.device.new_buffer_with_data(
                &dim_val as *const u32 as *const _,
                std::mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let outer_const = self.device.new_buffer_with_data(
                &outer_val as *const u32 as *const _,
                std::mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let inner_const = self.device.new_buffer_with_data(
                &inner_val as *const u32 as *const _,
                std::mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let function = library.get_function("softmax_kernel", None).ok()?;
            let pipeline_descriptor = ComputePipelineDescriptor::new();
            pipeline_descriptor.set_compute_function(Some(&function));
            let pipeline = self
                .device
                .new_compute_pipeline_state(&pipeline_descriptor)
                .ok()?;

            autoreleasepool(|| {
                let command_buffer = self.command_queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&a_buffer), 0);
                encoder.set_buffer(1, Some(&out_buffer), 0);
                encoder.set_buffer(2, Some(&dim_const), 0);
                encoder.set_buffer(3, Some(&outer_const), 0);
                encoder.set_buffer(4, Some(&inner_const), 0);
                let groups = MTLSize::new(outer.div_ceil(16) as u64, inner.div_ceil(16) as u64, 1);
                let group_size = MTLSize::new(16, 16, 1);
                encoder.dispatch_thread_groups(groups, group_size);
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
            });

            let mut result = vec![0.0f32; len];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    out_buffer.contents() as *const f32,
                    result.as_mut_ptr(),
                    len,
                );
            }
            Some(Tensor::from_vec(result, a.shape.clone()))
        }

        pub fn layer_norm_gpu(
            &self,
            a: &Tensor,
            weight: &Tensor,
            bias: &Tensor,
            eps: f32,
        ) -> Option<Tensor> {
            let library = self.library.as_ref()?;
            let last_dim = *a.shape.last()?;
            let n_elements = a.data.len();
            let n_vectors = n_elements / last_dim;
            if n_elements == 0 {
                return Some(Tensor::from_vec(vec![], a.shape.clone()));
            }

            let total_bytes = ((n_elements * 2 + weight.data.len() + bias.data.len())
                * std::mem::size_of::<f32>()) as u64;
            if self.would_exceed_gpu_memory(total_bytes) {
                return None;
            }

            let a_buffer = self.device.new_buffer_with_data(
                a.data.as_ptr() as *const _,
                (n_elements * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let w_buffer = self.device.new_buffer_with_data(
                weight.data.as_ptr() as *const _,
                (weight.data.len() * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let b_buffer = self.device.new_buffer_with_data(
                bias.data.as_ptr() as *const _,
                (bias.data.len() * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let out_buffer = self.device.new_buffer(
                (n_elements * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let last_dim_val = last_dim as u32;
            let n_vectors_val = n_vectors as u32;
            let last_dim_const = self.device.new_buffer_with_data(
                &last_dim_val as *const u32 as *const _,
                std::mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let eps_const = self.device.new_buffer_with_data(
                &eps as *const f32 as *const _,
                std::mem::size_of::<f32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let n_vectors_const = self.device.new_buffer_with_data(
                &n_vectors_val as *const u32 as *const _,
                std::mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let function = library.get_function("layer_norm_kernel", None).ok()?;
            let pipeline_descriptor = ComputePipelineDescriptor::new();
            pipeline_descriptor.set_compute_function(Some(&function));
            let pipeline = self
                .device
                .new_compute_pipeline_state(&pipeline_descriptor)
                .ok()?;

            autoreleasepool(|| {
                let command_buffer = self.command_queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&a_buffer), 0);
                encoder.set_buffer(1, Some(&w_buffer), 0);
                encoder.set_buffer(2, Some(&b_buffer), 0);
                encoder.set_buffer(3, Some(&out_buffer), 0);
                encoder.set_buffer(4, Some(&last_dim_const), 0);
                encoder.set_buffer(5, Some(&eps_const), 0);
                encoder.set_buffer(6, Some(&n_vectors_const), 0);
                let groups = MTLSize::new(
                    n_vectors.div_ceil(16) as u64,
                    last_dim.div_ceil(16) as u64,
                    1,
                );
                let group_size = MTLSize::new(16, 16, 1);
                encoder.dispatch_thread_groups(groups, group_size);
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
            });

            let mut result = vec![0.0f32; n_elements];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    out_buffer.contents() as *const f32,
                    result.as_mut_ptr(),
                    n_elements,
                );
            }
            Some(Tensor::from_vec(result, a.shape.clone()))
        }

        pub fn has_gpu(&self) -> bool {
            self.library.is_some()
        }
    }

    impl Default for MetalBackend {
        fn default() -> Self {
            Self::new().unwrap_or_else(|| {
                let device = Device::system_default().expect("No Metal device found");
                let command_queue = device.new_command_queue();
                Self {
                    device,
                    command_queue,
                    library: None,
                }
            })
        }
    }
}

#[cfg(target_os = "macos")]
pub use backend::*;

#[cfg(not(target_os = "macos"))]
pub struct MetalBackend;

#[cfg(not(target_os = "macos"))]
impl MetalBackend {
    pub fn new() -> Option<Self> {
        None
    }

    pub fn is_available() -> bool {
        false
    }

    pub fn matmul_gpu(&self, _a: &Tensor, _b: &Tensor) -> Option<Tensor> {
        None
    }

    pub fn relu_gpu(&self, _a: &Tensor) -> Option<Tensor> {
        None
    }

    pub fn add_gpu(&self, _a: &Tensor, _b: &Tensor) -> Option<Tensor> {
        None
    }

    pub fn mul_scalar_gpu(&self, _a: &Tensor, _s: f32) -> Option<Tensor> {
        None
    }

    pub fn softmax_gpu(&self, _a: &Tensor, _axis: usize) -> Option<Tensor> {
        None
    }

    pub fn layer_norm_gpu(
        &self,
        _a: &Tensor,
        _weight: &Tensor,
        _bias: &Tensor,
        _eps: f32,
    ) -> Option<Tensor> {
        None
    }

    pub fn has_gpu(&self) -> bool {
        false
    }
}
