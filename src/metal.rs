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

        pub fn matmul_gpu(&self, a: &Tensor, b: &Tensor) -> Option<Tensor> {
            let library = self.library.as_ref()?;

            let (m, k) = (a.shape[0], a.shape[1]);
            let n = b.shape[1];

            if a.shape[1] != b.shape[0] {
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
            let c_buffer = self.device.new_buffer(
                result_size as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Create compute pipeline
            let kernel_name = "matmul_kernel";
            let function = library.get_function(kernel_name, None).ok()?;

            let pipeline_descriptor = ComputePipelineDescriptor::new();
            pipeline_descriptor.set_compute_function(Some(&function));

            let pipeline = self.device.new_compute_pipeline_state(&pipeline_descriptor).ok()?;

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

                let thread_groups = MTLSize::new(((m + 15) / 16) as u64, ((n + 15) / 16) as u64, 1);
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

    pub fn has_gpu(&self) -> bool {
        false
    }
}
