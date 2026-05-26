//! Apple Silicon Metal acceleration backend (macOS only).
//!
//! This module provides GPU-accelerated matrix-vector multiplication
//! for MLP inference on Apple Silicon M-series chips.

#[cfg(target_os = "macos")]
mod backend {
    use metal::Device;

    pub struct MetalBackend {
        _device: Device,
    }

    impl MetalBackend {
        pub fn new() -> Option<Self> {
            let device = Device::system_default()?;
            Some(Self {
                _device: device,
            })
        }

        pub fn is_available() -> bool {
            Device::system_default().is_some()
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
}
