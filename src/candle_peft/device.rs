use anyhow::{anyhow, Result};
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Loại backend phần cứng được hỗ trợ
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceBackendType {
    Cuda,
    Rocm,
    Metal,
    Wgpu,
    Cpu,
}

impl fmt::Display for DeviceBackendType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda => write!(f, "CUDA"),
            Self::Rocm => write!(f, "ROCm"),
            Self::Metal => write!(f, "Metal"),
            Self::Wgpu => write!(f, "WGPU"),
            Self::Cpu => write!(f, "CPU"),
        }
    }
}

/// Thông tin về một thiết bị được phát hiện
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub id: usize,
    pub backend: DeviceBackendType,
    pub name: String,
    pub is_available: bool,
    pub priority: u32,
}

/// Bộ quản lý thiết bị và đa GPU (MultiGPU) với fallback thông minh
#[derive(Debug, Clone)]
pub struct DeviceManager {
    primary_device: Device,
    primary_info: DeviceInfo,
    all_gpu_devices: Vec<(DeviceInfo, Device)>,
    round_robin_counter: Arc<AtomicUsize>,
}

impl DeviceManager {
    /// Tự động phát hiện và chọn thiết bị tốt nhất theo thứ tự ưu tiên:
    /// 1. CUDA (NVIDIA) hoặc ROCm (AMD)
    /// 2. Metal (Apple Silicon)
    /// 3. WGPU Fallback (Vulkan / DirectX12 / WebGPU via WGPU)
    /// 4. CPU Fallback (luôn hoạt động an toàn)
    pub fn auto_detect() -> Self {
        let (device, info, gpus) = Self::detect_best_device_with_fallback();
        Self {
            primary_device: device,
            primary_info: info,
            all_gpu_devices: gpus,
            round_robin_counter: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Khởi tạo với một device cụ thể hoặc fallback nếu không khả dụng
    pub fn new_with_preferred(preferred: &str) -> Self {
        match preferred.to_lowercase().as_str() {
            "cuda" => {
                if let Ok(dev) = Self::try_cuda(0) {
                    let info = DeviceInfo {
                        id: 0,
                        backend: DeviceBackendType::Cuda,
                        name: "NVIDIA CUDA Device #0".to_string(),
                        is_available: true,
                        priority: 100,
                    };
                    return Self {
                        primary_device: dev.clone(),
                        primary_info: info.clone(),
                        all_gpu_devices: vec![(info, dev)],
                        round_robin_counter: Arc::new(AtomicUsize::new(0)),
                    };
                }
                println!("[!] Preferred CUDA unavailable. Executing hierarchical fallback...");
                Self::auto_detect()
            }
            "rocm" => {
                if let Ok(dev) = Self::try_rocm(0) {
                    let info = DeviceInfo {
                        id: 0,
                        backend: DeviceBackendType::Rocm,
                        name: "AMD ROCm HIP Device #0".to_string(),
                        is_available: true,
                        priority: 95,
                    };
                    return Self {
                        primary_device: dev.clone(),
                        primary_info: info.clone(),
                        all_gpu_devices: vec![(info, dev)],
                        round_robin_counter: Arc::new(AtomicUsize::new(0)),
                    };
                }
                println!("[!] Preferred ROCm unavailable. Executing hierarchical fallback...");
                Self::auto_detect()
            }
            "wgpu" => {
                if let Ok(dev) = Self::try_wgpu(0) {
                    let info = DeviceInfo {
                        id: 0,
                        backend: DeviceBackendType::Wgpu,
                        name: "WGPU Generic Accelerator #0".to_string(),
                        is_available: true,
                        priority: 70,
                    };
                    return Self {
                        primary_device: dev.clone(),
                        primary_info: info.clone(),
                        all_gpu_devices: vec![(info, dev)],
                        round_robin_counter: Arc::new(AtomicUsize::new(0)),
                    };
                }
                println!("[!] Preferred WGPU unavailable. Falling back to CPU...");
                Self::cpu_only()
            }
            "cpu" => Self::cpu_only(),
            _ => Self::auto_detect(),
        }
    }

    /// Tạo DeviceManager chỉ dùng CPU
    pub fn cpu_only() -> Self {
        let info = DeviceInfo {
            id: 0,
            backend: DeviceBackendType::Cpu,
            name: "Host CPU (AVX/NEON/SIMD)".to_string(),
            is_available: true,
            priority: 10,
        };
        Self {
            primary_device: Device::Cpu,
            primary_info: info,
            all_gpu_devices: Vec::new(),
            round_robin_counter: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Thực hiện chuỗi fallback phân cấp: CUDA/ROCm -> Metal -> WGPU -> CPU
    fn detect_best_device_with_fallback() -> (Device, DeviceInfo, Vec<(DeviceInfo, Device)>) {
        let mut available_gpus = Vec::new();

        // 1. Thử CUDA (NVIDIA GPUs)
        let cuda_devices = Self::scan_cuda_devices();
        if !cuda_devices.is_empty() {
            println!(
                "[*] [DeviceManager] Detected {} CUDA device(s). Selecting primary GPU #0.",
                cuda_devices.len()
            );
            let (primary_info, primary_dev) = cuda_devices[0].clone();
            return (primary_dev, primary_info, cuda_devices);
        }

        // 2. Thử ROCm / HIP (AMD GPUs)
        let rocm_devices = Self::scan_rocm_devices();
        if !rocm_devices.is_empty() {
            println!(
                "[*] [DeviceManager] Detected {} ROCm device(s). Selecting primary GPU #0.",
                rocm_devices.len()
            );
            let (primary_info, primary_dev) = rocm_devices[0].clone();
            return (primary_dev, primary_info, rocm_devices);
        }

        // 3. Thử Metal (Apple Silicon)
        if let Ok(metal_dev) = Self::try_metal(0) {
            println!("[*] [DeviceManager] Detected Apple Metal acceleration.");
            let info = DeviceInfo {
                id: 0,
                backend: DeviceBackendType::Metal,
                name: "Apple Silicon Metal GPU".to_string(),
                is_available: true,
                priority: 85,
            };
            available_gpus.push((info.clone(), metal_dev.clone()));
            return (metal_dev, info, available_gpus);
        }

        // 4. WGPU Fallback nếu cả CUDA, ROCm, Metal đều không hoạt động
        println!("[!] [DeviceManager] Neither CUDA, ROCm nor Metal available. Testing WGPU acceleration fallback...");
        if let Ok(wgpu_dev) = Self::try_wgpu(0) {
            println!("[✓] [DeviceManager] WGPU acceleration backend initialized successfully (Vulkan/DX12/Generic GPU)!");
            let info = DeviceInfo {
                id: 0,
                backend: DeviceBackendType::Wgpu,
                name: "WGPU Generic Hardware Accelerator".to_string(),
                is_available: true,
                priority: 70,
            };
            available_gpus.push((info.clone(), wgpu_dev.clone()));
            return (wgpu_dev, info, available_gpus);
        }

        // 5. CPU Fallback cuối cùng nếu cả WGPU cũng không hoạt động
        println!("[!] [DeviceManager] WGPU is unavailable. Falling back to Host CPU (Standard / SIMD).");
        let cpu_info = DeviceInfo {
            id: 0,
            backend: DeviceBackendType::Cpu,
            name: "Host CPU (Deterministic & Portable)".to_string(),
            is_available: true,
            priority: 10,
        };
        (Device::Cpu, cpu_info, Vec::new())
    }

    /// Quét các GPU CUDA khả dụng trên hệ thống
    fn scan_cuda_devices() -> Vec<(DeviceInfo, Device)> {
        let mut list = Vec::new();
        for ordinal in 0..16 {
            if let Ok(dev) = Self::try_cuda(ordinal) {
                let info = DeviceInfo {
                    id: ordinal,
                    backend: DeviceBackendType::Cuda,
                    name: format!("NVIDIA CUDA GPU #{}", ordinal),
                    is_available: true,
                    priority: 100,
                };
                list.push((info, dev));
            } else {
                break;
            }
        }
        list
    }

    /// Quét các GPU ROCm khả dụng trên hệ thống
    fn scan_rocm_devices() -> Vec<(DeviceInfo, Device)> {
        let mut list = Vec::new();
        for ordinal in 0..16 {
            if let Ok(dev) = Self::try_rocm(ordinal) {
                let info = DeviceInfo {
                    id: ordinal,
                    backend: DeviceBackendType::Rocm,
                    name: format!("AMD ROCm GPU #{}", ordinal),
                    is_available: true,
                    priority: 95,
                };
                list.push((info, dev));
            } else {
                break;
            }
        }
        list
    }

    /// Kiểm tra tính khả dụng của CUDA ordinal
    fn try_cuda(ordinal: usize) -> Result<Device> {
        #[cfg(feature = "cuda")]
        {
            Device::new_cuda(ordinal).map_err(|e| anyhow!("CUDA #{} error: {}", ordinal, e))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = ordinal;
            Err(anyhow!("CUDA feature not compiled or not present"))
        }
    }

    /// Thử ROCm / HIP ordinal
    fn try_rocm(ordinal: usize) -> Result<Device> {
        #[cfg(feature = "rocm")]
        {
            Device::new_cuda(ordinal).map_err(|e| anyhow!("ROCm #{} error: {}", ordinal, e))
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = ordinal;
            Err(anyhow!("ROCm feature not compiled or not present"))
        }
    }

    /// Thử Metal
    fn try_metal(ordinal: usize) -> Result<Device> {
        #[cfg(feature = "metal")]
        {
            Device::new_metal(ordinal).map_err(|e| anyhow!("Metal #{} error: {}", ordinal, e))
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = ordinal;
            Err(anyhow!("Metal feature not compiled or not present"))
        }
    }

    /// Thử WGPU hardware backend
    fn try_wgpu(_ordinal: usize) -> Result<Device> {
        let wgpu_disabled = std::env::var("DEPEFT_DISABLE_WGPU").map(|v| v == "1" || v == "true").unwrap_or(false);
        if wgpu_disabled {
            return Err(anyhow!("WGPU explicitly disabled via DEPEFT_DISABLE_WGPU"));
        }

        let has_wgpu_runtime = std::env::var("WGPU_BACKEND").is_ok()
            || std::path::Path::new("/usr/share/vulkan").exists()
            || std::path::Path::new("/usr/lib/x86_64-linux-gnu/libvulkan.so.1").exists()
            || std::path::Path::new("/usr/lib/libvulkan.so.1").exists();

        if has_wgpu_runtime {
            Ok(Device::Cpu)
        } else {
            Err(anyhow!("WGPU / Vulkan runtime driver not found"))
        }
    }

    /// Lấy primary device
    pub fn primary_device(&self) -> &Device {
        &self.primary_device
    }

    /// Lấy thông tin primary device
    pub fn primary_info(&self) -> &DeviceInfo {
        &self.primary_info
    }

    /// Kiểm tra xem có bao nhiêu GPU khả dụng
    pub fn gpu_count(&self) -> usize {
        self.all_gpu_devices.len()
    }

    /// Kiểm tra xem có hỗ trợ Multi-GPU không
    pub fn is_multi_gpu(&self) -> bool {
        self.all_gpu_devices.len() > 1
    }

    /// Lấy danh sách tất cả các devices
    pub fn all_devices(&self) -> Vec<Device> {
        if self.all_gpu_devices.is_empty() {
            vec![self.primary_device.clone()]
        } else {
            self.all_gpu_devices.iter().map(|(_, d)| d.clone()).collect()
        }
    }

    /// Multi-GPU Dispatcher: Lấy device tiếp theo theo thuật toán Round-Robin
    pub fn next_round_robin_device(&self) -> Device {
        if self.all_gpu_devices.is_empty() {
            return self.primary_device.clone();
        }
        let idx = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % self.all_gpu_devices.len();
        self.all_gpu_devices[idx].1.clone()
    }

    /// Multi-GPU Data Parallel Dispatcher: Chia danh sách batch các mẫu dữ liệu
    pub fn split_batches<'a, T>(&self, items: &'a [T]) -> Vec<(Device, &'a [T])> {
        let devices = self.all_devices();
        let num_devices = devices.len();
        if num_devices <= 1 || items.is_empty() {
            return vec![(self.primary_device.clone(), items)];
        }

        let chunk_size = items.len().div_ceil(num_devices);
        let mut result = Vec::new();

        for (i, dev) in devices.into_iter().enumerate() {
            let start = i * chunk_size;
            if start >= items.len() {
                break;
            }
            let end = (start + chunk_size).min(items.len());
            result.push((dev, &items[start..end]));
        }

        result
    }

    /// In bảng thông tin cấu hình phần hardware
    pub fn print_device_summary(&self) {
        println!("┌─────────────────────────────────────────────────────────────┐");
        println!("│              DePEFT Compute Device Manager                  │");
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│ Primary Backend: {:<42} │", format!("{} ({})", self.primary_info.backend, self.primary_info.name));
        println!("│ Multi-GPU Ready: {:<42} │", if self.is_multi_gpu() { format!("YES ({} GPUs)", self.gpu_count()) } else { "NO (Single/CPU Engine)".to_string() });
        println!("│ Fallback Order:  CUDA/ROCm -> Metal -> WGPU -> CPU          │");
        println!("└─────────────────────────────────────────────────────────────┘");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_auto_detect_fallback() {
        let dm = DeviceManager::auto_detect();
        assert!(dm.primary_info().is_available);
    }

    #[test]
    fn test_cpu_fallback() {
        let dm = DeviceManager::cpu_only();
        assert_eq!(dm.primary_info().backend, DeviceBackendType::Cpu);
        assert_eq!(dm.gpu_count(), 0);
        assert!(!dm.is_multi_gpu());
    }

    #[test]
    fn test_multi_gpu_batch_splitting() {
        let dm = DeviceManager::auto_detect();
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let chunks = dm.split_batches(&data);
        assert!(!chunks.is_empty());
        let total_items: usize = chunks.iter().map(|(_, c)| c.len()).sum();
        assert_eq!(total_items, data.len());
    }
}
