import psutil
import os
import time
import logging
from typing import Dict, Any, Optional

# Try to import pynvml for GPU stats
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

logger = logging.getLogger(__name__)

class SystemMonitor:
    def __init__(self, process_id: Optional[int] = None):
        """
        Initialize SystemMonitor.
        
        Args:
            process_id: Optional PID to monitor specifically. Defaults to current process.
        """
        self.process = psutil.Process(process_id or os.getpid())
        self.gpu_handles = []
        
        self.has_gpu = HAS_PYNVML
        
        if self.has_gpu:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    self.gpu_handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
                logger.info(f"SystemMonitor initialized with {len(self.gpu_handles)} GPUs.")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitoring: {e}")
                self.has_gpu = False

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current system metrics.
        
        Returns:
            Dict containing CPU, RAM, and GPU metrics.
        """
        metrics = {}
        
        # CPU & Memory (Global)
        metrics['cpu_percent_global'] = psutil.cpu_percent(interval=None)
        virtual_mem = psutil.virtual_memory()
        metrics['ram_percent_global'] = virtual_mem.percent
        metrics['ram_used_global_gb'] = virtual_mem.used / (1024**3)
        metrics['ram_total_global_gb'] = virtual_mem.total / (1024**3)

        # CPU & Memory (Current Process)
        try:
            with self.process.oneshot():
                metrics['cpu_percent_process'] = self.process.cpu_percent(interval=None)
                mem_info = self.process.memory_info()
                metrics['ram_rss_process_mb'] = mem_info.rss / (1024**2) # Resident Set Size
                metrics['ram_vms_process_mb'] = mem_info.vms / (1024**2) # Virtual Memory Size
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

        # GPU Metrics
        if self.has_gpu:
            gpu_stats = []
            try:
                for i, handle in enumerate(self.gpu_handles):
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    gpu_stats.append({
                        'id': i,
                        'name': pynvml.nvmlDeviceGetName(handle),
                        'memory_used_mb': info.used / (1024**2),
                        'memory_total_mb': info.total / (1024**2),
                        'memory_percent': (info.used / info.total) * 100,
                        'gpu_utilization': util.gpu,
                        'temperature_c': temp
                    })
            except Exception as e:
                logger.warning(f"Error reading GPU metrics: {e}")
            
            metrics['gpus'] = gpu_stats

        return metrics

    def format_metrics(self, metrics: Dict[str, Any]) -> str:
        """
        Format metrics into a readable string.
        """
        lines = []
        lines.append(f"CPU (Global): {metrics.get('cpu_percent_global', 0):.1f}%")
        lines.append(f"RAM (Global): {metrics.get('ram_used_global_gb', 0):.2f}GB / {metrics.get('ram_total_global_gb', 0):.2f}GB ({metrics.get('ram_percent_global', 0):.1f}%)")
        
        if 'ram_rss_process_mb' in metrics:
             lines.append(f"App RAM: {metrics['ram_rss_process_mb']:.1f} MB (RSS)")
             
        if 'gpus' in metrics and metrics['gpus']:
            for gpu in metrics['gpus']:
                lines.append(f"GPU {gpu['id']} ({gpu['name']}): Util {gpu['gpu_utilization']}% | VRAM {gpu['memory_used_mb']:.0f}MB / {gpu['memory_total_mb']:.0f}MB ({gpu['memory_percent']:.1f}%) | Temp {gpu['temperature_c']}C")
        elif self.has_gpu: # If we thought we had GPU but getting metrics failed
             lines.append("GPU: Stats unavailable")
        elif HAS_PYNVML: # Library installed but initialization failed
             lines.append("GPU: Monitoring failed to initialize")
        else:
             lines.append("GPU: Monitoring not available (pynvml not installed)")
            
        return " | ".join(lines)

    def __del__(self):
        if hasattr(self, 'has_gpu') and self.has_gpu:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
