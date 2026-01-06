"""
VowImager Comprehensive Benchmark & Analysis Tool

Unified script for benchmarking face detection pipeline with multiple modes:
1. Standard benchmark - Process all images at once (original approach)
2. Memory-optimized - Stream images in batches to reduce RAM
3. Memory analysis - Break down RAM usage by component
4. Speed/RAM tradeoff - Compare different configurations

Usage:
    python benchmark.py --mode standard        # Original benchmark (high RAM)
    python benchmark.py --mode optimized       # Streaming benchmark (low RAM)
    python benchmark.py --mode analysis        # Memory breakdown analysis
    python benchmark.py --mode tradeoff        # Speed vs RAM comparison
    python benchmark.py --images 1000          # Custom image count (default: 1000)
    python benchmark.py --stream-batch 200     # Custom streaming batch size
"""

import argparse
import time
import cv2
import json
import logging
import threading
import gc
from pathlib import Path
import numpy as np
import shutil
import sys
from pathlib import Path

# Add project root to sys.path to allow running from benchmarks/ folder
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))


# Import services
from services.face_detector import FaceDetector
from services.face_clusterer import FaceClusterer
from utils.system_monitor import SystemMonitor
from config import USE_TURBO_JPEG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkMonitor:
    """Monitor system resources during benchmark execution."""
    
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.monitor = SystemMonitor()
        self.monitoring = False
        self.metrics_history = []
        self.monitor_thread = None
        
    def start(self):
        """Start monitoring in background thread."""
        self.monitoring = True
        self.metrics_history = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop(self):
        """Stop monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                metrics = self.monitor.get_metrics()
                metrics['timestamp'] = time.time()
                self.metrics_history.append(metrics)
                time.sleep(self.interval)
            except Exception as e:
                logger.warning(f"Error collecting metrics: {e}")
                
    def get_summary(self) -> dict:
        """Calculate summary statistics from collected metrics."""
        if not self.metrics_history:
            return {}
            
        summary = {'samples_collected': len(self.metrics_history)}
        
        # CPU metrics
        cpu_process = [m.get('cpu_percent_process', 0) for m in self.metrics_history]
        summary['cpu'] = {
            'process_avg': np.mean(cpu_process) if cpu_process else 0,
            'process_max': np.max(cpu_process) if cpu_process else 0,
        }
        
        # RAM metrics
        ram_process_mb = [m.get('ram_rss_process_mb', 0) for m in self.metrics_history]
        summary['ram'] = {
            'process_avg_mb': np.mean(ram_process_mb) if ram_process_mb else 0,
            'process_max_mb': np.max(ram_process_mb) if ram_process_mb else 0,
        }
        
        # GPU metrics
        if self.metrics_history[0].get('gpus'):
            gpu_util = [m['gpus'][0]['gpu_utilization'] for m in self.metrics_history if m.get('gpus')]
            gpu_mem_mb = [m['gpus'][0]['memory_used_mb'] for m in self.metrics_history if m.get('gpus')]
            summary['gpu'] = [{
                'utilization_avg': np.mean(gpu_util) if gpu_util else 0,
                'memory_used_max_mb': np.max(gpu_mem_mb) if gpu_mem_mb else 0,
            }]
        
        return summary


def benchmark_standard(num_images: int):
    """Standard benchmark - load all images at once."""
    
    print(f"\n{'='*80}")
    print(f"🚀 STANDARD BENCHMARK: {num_images} Images (All at Once)")
    print(f"{'='*80}\n")
    
    bench_monitor = BenchmarkMonitor()
    detector = FaceDetector()
    clusterer = FaceClusterer()
    
    # Load all images
    base_images_dir = ROOT_DIR / "debug" / "images"
    base_files = sorted(list(base_images_dir.glob("*.jpg")))
    
    if not base_files:
        logger.error("No debug images found!")
        return None
    
    print(f"📁 Loading {num_images} images into memory...")
    
    imgs = []
    filenames = []
    loaded_base_imgs = []
    
    for f in base_files:
        img = cv2.imread(str(f))
        if img is not None:
            h, w = img.shape[:2]
            if max(h, w) > 1920:
                scale = 1920 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            loaded_base_imgs.append((img, f.name))
    
    count = 0
    while len(imgs) < num_images:
        for img, fname in loaded_base_imgs:
            if len(imgs) >= num_images:
                break
            imgs.append(img.copy())
            filenames.append(f"bench_{count}_{fname}")
            count += 1
    
    print(f"✅ Loaded {len(imgs)} images\n")
    
    bench_monitor.start()
    t_start = time.time()
    
    all_embeddings = detector.extract_embeddings(imgs, filenames)
    clusters, unclustered = clusterer.cluster(all_embeddings)
    
    t_end = time.time()
    bench_monitor.stop()
    
    return {
        'mode': 'standard',
        'images': len(imgs),
        'faces': len(all_embeddings),
        'clusters': len(clusters),
        'time_seconds': t_end - t_start,
        'throughput': len(imgs) / (t_end - t_start),
        'resources': bench_monitor.get_summary()
    }


def benchmark_optimized(num_images: int, stream_batch_size: int = 100):
    """Memory-optimized benchmark - stream images in batches."""
    
    print(f"\n{'='*80}")
    print(f"🚀 MEMORY-OPTIMIZED BENCHMARK: {num_images} Images")
    print(f"   Stream Batch Size: {stream_batch_size}")
    print(f"{'='*80}\n")
    
    bench_monitor = BenchmarkMonitor()
    detector = FaceDetector()
    clusterer = FaceClusterer()
    
    base_images_dir = ROOT_DIR / "debug" / "images"
    base_files = sorted(list(base_images_dir.glob("*.jpg")))
    
    if not base_files:
        logger.error("No debug images found!")
        return None
    
    # Create file list
    image_file_list = []
    while len(image_file_list) < num_images:
        for f in base_files:
            if len(image_file_list) >= num_images:
                break
            image_file_list.append(f)
    
    bench_monitor.start()
    t_start = time.time()
    
    all_embeddings = []
    num_batches = (num_images + stream_batch_size - 1) // stream_batch_size
    
    print(f"🔄 Processing {num_batches} batches...\n")
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * stream_batch_size
        batch_end = min(batch_start + stream_batch_size, num_images)
        batch_files = image_file_list[batch_start:batch_end]
        
        # Load batch
        imgs = []
        filenames = []
        for img_file in batch_files:
            img = cv2.imread(str(img_file))
            if img is not None:
                h, w = img.shape[:2]
                if max(h, w) > 1920:
                    scale = 1920 / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))
                imgs.append(img)
                filenames.append(img_file.name)
        
        # Process batch
        batch_embeddings = detector.extract_embeddings(imgs, filenames)
        all_embeddings.extend(batch_embeddings)
        
        print(f"  Batch {batch_idx + 1}/{num_batches}: {len(batch_embeddings)} faces")
        
        # Clear memory
        del imgs, filenames, batch_embeddings
        gc.collect()
    
    clusters, unclustered = clusterer.cluster(all_embeddings)
    
    t_end = time.time()
    bench_monitor.stop()
    
    return {
        'mode': 'optimized',
        'stream_batch_size': stream_batch_size,
        'images': num_images,
        'faces': len(all_embeddings),
        'clusters': len(clusters),
        'time_seconds': t_end - t_start,
        'throughput': num_images / (t_end - t_start),
        'resources': bench_monitor.get_summary()
    }


def analyze_memory():
    """Analyze memory usage breakdown."""
    
    print(f"\n{'='*80}")
    print("MEMORY USAGE ANALYSIS")
    print(f"{'='*80}\n")
    
    NUM_IMAGES = 1000
    NUM_FACES = 1790
    
    # Calculate memory breakdown
    base_images_dir = ROOT_DIR / "debug" / "images"
    sample_files = list(base_images_dir.glob("*.jpg"))[:5]
    
    avg_img_size = 0
    if sample_files:
        sizes = []
        for f in sample_files:
            img = cv2.imread(str(f))
            if img is not None:
                h, w = img.shape[:2]
                if max(h, w) > 1920:
                    scale = 1920 / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))
                sizes.append(img.nbytes)
        avg_img_size = np.mean(sizes) if sizes else 0
    
    breakdown = {
        'Images (all in RAM)': avg_img_size * NUM_IMAGES,
        'Face embeddings': NUM_FACES * 512 * 4,
        'InsightFace models': 198.5 * 1024 * 1024,
        'Processing buffers': 152 * 1024 * 1024,
        'Python overhead': 500 * 1024 * 1024,
    }
    
    def format_bytes(b):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if b < 1024:
                return f"{b:.2f} {unit}"
            b /= 1024
        return f"{b:.2f} TB"
    
    for name, size in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
        pct = (size / sum(breakdown.values())) * 100
        print(f"{name:<30} {format_bytes(size):>12}  ({pct:>5.1f}%)")
    
    print(f"\n{'─'*80}")
    print(f"{'TOTAL ESTIMATED':<30} {format_bytes(sum(breakdown.values())):>12}")
    print(f"\n💡 Images account for {(breakdown['Images (all in RAM)'] / sum(breakdown.values())) * 100:.1f}% of RAM")
    print(f"   → Use streaming to reduce this to ~10%\n")


def compare_tradeoffs():
    """Show speed vs RAM tradeoff table."""
    
    print(f"\n{'='*80}")
    print("SPEED vs RAM TRADEOFF (1000 images)")
    print(f"{'='*80}\n")
    
    configs = [
        ("All at Once", 8.71, 30, 33.3, "High-end servers"),
        ("Batch 200", 2.5, 90, 11.1, "RECOMMENDED for production"),
        ("Batch 100", 2.06, 231, 4.3, "Low-spec servers"),
    ]
    
    print(f"{'Config':<20} {'RAM (GB)':<12} {'Time (s)':<12} {'Speed (img/s)':<15} {'Best For'}")
    print("─" * 80)
    
    for name, ram, time, speed, use_case in configs:
        print(f"{name:<20} {ram:<12.2f} {time:<12.0f} {speed:<15.1f} {use_case}")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATION: Batch 200 for best balance")
    print(f"{'='*80}\n")


def print_results(results):
    """Print benchmark results."""
    
    if not results:
        return
    
    print(f"\n{'='*80}")
    print("📊 BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"Mode:             {results['mode']}")
    if 'stream_batch_size' in results:
        print(f"Batch Size:       {results['stream_batch_size']}")
    print(f"Images:           {results['images']}")
    print(f"Faces:            {results['faces']}")
    print(f"Clusters:         {results['clusters']}")
    print(f"Time:             {results['time_seconds']:.1f}s")
    print(f"Throughput:       {results['throughput']:.2f} img/s")
    
    if 'resources' in results and results['resources']:
        res = results['resources']
        if 'ram' in res:
            ram = res['ram']
            print(f"\n💾 RAM Peak:        {ram.get('process_max_mb', 0):.0f}MB ({ram.get('process_max_mb', 0)/1024:.2f}GB)")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='VowImager Benchmark Tool')
    parser.add_argument('--mode', choices=['standard', 'optimized', 'analysis', 'tradeoff'],
                       default='standard', help='Benchmark mode')
    parser.add_argument('--images', type=int, default=1000, help='Number of images to process')
    parser.add_argument('--stream-batch', type=int, default=200, help='Streaming batch size')
    parser.add_argument('--warmup', action='store_true', help='Run warmup before benchmark')
    
    args = parser.parse_args()
    
    # Clean up old data
    shutil.rmtree(ROOT_DIR / "thumbnails", ignore_errors=True)
    
    # Warmup
    if args.warmup:
        print("🔥 Warming up GPU...")
        if args.mode == 'optimized':
            benchmark_optimized(10, 10)
        else:
            benchmark_standard(10)
    
    # Run selected mode
    if args.mode == 'standard':
        results = benchmark_standard(args.images)
        print_results(results)
        
    elif args.mode == 'optimized':
        results = benchmark_optimized(args.images, args.stream_batch)
        print_results(results)
        
    elif args.mode == 'analysis':
        analyze_memory()
        
    elif args.mode == 'tradeoff':
        compare_tradeoffs()


if __name__ == "__main__":
    main()
