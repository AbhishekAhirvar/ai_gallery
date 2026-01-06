"""
Automated Recognition Batch Size Experiment

Tests batch sizes 8, 12, 16, 24, 32 automatically.
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.face_detector import FaceDetector
import cv2

def run_experiment():
    """Run automated batch size sweep."""
    
    # Load test images
    test_dir = Path("debug/images")
    image_files = sorted(list(test_dir.glob("*.jpg")))[:50]
    
    print(f"\n📂 Loading {len(image_files)} test images...")
    imgs = []
    filenames = []
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is not None:
            h, w = img.shape[:2]
            max_dim = max(h, w)
            if max_dim > 1920:
                scale = 1920 / max_dim
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            imgs.append(img)
            filenames.append(img_file.name)
    
    print(f"✅ Loaded {len(imgs)} images\n")
    
    batch_sizes = [8, 12, 16, 24, 32]
    results = []
    
    print("="*60)
    print("RECOGNITION BATCH SIZE EXPERIMENT")
    print("="*60 + "\n")
    
    for batch_size in batch_sizes:
        print(f"\n🧪 Testing batch_size={batch_size}...")
        
        try:
            # Create detector with specific batch size
            detector = FaceDetector(batch_size=batch_size)
            
            start = time.time()
            embeddings = detector.extract_embeddings(imgs, filenames)
            elapsed = time.time() - start
            
            result = {
                'batch_size': batch_size,
                'images': len(imgs),
                'faces': len(embeddings),
                'time': round(elapsed, 3),
                'throughput': round(len(imgs) / elapsed, 2),
                'success': True
            }
            
            print(f"   ✅ {len(embeddings)} faces in {elapsed:.3f}s ({result['throughput']:.2f} img/s)")
            results.append(result)
            
        except RuntimeError as e:
            if "CUDA" in str(e) or "memory" in str(e).lower():
                print(f"   ❌ OOM - Batch too large")
                results.append({
                    'batch_size': batch_size,
                    'success': False,
                    'error': 'OOM'
                })
                break  # Stop testing larger batches
            else:
                raise
    
    # Save results
    output = Path("experiments/recognition_batch_results.json")
    output.write_text(json.dumps(results, indent=2))
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print("="*60 + "\n")
    print(f"{'Batch':<8} {'Status':<10} {'Time':<10} {'Throughput':<15}")
    print("-"*50)
    
    for r in results:
        if r['success']:
            print(f"{r['batch_size']:<8} ✅ OK      {r['time']:<10.3f} {r['throughput']:<.2f} img/s")
        else:
            print(f"{r['batch_size']:<8} ❌ {r.get('error', 'FAIL'):<7}")
    
    successful = [r for r in results if r['success']]
    if successful:
        best = min(successful, key=lambda x: x['time'])
        baseline = next(r for r in results if r['batch_size'] == 8)
        speedup = (baseline['time'] - best['time']) / baseline['time'] * 100
        
        print(f"\n🏆 Best: batch_size={best['batch_size']}")
        print(f"   Time: {best['time']:.3f}s (vs {baseline['time']:.3f}s baseline)")
        print(f"   Speedup: {speedup:.1f}%")
    
    print(f"\n💾 Results saved to: {output}\n")

if __name__ == "__main__":
    run_experiment()
