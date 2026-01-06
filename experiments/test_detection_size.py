"""
Automated Detection Size Experiment

Tests detection sizes 640, 480, 384, 320 to find optimal speed/accuracy trade-off.
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from insightface.app import FaceAnalysis
import cv2
import onnxruntime

def get_detector(det_size):
    """Create a fresh detector with specific size"""
    app = FaceAnalysis(name='buffalo_l')
    
    # Check for GPU
    providers = onnxruntime.get_available_providers()
    has_gpu = any(p in providers for p in ['CUDAExecutionProvider', 'ROCMExecutionProvider', 'CoreMLExecutionProvider'])
    
    ctx_id = 0 if has_gpu else -1
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    return app

def run_experiment():
    """Run automated detection size sweep."""
    
    # Load test images
    test_dir = Path("debug/images")
    image_files = sorted(list(test_dir.glob("*.jpg")))[:50]
    
    print(f"\n📂 Loading {len(image_files)} test images...")
    filenames = []
    original_imgs = []
    
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is not None:
            # Don't resize yet - we'll resize based on det_size
            original_imgs.append(img)
            filenames.append(img_file.name)
    
    print(f"✅ Loaded {len(original_imgs)} images\n")
    
    det_sizes = [640, 480, 384, 320]
    results = []
    
    print("="*60)
    print("DETECTION SIZE EXPERIMENT")
    print("="*60 + "\n")
    
    for det_size in det_sizes:
        print(f"\n🧪 Testing det_size={det_size}x{det_size}...")
        
        try:
            # Create detector with specific det_size
            app = get_detector(det_size)
            
            # Resize images to detection size
            imgs = []
            for img in original_imgs:
                h, w = img.shape[:2]
                max_dim = max(h, w)
                if max_dim > det_size * 3:  # Resize if much larger
                    scale = (det_size * 3) / max_dim
                    img_resized = cv2.resize(img, (int(w * scale), int(h * scale)))
                else:
                    img_resized = img
                imgs.append(img_resized)
            
            # Run detection
            total_faces = 0
            start = time.time()
            
            for img in imgs:
                faces = app.get(img)
                total_faces += len(faces)
            
            elapsed = time.time() - start
            
            result = {
                'det_size': det_size,
                'images': len(imgs),
                'faces': total_faces,
                'time': round(elapsed, 3),
                'throughput': round(len(imgs) / elapsed, 2),
                'ms_per_image': round(elapsed / len(imgs) * 1000, 1),
                'success': True
            }
            
            print(f"   ✅ {total_faces} faces in {elapsed:.3f}s ({result['throughput']:.2f} img/s, {result['ms_per_image']:.1f}ms/img)")
            results.append(result)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'det_size': det_size,
                'success': False,
                'error': str(e)
            })
    
    # Save results
    output = Path("experiments/detection_size_results.json")
    output.write_text(json.dumps(results, indent=2))
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print("="*60 + "\n")
    print(f"{'Size':<8} {'Faces':<8} {'Time':<10} {'ms/img':<10} {'Throughput':<15}")
    print("-"*60)
    
    baseline = results[0]  # 640 is baseline
    
    for r in results:
        if r['success']:
            face_diff = r['faces'] - baseline['faces']
            face_pct = (face_diff / baseline['faces'] * 100) if baseline['faces'] > 0 else 0
            speedup = (baseline['time'] - r['time']) / baseline['time'] * 100
            
            print(f"{r['det_size']:<8} {r['faces']:<8} {r['time']:<10.3f} {r['ms_per_image']:<10.1f} {r['throughput']:.2f} img/s")
            if r['det_size'] != 640:
                print(f"         ↳ {face_diff:+d} faces ({face_pct:+.1f}%), {speedup:+.1f}% speed")
    
    # Find best
    successful = [r for r in results if r['success']]
    if successful:
        # Best = fastest with <5% face loss
        candidates = [r for r in successful 
                     if abs(r['faces'] - baseline['faces']) / baseline['faces'] < 0.05]
        
        if candidates:
            best = min(candidates, key=lambda x: x['time'])
            speedup = (baseline['time'] - best['time']) / baseline['time'] * 100
            
            print(f"\n🏆 Best: det_size={best['det_size']}")
            print(f"   Faces found: {best['faces']} (vs {baseline['faces']} at 640)")
            print(f"   Time: {best['time']:.3f}s (vs {baseline['time']:.3f}s)")
            print(f"   Speedup: {speedup:.1f}%")
            print(f"   Trade-off: {abs(best['faces'] - baseline['faces'])} faces difference")
        else:
            print(f"\n⚠️  No size had <5% face loss. Using 640 (baseline).")
    
    print(f"\n💾 Results saved to: {output}\n")

if __name__ == "__main__":
    run_experiment()
