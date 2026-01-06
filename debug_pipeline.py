
import os
import cv2
import time
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.face_detector import FaceDetector
from utils.image_utils import decode_image, encode_image

def run_debug():
    print("🚀 Starting Debug Pipeline...")
    
    # Setup
    images_dir = "debug/images"
    if not os.path.exists(images_dir):
        print(f"❌ No debug images folder found at {images_dir}")
        return
        
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("❌ No images found in debug/images. Please add some .jpg files.")
        return
        
    print(f"📂 Found {len(image_files)} test images.")
    
    # Initialize Detector
    print("⏳ Initializing FaceDetector (loading models)...")
    detector = FaceDetector()
    print("✅ Detector ready.")
    
    # Load Images
    batch_images = []
    batch_filenames = []
    
    print("🔄 Preprocessing images...")
    t0 = time.time()
    for fname in image_files:
        path = os.path.join(images_dir, fname)
        with open(path, 'rb') as f:
            img_bytes = f.read()
            img = decode_image(img_bytes)
            if img is not None:
                batch_images.append(img)
                batch_filenames.append(fname)
    t1 = time.time()
    print(f"✅ Loaded {len(batch_images)} images in {t1-t0:.3f}s")
    
    if not batch_images:
        return

    # Run Inference
    print("\n🔍 Running Inference (with Profiling)...")
    print("-" * 50)
    
    # This calls the method we added profiling headers to
    results = detector.extract_embeddings(batch_images, batch_filenames)
    
    print("-" * 50)
    print(f"✅ Processed {len(batch_images)} images.")
    print(f"found {len(results)} faces total.")
    
    print("\n💡 NOTE: Check the lines above starting with '🛑 GPU Profiling' for the breakdown.")

if __name__ == "__main__":
    run_debug()
