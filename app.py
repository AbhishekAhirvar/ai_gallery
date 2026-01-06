"""
VowScan - Wedding Photo Gallery with Face Search
Main application entry point.
"""

import streamlit as st
import time

# Set page config
st.set_page_config(
    page_title="VowScan",
    page_icon="💒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from services.face_detector import FaceDetector
from services.face_clusterer import FaceClusterer
from services.file_handler import FileHandler
from ui.gallery import render_gallery
from ui.uploader import render_upload_section
from ui.components import apply_custom_css, render_header
from config import MAX_IMAGE_SIZE


def init_session_state():
    if 'file_handler' not in st.session_state:
        st.session_state.file_handler = FileHandler()
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'clusters' not in st.session_state:
        st.session_state.clusters = []
    if 'no_face_filenames' not in st.session_state:
        st.session_state.no_face_filenames = []
    if 'unclustered_embeddings' not in st.session_state:
        st.session_state.unclustered_embeddings = []
    if 'upload_id' not in st.session_state:
        st.session_state.upload_id = None
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'gallery'


def main():
    init_session_state()
    apply_custom_css()
    render_header()
    
    # Logic: Upload -> Auto Process -> Show Gallery
    
    # If not processed, show uploader
    if not st.session_state.processed:
        uploaded_files = render_upload_section()
        
        # Auto-process if files are present AND different from last upload
        if uploaded_files:
            # Create unique ID based on file names and sizes
            current_upload_id = hash(tuple((f.name, f.size) for f in uploaded_files))
            
            # Only process if this is a new upload
            if st.session_state.upload_id != current_upload_id:
                st.session_state.upload_id = current_upload_id
                process_photos(uploaded_files)
    
    else:
        render_gallery(
            clusters=st.session_state.clusters,
            file_handler=st.session_state.file_handler,
            unidentified_filenames=st.session_state.no_face_filenames,
            unclustered_embeddings=st.session_state.unclustered_embeddings
        )


def preprocess_image(img_data: tuple) -> tuple:
    """
    Preprocess a single image: decode and resize.
    
    Args:
        img_data: Tuple of (img_bytes, filename)
        
    Returns:
        Tuple of (img_resized, filename) or (None, None) if failed
    """
    try:
        from utils.image_utils import decode_image, resize_image
        from config import MAX_IMAGE_SIZE
        
        img_bytes, filename = img_data
        
        # Decode
        img = decode_image(img_bytes)
        
        if img is None:
            return None, None
        
        # Resize for detection
        img_resized = resize_image(img, max_size=MAX_IMAGE_SIZE)
        
        return img_resized, filename
        
    except Exception as e:
        print(f"❌ Error preprocessing {img_data[1] if len(img_data) > 1 else 'unknown'}: {e}")
        return None, None



def process_photos(uploaded_files):
    """Execute processing workflow with parallel preprocessing."""
    st.info("🚀 Processing photos... please wait.")
    
    file_handler = st.session_state.file_handler
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 1. Extract
        status_text.text("Extracting images...")
        image_files = file_handler.extract_images_from_uploads(uploaded_files)
        
        if not image_files:
            st.error("No valid images found.")
            return

        # 2. Initialize AI models
        status_text.text("Initializing AI models...")
        detector = FaceDetector() # This uses cached model
        clusterer = FaceClusterer()
        
        
        all_embeddings = []
        no_face_images = []
        total = len(image_files)
        
        # Import for parallel processing
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from config import PREPROCESSING_WORKERS, IMAGE_BATCH_SIZE
        
        # Batch processing configuration
        BATCH_SIZE = IMAGE_BATCH_SIZE  # Use config value
        
        # PARALLEL PREPROCESSING: Process images in parallel batches
        status_text.text(f"Processing {total} images with {PREPROCESSING_WORKERS} workers...")
        
        # ========== PERFORMANCE TRACKING ==========
        preprocessing_start = time.time()
        detection_total_time = 0
        batch_count = 0
        failed_count = 0
        # ==========================================
        
        processed_count = 0
        batch_images = []
        batch_filenames = []
        
        # Read all file bytes first (fast, serial)
        print(f"📂 Reading {total} image files...")
        image_data = []
        for img_file in image_files:
            try:
                img_bytes = img_file.read()
                image_data.append((img_bytes, img_file.filename))
            except Exception as e:
                print(f"❌ Failed to read {getattr(img_file, 'filename', 'unknown')}: {e}")
        
        print(f"✓ Read {len(image_data)} files, starting parallel preprocessing...")
        
        # Use ThreadPoolExecutor for parallel decode/resize
        with ThreadPoolExecutor(max_workers=PREPROCESSING_WORKERS) as executor:
            # Submit all preprocessing tasks
            future_to_data = {executor.submit(preprocess_image, img_data): img_data 
                            for img_data in image_data}
            
            # Process results as they complete
            for future in as_completed(future_to_data):
                img_resized, filename = future.result()
                processed_count += 1
                
                # Update progress
                progress_bar.progress(processed_count / total)
                status_text.text(f"Processing photo {processed_count}/{total}...")
                
                if img_resized is None:
                    failed_count += 1
                    print(f"⚠️ Failed to preprocess image {processed_count}/{total}")
                    continue
                
                print(f"✓ Preprocessed: {filename}")
                
                # Add to batch
                batch_images.append(img_resized)
                batch_filenames.append(filename)
                
                # Process batch when it reaches BATCH_SIZE or at the end
                if len(batch_images) >= BATCH_SIZE or processed_count == total:
                    # GPU BATCH INFERENCE: Process entire batch at once
                    batch_start = time.time()
                    batch_embeddings = detector.extract_embeddings(batch_images, batch_filenames)
                    batch_time = time.time() - batch_start
                    detection_total_time += batch_time
                    batch_count += 1
                    
                    print(f"📊 Batch {batch_count}: {len(batch_images)} images → {len(batch_embeddings)} faces in {batch_time:.2f}s ({len(batch_images)/batch_time:.1f} img/s)")
                    
                    # Separate embeddings by filename
                    embeddings_by_file = {}
                    for emb in batch_embeddings:
                        if emb.filename not in embeddings_by_file:
                            embeddings_by_file[emb.filename] = []
                        embeddings_by_file[emb.filename].append(emb)
                    
                    # Add to all_embeddings and track no-face images
                    for batch_filename in batch_filenames:
                        file_embeddings = embeddings_by_file.get(batch_filename, [])
                        if file_embeddings:
                            all_embeddings.extend(file_embeddings)
                        else:
                            no_face_images.append(batch_filename)
                    
                    # Clear batch
                    batch_images = []
                    batch_filenames = []
        
        preprocessing_time = time.time() - preprocessing_start
        
        # 3. Cluster
        status_text.text("Grouping faces...")
        clustering_start = time.time()
        clusters, unclustered_embeddings = clusterer.cluster(all_embeddings)
        clustering_time = time.time() - clustering_start
        
        # Merge unclustered faces with no-face images for the "Unidentified" bucket
        unclustered_filenames = list(set([fe.filename for fe in unclustered_embeddings]))
        final_unidentified_filenames = list(set(no_face_images + unclustered_filenames))
        
        # ========== PERFORMANCE CALCULATIONS ==========
        total_time = preprocessing_time + clustering_time
        decode_time = preprocessing_time - detection_total_time
        throughput = total / total_time if total_time > 0 else 0
        
        # 4. Save State
        st.session_state.clusters = clusters
        st.session_state.no_face_filenames = final_unidentified_filenames
        st.session_state.unclustered_embeddings = unclustered_embeddings
        st.session_state.processed = True
        
        # ========== PERFORMANCE SUMMARY ==========
        print("\n" + "="*60)
        print("📈 PERFORMANCE REPORT")
        print("="*60)
        print(f"Total Images: {total}")
        print(f"Successfully Processed: {total - failed_count}")
        print(f"Failed to Process: {failed_count}")
        print(f"Total Faces Detected: {len(all_embeddings)}")
        print(f"Person Clusters: {len(clusters)}")
        print(f"\n⏱️  TIMING BREAKDOWN:")
        
        # Safe division with zero checks
        decode_speed = total / decode_time if decode_time > 0 else 0
        detect_speed = total / detection_total_time if detection_total_time > 0 else 0
        
        print(f"  Decode + Resize:  {decode_time:.2f}s ({decode_time/total_time*100:.1f}%) - {decode_speed:.1f} img/s")
        print(f"  Face Detection:   {detection_total_time:.2f}s ({detection_total_time/total_time*100:.1f}%) - {detect_speed:.1f} img/s")
        print(f"  Clustering:       {clustering_time:.2f}s ({clustering_time/total_time*100:.1f}%)")
        print(f"  TOTAL:            {total_time:.2f}s")
        print(f"\n🚀 THROUGHPUT: {throughput:.2f} images/second")
        
        if throughput > 0:
            print(f"📊 Est. time for 1000 images: {1000/throughput:.1f}s")
        print("="*60 + "\n")
        
        # Display success message with stats
        st.success(f"✅ Processed {total} images in {total_time:.1f}s ({throughput:.1f} img/s)")
        st.info(f"📊 Found {len(all_embeddings)} faces in {len(clusters)} groups")
        
        time.sleep(1)  # Brief pause so user can see the message
        st.rerun()
        
    except Exception as e:
        st.error(f"Error processing: {e}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
