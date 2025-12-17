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


def main():
    init_session_state()
    apply_custom_css()
    render_header()
    
    # Logic: Upload -> Auto Process -> Show Gallery
    
    # If not processed, show uploader
    if not st.session_state.processed:
        uploaded_files = render_upload_section()
        
        # Auto-process if files are present
        if uploaded_files:
            process_photos(uploaded_files)
    
    # If processed, show gallery
    else:
        render_gallery(
            clusters=st.session_state.clusters,
            file_handler=st.session_state.file_handler,
            unidentified_filenames=st.session_state.no_face_filenames,
            unclustered_embeddings=st.session_state.unclustered_embeddings
        )


def process_photos(uploaded_files):
    """Execute processing workflow automatically."""
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

        # 2. Detect
        status_text.text("Initializing AI models...")
        detector = FaceDetector() # This uses cached model
        clusterer = FaceClusterer()
        
        all_embeddings = []
        no_face_images = []
        total = len(image_files)
        
        for idx, img_file in enumerate(image_files):
            status_text.text(f"Analyzing photo {idx+1}/{total}...")
            progress_bar.progress((idx + 1) / total)
            
            # Store original
            file_handler.store_image(img_file.filename, img_file.data)
            
            # Decode using robust utils
            from utils.image_utils import decode_image, resize_image, encode_image
            
            img = decode_image(img_file.data)
            if img is None:
                continue
            
            # --- Image Optimization Phase ---
            # 1. Thumb (300px) - Keep this for fast gallery grid
            img_thumb = resize_image(img, max_size=300)
            thumb_bytes = encode_image(img_thumb)
            
            # Store derived versions (Only Thumb + Original now)
            file_handler.store_derived_images(img_file.filename, None, thumb_bytes)
            
            # 2. Model Detection
            # Use config driven size
            from config import MAX_IMAGE_SIZE
            img_det = resize_image(img, max_size=MAX_IMAGE_SIZE)
            
            embeddings = detector.extract_embeddings(img_det, img_file.filename)
            
            if embeddings:
                all_embeddings.extend(embeddings)
            else:
                no_face_images.append(img_file.filename)
            
            # Free memory
            del img, img_thumb, img_det
        
        # 3. Cluster
        status_text.text("Grouping faces...")
        clusters, unclustered_embeddings = clusterer.cluster(all_embeddings)
        
        # Merge unclustered faces with no-face images for the "Unidentified" bucket
        # "Unidentified" view will need to handle both:
        # A) Images with faces that weren't clustered (draw boxes)
        # B) Images with NO faces (just show image)
        
        unclustered_filenames = list(set([fe.filename for fe in unclustered_embeddings]))
        final_unidentified_filenames = list(set(no_face_images + unclustered_filenames))
        
        # 4. Save State
        st.session_state.clusters = clusters
        st.session_state.no_face_filenames = final_unidentified_filenames
        st.session_state.unclustered_embeddings = unclustered_embeddings # For drawing boxes
        st.session_state.processed = True
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error processing: {e}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
