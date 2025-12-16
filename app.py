"""
VowScan - Wedding Photo Gallery with Face Search
Main application entry point.
"""

import streamlit as st
import cv2
import numpy as np

# Set page config MUST be the first Streamlit command
st.set_page_config(
    page_title="VowScan - Wedding Photo Gallery",
    page_icon="💒",
    layout="wide"
)

from services.face_detector import FaceDetector, FaceEmbedding
from services.face_clusterer import FaceClusterer
from services.file_handler import FileHandler
from utils.image_utils import resize_image, decode_image
from ui.components import apply_custom_css, render_header
from ui.uploader import render_upload_section
from ui.gallery import render_gallery


def init_session_state():
    """Initialize session state variables."""
    if 'file_handler' not in st.session_state:
        st.session_state.file_handler = FileHandler()
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'clusters' not in st.session_state:
        st.session_state.clusters = []
    if 'no_face_images' not in st.session_state:
        st.session_state.no_face_images = []


def main():
    """Main application loop."""
    init_session_state()
    apply_custom_css()
    render_header()
    
    # 1. Gallery View (if processed)
    if st.session_state.processed:
        render_gallery(
            clusters=st.session_state.clusters,
            file_handler=st.session_state.file_handler,
            no_face_images=st.session_state.no_face_images
        )
        return

    # 2. Upload View
    uploaded_files = render_upload_section()
    
    if uploaded_files:
        if st.button("🚀 Process Photos", type="primary", use_container_width=True):
            _process_workflow(uploaded_files)


def _process_workflow(uploaded_files):
    """Execute the processing workflow."""
    file_handler = st.session_state.file_handler
    detector = FaceDetector()
    clusterer = FaceClusterer()
    
    try:
        # Progress UI
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 1. Extract Images
        status_text.text("Extracting images...")
        image_files = file_handler.extract_images_from_uploads(uploaded_files)
        
        if not image_files:
            st.error("No valid images found in uploads.")
            return

        st.info(f"📁 {len(image_files)} images ready for processing")
        
        # 2. Face Detection & Embedding
        all_embeddings: List[FaceEmbedding] = []
        no_face_images: List[str] = []
        total_images = len(image_files)
        
        with st.spinner("Analyzing faces... (this may take a while)"):
            for idx, img_file in enumerate(image_files):
                status_text.text(f"Processing {idx + 1}/{total_images}: {img_file.filename}")
                progress_bar.progress((idx + 1) / total_images)
                
                # Store original for display
                file_handler.store_image(img_file.filename, img_file.data)
                
                # Decode and resize for processing
                img = decode_image(img_file.data)
                if img is None:
                    continue
                    
                img_processed = resize_image(img)
                
                # Detect faces
                embeddings = detector.extract_embeddings(img_processed, img_file.filename)
                
                if embeddings:
                    all_embeddings.extend(embeddings)
                else:
                    no_face_images.append(img_file.filename)
        
        # 3. Clustering
        status_text.text("Grouping faces...")
        clusters = clusterer.cluster(all_embeddings)
        
        # 4. Update State
        st.session_state.clusters = clusters
        st.session_state.no_face_images = no_face_images
        st.session_state.processed = True
        
        # Cleanup detector resources
        detector.cleanup()
        
        status_text.text("✅ Processing complete!")
        progress_bar.progress(1.0)
        st.rerun()
        
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        # Ideally log the error
        print(f"Error: {e}")
    finally:
        detector.cleanup()


if __name__ == "__main__":
    main()
