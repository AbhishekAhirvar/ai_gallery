"""
Gallery view UI logic.
"""

import streamlit as st
from typing import Dict, List

from config import GALLERY_COLUMNS, DEFAULT_EXPANDED_GROUPS
from services.face_clusterer import PersonCluster
from services.file_handler import FileHandler

def render_gallery(
    clusters: List[PersonCluster], 
    file_handler: FileHandler, 
    no_face_images: List[str]
):
    """
    Render the photo gallery.
    
    Args:
        clusters: List of person clusters
        file_handler: File handler instance for retrieving images
        no_face_images: List of filenames for images with no detected faces
    """
    # Summary Box
    total_people = len(clusters)
    total_photos = len(file_handler.get_all_images())
    no_face_count = len(no_face_images)
    
    st.markdown(f"""
    <div class="summary-box">
        <h2>📸 Gallery Ready!</h2>
        <p style="font-size: 1.2rem; margin: 0;">
            Found <strong>{total_people} people</strong> across <strong>{total_photos} photos</strong>
            {f'({no_face_count} photos had no faces)' if no_face_count > 0 else ''}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if we have any clusters to display
    if not clusters and not no_face_images:
        st.warning("No photos found to display.")
        return

    # Render Person Groups
    for idx, cluster in enumerate(clusters):
        expanded = idx < DEFAULT_EXPANDED_GROUPS
        
        with st.expander(f"👤 {cluster.label} ({cluster.photo_count} photos)", expanded=expanded):
            # Header with Download Button
            col_header, col_download = st.columns([3, 1])
            with col_download:
                safe_name = cluster.label.lower().replace(' ', '_')
                zip_buffer = file_handler.create_zip(cluster.filenames, safe_name)
                
                st.download_button(
                    label="📥 Download ZIP",
                    data=zip_buffer,
                    file_name=f"{safe_name}_photos.zip",
                    mime="application/zip",
                    key=f"download_{safe_name}"
                )
            
            # Photo Grid
            _render_photo_grid(cluster.filenames, file_handler)

    # Render No Face Group
    if no_face_images:
        with st.expander(f"❓ No Face Group ({len(no_face_images)} photos)", expanded=False):
            col_header, col_download = st.columns([3, 1])
            with col_download:
                zip_buffer = file_handler.create_zip(no_face_images, "no_face_photos")
                
                st.download_button(
                    label="📥 Download ZIP",
                    data=zip_buffer,
                    file_name="no_face_photos.zip",
                    mime="application/zip",
                    key="download_no_face"
                )
            
            _render_photo_grid(no_face_images, file_handler)
            
    # Reset Button
    st.markdown("---")
    if st.button("🔄 Upload New Photos", type="primary", use_container_width=True):
        st.session_state.processed = False
        file_handler.reset()  # Ideally, this should be handled in app.py's reset logic
        # We trigger rerun in the main app loop or via callback, 
        # but here we can just set the flag and rerun.
        st.session_state.clear() # Clear all state to be safe and restart
        st.rerun()

def _render_photo_grid(filenames: List[str], file_handler: FileHandler):
    """Helper to render a grid of photos."""
    cols = st.columns(GALLERY_COLUMNS)
    for i, filename in enumerate(filenames):
        img_bytes = file_handler.get_image(filename)
        if img_bytes:
            with cols[i % GALLERY_COLUMNS]:
                st.image(img_bytes, caption=filename, use_container_width=True)
