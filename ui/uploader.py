"""
Upload section UI logic.
"""

import streamlit as st
from typing import List, Optional

from config import SUPPORTED_UPLOAD_TYPES

def render_upload_section() -> Optional[List]:
    """
    Render the file upload section.
    
    Returns:
        List of uploaded files if any, else None
    """
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Your Wedding Photos")
    st.markdown("Upload a ZIP file or multiple images (up to 1000 photos). Supported formats: JPG, JPEG, PNG")
    
    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        type=SUPPORTED_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="file_uploader"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    return uploaded_files
