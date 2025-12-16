"""
File Handling Service for VowScan.
Manages file uploads, ZIP extraction, and temporary storage.
"""

import io
import os
import zipfile
import tempfile
import shutil
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from config import SUPPORTED_IMAGE_EXTENSIONS


@dataclass
class ImageFile:
    """Represents an uploaded image file."""
    filename: str
    data: bytes
    
    def read(self) -> bytes:
        return self.data


class FileHandler:
    """
    Service for handling file uploads and ZIP operations.
    Manages temporary storage and cleanup.
    """
    
    def __init__(self):
        self._temp_dir: Optional[str] = None
        self._image_store: Dict[str, bytes] = {}
    
    @property
    def temp_dir(self) -> str:
        """Get or create temporary directory."""
        if self._temp_dir is None or not os.path.exists(self._temp_dir):
            self._temp_dir = tempfile.mkdtemp(prefix="vowscan_files_")
        return self._temp_dir
    
    def extract_images_from_uploads(
        self,
        uploaded_files: List
    ) -> List[ImageFile]:
        """
        Extract images from uploaded files (handles both images and ZIPs).
        
        Args:
            uploaded_files: List of Streamlit UploadedFile objects
            
        Returns:
            List of ImageFile objects
        """
        images = []
        
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            
            if filename.lower().endswith('.zip'):
                # Extract images from ZIP
                images.extend(self._extract_from_zip(uploaded_file))
            elif filename.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                # Direct image upload
                file_bytes = uploaded_file.read()
                uploaded_file.seek(0)
                images.append(ImageFile(filename=filename, data=file_bytes))
        
        return images
    
    def _extract_from_zip(self, zip_file) -> List[ImageFile]:
        """Extract images from a ZIP file."""
        images = []
        
        try:
            zip_bytes = zip_file.read()
            zip_file.seek(0)
            
            with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
                for name in zf.namelist():
                    # Skip macOS metadata
                    if name.startswith('__MACOSX'):
                        continue
                    
                    # Check if it's an image
                    if name.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                        img_data = zf.read(name)
                        # Use basename to avoid path issues
                        filename = os.path.basename(name)
                        if filename:  # Skip if empty (directory entries)
                            images.append(ImageFile(filename=filename, data=img_data))
        except Exception:
            pass  # Invalid ZIP, return empty list
        
        return images
    
    def store_image(self, filename: str, data: bytes):
        """Store image data for later retrieval."""
        self._image_store[filename] = data
    
    def get_image(self, filename: str) -> bytes:
        """Retrieve image bytes."""
        return self._image_store.get(filename)

    def get_image_bytes(self, filename: str) -> bytes:
        """Alias for get_image."""
        return self.get_image(filename)
    
    def get_all_images(self) -> Dict[str, bytes]:
        """Get all stored images."""
        return self._image_store.copy()
    
    def create_zip(self, filenames: List[str], zip_name: str) -> io.BytesIO:
        """
        Create a ZIP file containing specified images.
        
        Args:
            filenames: List of filenames to include
            zip_name: Name for the ZIP file (without extension)
            
        Returns:
            BytesIO buffer containing the ZIP file
        """
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for filename in filenames:
                if filename in self._image_store:
                    zf.writestr(filename, self._image_store[filename])
        
        zip_buffer.seek(0)
        return zip_buffer
    
    def cleanup(self):
        """Clean up all temporary resources."""
        # Clean temp directory
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
        
        # Clear image store
        self._image_store.clear()
    
    def reset(self):
        """Reset handler for new session."""
        self.cleanup()
