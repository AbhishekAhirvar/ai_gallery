"""
File Handling Service for VowScan.
Manages file uploads, ZIP extraction, and temporary storage.
"""

import io
import os
import zipfile
import tempfile
import shutil
import pathlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from config import SUPPORTED_IMAGE_EXTENSIONS
from utils.image_utils import decode_image, resize_image, encode_image


@dataclass
class ImageFile:
    """Represents an uploaded image file."""
    filename: str
    path: str  # Path to the file on disk
    
    def read(self) -> bytes:
        """Read file content from disk."""
        with open(self.path, 'rb') as f:
            return f.read()


class FileHandler:
    """
    Service for handling file uploads and ZIP operations.
    Manages temporary storage and cleanup.
    """
    
    def __init__(self):
        self._temp_dir: Optional[str] = None
        # Stores map of filename -> local_path
        self._image_store: Dict[str, str] = {} 
        # Stores map of filename -> thumb_bytes (keep thumbs in RAM for speed, they are small)
        self._thumb_store: Dict[str, bytes] = {} 

    def store_derived_images(self, filename: str, medium_data: Optional[bytes], thumb_data: bytes):
        """Store resizing versions of the image. medium_data is ignored now."""
        # Lazy init for hot-reload safety
        if not hasattr(self, '_thumb_store'): self._thumb_store = {}
        
        if thumb_data:
            self._thumb_store[filename] = thumb_data

    def get_medium_image(self, filename: str) -> Optional[bytes]:
        """Get 1080p version. Generates on fly if missing to speed up transfer."""
        print(f"DEBUG: Requesting medium image for {filename}")
        
        # Since we are moving to disk, we can generate this on demand from the disk file
        # We won't cache medium images in RAM to save space, or maybe save to disk?
        # For now, let's generate from original on disk to avoid RAM usage.
            
        # Lazy generate from Original
        original_bytes = self.get_image(filename)
        if original_bytes:
            print("DEBUG: Generating medium from original...")
            try:
                # We need to resize for performance
                img = decode_image(original_bytes)
                if img is not None:
                    img_medium = resize_image(img, max_size=1920)
                    medium_data = encode_image(img_medium)
                    # We return bytes but don't cache in RAM to huge extent
                    # potentially could cache to disk if needed, but for now just return
                    return medium_data
            except Exception as e:
                print(f"DEBUG: Generation failed: {e}")
                pass
        else:
            print("DEBUG: Original not found!")
        
        return original_bytes

    def get_thumbnail_image(self, filename: str) -> Optional[bytes]:
        """Get 300px version. Generates on fly if missing."""
        if not hasattr(self, '_thumb_store'): self._thumb_store = {}
        
        # Try cache
        if filename in self._thumb_store:
            return self._thumb_store[filename]
            
        # Lazy generate
        original_bytes = self.get_image(filename)
        if original_bytes:
            try:
                img = decode_image(original_bytes)
                if img is not None:
                    img_thumb = resize_image(img, max_size=300)
                    thumb_data = encode_image(img_thumb)
                    self._thumb_store[filename] = thumb_data
                    return thumb_data
            except Exception:
                pass
                
        return None
    
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
        Saves files to disk immediately.
        
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
                
                # Write to disk
                save_path = os.path.join(self.temp_dir, filename)
                # Handle duplicates if needed, but for now overwrite or unique
                # Make unique
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(save_path):
                    save_path = os.path.join(self.temp_dir, f"{base}_{counter}{ext}")
                    counter += 1
                    
                with open(save_path, 'wb') as f:
                    f.write(file_bytes)
                
                final_filename = os.path.basename(save_path)
                images.append(ImageFile(filename=final_filename, path=save_path))
                
                # Update store
                self._image_store[final_filename] = save_path
        
        return images
    
    def _extract_from_zip(self, zip_file) -> List[ImageFile]:
        """Extract images from a ZIP file to disk."""
        images = []
        
        try:
            zip_bytes = zip_file.read()
            zip_file.seek(0)
            
            with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
                for name in zf.namelist():
                    # Skip macOS metadata and directories
                    if name.startswith('__MACOSX') or name.endswith('/'):
                        continue
                    
                    # Check if it's an image
                    if name.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                        # Use basename to avoid path issues
                        filename = os.path.basename(name)
                        if not filename:
                            continue
                            
                        # Unique path
                        save_path = os.path.join(self.temp_dir, filename)
                        base, ext = os.path.splitext(filename)
                        counter = 1
                        while os.path.exists(save_path):
                            save_path = os.path.join(self.temp_dir, f"{base}_{counter}{ext}")
                            counter += 1
                        
                        # Extract
                        with open(save_path, 'wb') as f:
                            f.write(zf.read(name))
                            
                        final_filename = os.path.basename(save_path)
                        images.append(ImageFile(filename=final_filename, path=save_path))
                        
                        # Update store
                        self._image_store[final_filename] = save_path
                        
        except Exception as e:
            print(f"Error extracting zip: {e}")
            pass
        
        return images
    
    def store_image(self, filename: str, data: bytes):
        """
        Store image data for later retrieval.
        Writes to disk if not already there.
        """
        # If we have a path already, we might assume it's done
        if filename in self._image_store:
            return

        # Write to disk
        save_path = os.path.join(self.temp_dir, filename)
        with open(save_path, 'wb') as f:
            f.write(data)
        self._image_store[filename] = save_path
    
    def get_image(self, filename: str) -> Optional[bytes]:
        """Retrieve image bytes from disk."""
        path = self._image_store.get(filename)
        if path and os.path.exists(path):
            with open(path, 'rb') as f:
                return f.read()
        return None

    def get_image_bytes(self, filename: str) -> Optional[bytes]:
        """Alias for get_image."""
        return self.get_image(filename)
    
    def get_all_images(self) -> Dict[str, bytes]:
        """
        Get all stored images.
        WARNING: This loads EVERYTHING into RAM. Use with caution.
        Better to iterate if possible, but keeping signature for compatibility.
        """
        # Ideally we shouldn't use this method for 10k images
        all_imgs = {}
        for fname, path in self._image_store.items():
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    all_imgs[fname] = f.read()
        return all_imgs
    
    def create_zip(self, filenames: List[str], zip_name: str) -> io.BytesIO:
        """
        Create a ZIP file containing specified images.
        """
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for filename in filenames:
                path = self._image_store.get(filename)
                if path and os.path.exists(path):
                    zf.write(path, filename)
        
        zip_buffer.seek(0)
        return zip_buffer
    
    def cleanup(self):
        """Clean up all temporary resources."""
        # Clean temp directory
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Error cleaning up: {e}")
            self._temp_dir = None
        
        # Clear image store
        self._image_store.clear()
        if hasattr(self, '_thumb_store'):
            self._thumb_store.clear()
        if hasattr(self, '_medium_store'):
            self._medium_store.clear()
    
    def reset(self):
        """Reset handler for new session."""
        self.cleanup()
