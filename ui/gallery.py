from streamlit_clickable_images import clickable_images
from utils.image_utils import encode_image_to_base64
import streamlit as st
import streamlit.components.v1 as components
import os
import cv2
from typing import List, Dict, Optional

from services.face_clusterer import PersonCluster
from services.file_handler import FileHandler
from services.face_detector import FaceEmbedding

@st.cache_data(show_spinner=False)
def get_cached_thumbnail(path: str) -> Optional[str]:
    """Cache the base64 encoding of thumbnails to prevent disk I/O on every rerun."""
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            import base64
            b64 = base64.b64encode(f.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{b64}"
    return None

def render_gallery(
    clusters: List[PersonCluster], 
    file_handler: FileHandler, 
    unidentified_filenames: List[str],
    unclustered_embeddings: List[FaceEmbedding]
):
    """
    Render gallery with clickable images.
    """
    # --- PERSISTENT DIALOG LOGIC ---
    
    # Global Gallery CSS (Hover effects)
    st.markdown("""
    <style>
    img {
        opacity: 1 !important;
        animation: none !important;
        transition: transform 0.2s ease !important;
        filter: none !important;
    }
    img:hover {
        transform: scale(1.05);
        opacity: 1 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.session_state.get('active_album_cluster'):
        # Ensure we are viewing the correct cluster object (sync with list)
        label = st.session_state.active_album_cluster.label
        current = next((c for c in clusters if c.label == label), st.session_state.active_album_cluster)
        st.session_state.active_album_cluster = current
        view_album_dialog()
        
    if st.session_state.get('active_unidentified'):
        view_unidentified_album(unidentified_filenames, unclustered_embeddings, file_handler)


    # Top stats
    total_people = len(clusters)
    total_photos = len(file_handler.get_all_images())
    
    st.markdown(f"### 📸 **{total_people} People** found in {total_photos} photos")
    
    # --- Reset Action ---
    if st.button("🔄 Start Over", type="secondary"):
        st.session_state.clear()
        st.rerun()
    st.markdown("---")
    
    # --- People Grid ---
    if not clusters and not unidentified_filenames:
        st.info("No people found.")
        return

    # Prepare data for clickable images
    images_urls = []
    labels = []
    
    for cluster in clusters:
        thumb_path = cluster.thumbnail_path
        img_src = "https://placehold.co/200x200?text=?"
        
        cached_src = get_cached_thumbnail(thumb_path)
        if cached_src:
            img_src = cached_src
        
        images_urls.append(img_src)
        labels.append(f"{cluster.label} ({len(cluster.filenames)})")

    # Render grid using clickable_images
    if images_urls:
        clicked_index = clickable_images(
            images_urls, 
            titles=labels,
            div_style={
                "display": "grid",
                "grid-template-columns": "repeat(auto-fill, minmax(150px, 1fr))",
                "grid-gap": "20px",
                "justify-items": "center"
            },
            img_style={
                "width": "150px",
                "height": "150px",
                "object-fit": "cover",
                "border-radius": "50%",
                "border": "3px solid #ff4b4b",
                "cursor": "pointer"
            }
        )
        
        if clicked_index > -1:
            st.session_state.active_album_cluster = clusters[clicked_index]
            st.session_state.album_selected_image = None
            st.session_state.active_unidentified = False
            st.rerun()
        
        if clicked_index > -1:
            st.session_state.active_album_cluster = clusters[clicked_index]
            st.session_state.album_selected_image = None
            st.session_state.active_unidentified = False
            st.rerun()
            
    # Unidentified Section
    if unidentified_filenames:
        st.markdown("---")
        if st.button(f"📂 View Unidentified Photos ({len(unidentified_filenames)})"):
            st.session_state.active_unidentified = True
            st.session_state.active_album_cluster = None
            st.rerun()


@st.dialog("Person Album", width="large")
def view_album_dialog():
    """
    Stateful dialog that can navigate between people and photos.
    Requires: st.session_state.active_album_cluster
    """
    if 'active_album_cluster' not in st.session_state or not st.session_state.active_album_cluster:
        st.error("No album selected.")
        return

    # Active cluster
    cluster = st.session_state.active_album_cluster
    file_handler = st.session_state.file_handler
    all_clusters = st.session_state.clusters
    
    # State: Selected Image (for detail view)
    if 'album_selected_image' not in st.session_state:
        st.session_state.album_selected_image = None
        
    # --- NAVIGATION: Detail View vs Grid View ---
    
    # 1. DETAIL VIEW
    if st.session_state.album_selected_image:
        filename = st.session_state.album_selected_image
        
        # Back Button
        if st.button("← Back to Album", key="back_to_grid"):
            st.session_state.album_selected_image = None
            st.rerun()
            
        # Navigation & Image Display
        try:
            current_idx = cluster.filenames.index(filename)
            total_imgs = len(cluster.filenames)
        except ValueError:
            current_idx = 0
            total_imgs = 0
            
        c_prev, c_main, c_next = st.columns([1, 12, 1], vertical_alignment="center")
        
        with c_prev:
            if st.button("◀", key="nav_prev", use_container_width=True):
                prev_idx = (current_idx - 1) % total_imgs
                st.session_state.album_selected_image = cluster.filenames[prev_idx]
                st.rerun()
                
        with c_next:
            if st.button("▶", key="nav_next", use_container_width=True):
                next_idx = (current_idx + 1) % total_imgs
                st.session_state.album_selected_image = cluster.filenames[next_idx]
                st.rerun()

        # Keyboard Navigation (JS Injection)
        components.html("""
        <script>
        document.addEventListener('keydown', function(e) {
            if (e.key === 'ArrowLeft') {
                const buttons = Array.from(window.parent.document.querySelectorAll('button'));
                const prevBtn = buttons.find(el => el.innerText === "◀");
                if (prevBtn) prevBtn.click();
            }
            if (e.key === 'ArrowRight') {
                const buttons = Array.from(window.parent.document.querySelectorAll('button'));
                const nextBtn = buttons.find(el => el.innerText === "▶");
                if (nextBtn) nextBtn.click();
            }
        });
        </script>
        """, height=0, width=0)
        
        with c_main:
            # Full Image (Medium Optimized)
            img_data = file_handler.get_medium_image(filename)
            if not img_data:
                img_data = file_handler.get_image(filename) # Fallback to original
                
            if img_data:
                st.image(img_data, width="stretch")
                
                # Download Option
                original_data = file_handler.get_image(filename)
                if original_data:
                    st.download_button(
                        label="⬇ Download Original",
                        data=original_data,
                        file_name=filename,
                        mime="image/jpeg",
                        use_container_width=True
                    )
            
        st.markdown("### 👥 People in this photo")
        
        # Find who else is in this photo
        people_in_photo = []
        for c in all_clusters:
            if filename in c.filenames:
                people_in_photo.append(c)
        
        if not people_in_photo:
            st.caption("No other identified people found.")
        else:
            # Render thumbnails for people
            people_urls = []
            people_titles = []
            
            for p in people_in_photo:
                 # Highlight current person
                is_current = (p.label == cluster.label)
                # clickable_images doesn't easy support per-image styling tokens in list. 
                # We'll rely on the tooltip or order.
                
                thumb_path = p.thumbnail_path
                img_src = "https://placehold.co/100x100?text=?"
                
                cached_src = get_cached_thumbnail(thumb_path)
                if cached_src:
                    img_src = cached_src
                
                people_urls.append(img_src)
                people_titles.append(f"{p.label}{' (Current)' if is_current else ''}")
                
            clicked_person = clickable_images(
                people_urls,
                titles=people_titles,
                div_style={"display": "flex", "flex-wrap": "wrap", "gap": "10px"},
                img_style={
                    "width": "80px", 
                    "height": "80px", 
                    "object-fit": "cover", 
                    "border-radius": "50%", 
                    "border": "2px solid white", 
                    "cursor": "pointer"
                }
            )
            
            if clicked_person > -1:
                target_person = people_in_photo[clicked_person]
                if target_person.label != cluster.label:
                    st.session_state.active_album_cluster = target_person
                    st.session_state.album_selected_image = None
                    st.rerun()
                    
    # 2. GRID VIEW
    else:
        # Check for CSS polish injection
        st.markdown("""
        <style>
        img {
            transition: transform 0.2s ease;
        }
        img:hover {
            transform: scale(1.05);
            opacity: 1 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # Header with Editable Name & Close
        c1, c2 = st.columns([8, 2])
        with c1:
            # Editable Name
            new_name = st.text_input("Name", value=cluster.label, key=f"edit_name_{cluster.label}")
            if new_name != cluster.label:
                # Update label
                cluster.label = new_name
                st.rerun()
                
        if c2.button("✖ Close"):
            st.session_state.active_album_cluster = None
            st.rerun()
        
        # Download
        safe_name = cluster.label.replace(" ", "_")
        zip_buf = file_handler.create_zip(cluster.filenames, safe_name)
        st.download_button(
            "📥 Download All Photos",
            data=zip_buf,
            file_name=f"{safe_name}.zip",
            mime="application/zip",
            use_container_width=True
        )
        
        st.divider()
        
        filenames = cluster.filenames
        
        # Prepare clickable images for the album
        album_urls = []
        album_titles = []
        
        for fname in filenames:
            # Try to get existing thumbnail (Fastest)
            img_bytes = file_handler.get_thumbnail_image(fname)
            
            # Fallback to creating on fly if missing (e.g. old uploads)
            if not img_bytes:
                 img_bytes = file_handler.get_image(fname)
                 # If we have to resize here, it's slow. But new flow ensures thumbs exist.
                 if img_bytes:
                     # Resize on fly fallback
                    from utils.image_utils import decode_image
                    img_cv = decode_image(img_bytes)
                    if img_cv is not None:
                        img_small = cv2.resize(img_cv, (300, 300))
                        _, buffer = cv2.imencode('.jpg', img_small)
                        img_bytes = buffer.tobytes()

            if img_bytes:
                import base64
                b64 = base64.b64encode(img_bytes).decode('utf-8')
                img_src = f"data:image/jpeg;base64,{b64}"
                album_urls.append(img_src)
                album_titles.append(fname)
            else:
                 album_urls.append("https://placehold.co/200x200?text=Missing")
                 album_titles.append("Missing")
        
        if album_urls:
            clicked_photo = clickable_images(
                album_urls,
                titles=album_titles,
                div_style={
                    "display": "grid",
                    "grid-template-columns": "repeat(auto-fill, minmax(150px, 1fr))",
                    "grid-gap": "10px"
                },
                img_style={
                    "width": "100%", 
                    "height": "100%", # Fill grid
                    "aspect-ratio": "1 / 1", 
                    "object-fit": "cover",
                    "border-radius": "5px", 
                    "cursor": "pointer"
                }
            )
            
            if clicked_photo > -1:
                st.session_state.album_selected_image = filenames[clicked_photo]
                st.rerun()
            



@st.dialog("Unidentified Photos", width="large")
def view_unidentified_album(filenames: List[str], unclustered_embeddings: List[FaceEmbedding], file_handler: FileHandler):
    """
    Modal dialog for unidentified photos with overlay boxes.
    """
    c1, c2 = st.columns([8, 2])
    c1.header("Unidentified Photos")
    if c2.button("✖ Close", key="close_unidentified"):
        st.session_state.active_unidentified = False
        st.rerun()
    
    st.info("🟥 Red boxes indicate faces that were detected but not grouped (low quality or not enough matches).")
    
    zip_buf = file_handler.create_zip(filenames, "unidentified")
    st.download_button(
        "📥 Download All",
        data=zip_buf,
        file_name="unidentified.zip",
        mime="application/zip",
        use_container_width=True
    )
    
    st.divider()
    
    col_count = 3
    rows = [filenames[i:i + col_count] for i in range(0, len(filenames), col_count)]
    
    for row in rows:
        cols = st.columns(col_count)
        for i, filename in enumerate(row):
            with cols[i]:
                # Get raw byte data
                image_bytes = file_handler.get_image_bytes(filename) # need bytes for opencv/pil
                if not image_bytes:
                    continue
                    
                # Decode for drawing
                # We need a robust way to draw. Use opencv (numpy)
                from utils.image_utils import decode_image
                img_bgr = decode_image(image_bytes)
                
                if img_bgr is not None:
                    # Find embeddings for this file
                    faces_in_file = [fe for fe in unclustered_embeddings if fe.filename == filename]
                    
                    # Draw boxes
                    for fe in faces_in_file:
                        if fe.facial_area:
                            x = fe.facial_area['x']
                            y = fe.facial_area['y']
                            w = fe.facial_area['w']
                            h = fe.facial_area['h']
                            # Draw rectangle (Red in BGR is 0,0,255)
                            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 0, 255), 4)
                            
                    # Convert BGR to RGB for Streamlit
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, width='stretch')
                else:
                    st.error("Error loading image")

