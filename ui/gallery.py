from streamlit_clickable_images import clickable_images
from utils.image_utils import encode_image_to_base64
import streamlit as st
import os
import cv2
from typing import List, Dict, Optional

from services.face_clusterer import PersonCluster
from services.file_handler import FileHandler
from services.face_detector import FaceEmbedding

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
    # We'll use CSS to style the captions (names) overlay or below?
    # clickable_images allows simple img tags. Styling caption is tricky inside the component.
    # Actually, clickable_images just renders images.
    # To show names, we might want to stick to the grid OR use the 'titles' feature if available?
    # The standard library is simple.
    # Workaround: Use div with image and text inside the HTML passed to clickable_images.
    
    # Prepare data for clickable images
    images_urls = []
    labels = []
    
    for cluster in clusters:
        thumb_path = cluster.thumbnail_path
        img_src = "https://placehold.co/200x200?text=?"
        
        if thumb_path and os.path.exists(thumb_path):
             # Read file directly
            with open(thumb_path, "rb") as f:
                import base64
                b64 = base64.b64encode(f.read()).decode('utf-8')
                img_src = f"data:image/jpeg;base64,{b64}"
        
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
            
        # Full Image
        img_data = file_handler.get_image(filename)
        if img_data:
            st.image(img_data, use_container_width=True)
            
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
                if thumb_path and os.path.exists(thumb_path):
                    with open(thumb_path, "rb") as f:
                        import base64
                        b64 = base64.b64encode(f.read()).decode('utf-8')
                        img_src = f"data:image/jpeg;base64,{b64}"
                
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
            img_bytes = file_handler.get_image(fname)
            if img_bytes:
                from utils.image_utils import decode_image
                img_cv = decode_image(img_bytes)
                if img_cv is not None:
                     # Resize for grid
                    img_small = cv2.resize(img_cv, (200, 200)) # Simple resize
                    
                    _, buffer = cv2.imencode('.jpg', img_small)
                    import base64
                    b64 = base64.b64encode(buffer).decode('utf-8')
                    img_src = f"data:image/jpeg;base64,{b64}"
                    album_urls.append(img_src)
                    album_titles.append(fname)
                else:
                    album_urls.append("https://placehold.co/200x200?text=Error")
                    album_titles.append("Error")
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
                    "height": "auto", 
                    "border-radius": "5px", 
                    "cursor": "pointer"
                }
            )
            
            if clicked_photo > -1:
                st.session_state.album_selected_image = filenames[clicked_photo]
                st.rerun()
            
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

