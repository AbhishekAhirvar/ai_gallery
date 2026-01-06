"""
Dashboard view for VowScan - displays analytics and stats
"""
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import json
from datetime import datetime


def render_dashboard():
    """Render the analytics dashboard with real data from session state."""
    
    # Calculate real statistics from session state
    stats = calculate_dashboard_stats()
    
    # Load the dashboard HTML/CSS/JS files
    dashboard_dir = Path(__file__).parent.parent / "dashboard"
    
    with open(dashboard_dir / "dashboard.html", "r") as f:
        html_content = f.read()
    
    with open(dashboard_dir / "dashboard.css", "r") as f:
        css_content = f.read()
    
    with open(dashboard_dir / "dashboard.js", "r") as f:
        js_content = f.read()
    
    # Inject real data into JavaScript
    data_injection = f"""
    <script>
        // Override dummy data with real data from Streamlit session
        const dashboardData = {{
            stats: {{
                totalImages: {stats['total_images']},
                facesDetected: {stats['faces_detected']},
                personClusters: {stats['person_clusters']},
                processingSpeed: {stats['processing_speed']}
            }},
            performanceData: {{
                labels: {json.dumps(stats['performance_labels'])},
                datasets: [
                    {{
                        label: 'Images Processed',
                        data: {json.dumps(stats['images_processed_data'])},
                        borderColor: '#7C3AED',
                        backgroundColor: 'rgba(124, 58, 237, 0.1)',
                        tension: 0.4,
                        fill: true
                    }},
                    {{
                        label: 'Faces Detected',
                        data: {json.dumps(stats['faces_detected_data'])},
                        borderColor: '#3B82F6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4,
                        fill: true
                    }}
                ]
            }},
            distributionData: {{
                labels: {json.dumps(stats['distribution_labels'])},
                datasets: [{{
                    label: 'Image Distribution',
                    data: {json.dumps(stats['distribution_data'])},
                    backgroundColor: [
                        'rgba(124, 58, 237, 0.8)',
                        'rgba(59, 130, 246, 0.8)',
                        'rgba(16, 185, 129, 0.8)',
                        'rgba(245, 158, 11, 0.8)',
                        'rgba(239, 68, 68, 0.8)'
                    ],
                    borderWidth: 0
                }}]
            }},
            recentClusters: {json.dumps(stats['recent_clusters'])},
            recentActivity: {json.dumps(stats['recent_activity'])}
        }};
    </script>
    """
    
    # Combine HTML with inline CSS and JS
    full_html = html_content.replace(
        '<link rel="stylesheet" href="dashboard.css">',
        f'<style>{css_content}</style>'
    ).replace(
        '<script src="dashboard.js"></script>',
        f'{data_injection}<script>{js_content}</script>'
    )
    
    # Render in Streamlit
    components.html(full_html, height=2000, scrolling=True)


def calculate_dashboard_stats():
    """Calculate dashboard statistics from session state."""
    
    # Initialize default values
    total_images = 0
    faces_detected = 0
    person_clusters = 0
    processing_speed = 0
    
    # Get data from session state if available
    if hasattr(st.session_state, 'file_handler') and st.session_state.file_handler:
        # Total images
        try:
            total_images = len(st.session_state.file_handler.image_files) if hasattr(
                st.session_state.file_handler, 'image_files'
            ) else 0
        except:
            total_images = 0
    
    if hasattr(st.session_state, 'clusters') and st.session_state.clusters:
        # Person clusters
        person_clusters = len(st.session_state.clusters)
        
        # Faces detected (sum of all faces in clusters)
        faces_detected = sum(len(cluster.embeddings) for cluster in st.session_state.clusters)
    
    # Add unclustered faces
    if hasattr(st.session_state, 'unclustered_embeddings') and st.session_state.unclustered_embeddings:
        faces_detected += len(st.session_state.unclustered_embeddings)
    
    # Processing speed from stats if available
    if hasattr(st.session_state, 'processing_stats') and st.session_state.processing_stats:
        processing_speed = st.session_state.processing_stats.get('images_per_second', 0)
    
    # Calculate face distribution
    distribution_labels, distribution_data = calculate_face_distribution()
    
    # Recent clusters for gallery preview
    recent_clusters = get_recent_clusters()
    
    # Recent activity timeline
    recent_activity = get_recent_activity()
    
    # Performance data (mock for now, can be enhanced with real tracking)
    performance_labels = ['Session']
    images_processed_data = [total_images]
    faces_detected_data = [faces_detected]
    
    return {
        'total_images': total_images,
        'faces_detected': faces_detected,
        'person_clusters': person_clusters,
        'processing_speed': int(processing_speed),
        'distribution_labels': distribution_labels,
        'distribution_data': distribution_data,
        'performance_labels': performance_labels,
        'images_processed_data': images_processed_data,
        'faces_detected_data': faces_detected_data,
        'recent_clusters': recent_clusters,
        'recent_activity': recent_activity
    }


def calculate_face_distribution():
    """Calculate how many images have 0, 1, 2-3, 4-6, 7+ faces."""
    distribution = {
        'single': 0,    # 1 face
        'few': 0,       # 2-3 faces
        'several': 0,   # 4-6 faces
        'many': 0,      # 7+ faces
        'none': 0       # 0 faces
    }
    
    if not hasattr(st.session_state, 'clusters') or not st.session_state.clusters:
        return ['No Data'], [1]
    
    # Count faces per image
    image_face_counts = {}
    
    for cluster in st.session_state.clusters:
        for embedding in cluster.embeddings:
            filename = embedding.filename
            if filename not in image_face_counts:
                image_face_counts[filename] = 0
            image_face_counts[filename] += 1
    
    # Add images with no faces
    if hasattr(st.session_state, 'no_face_filenames'):
        for filename in st.session_state.no_face_filenames:
            image_face_counts[filename] = 0
    
    # Categorize
    for filename, count in image_face_counts.items():
        if count == 0:
            distribution['none'] += 1
        elif count == 1:
            distribution['single'] += 1
        elif 2 <= count <= 3:
            distribution['few'] += 1
        elif 4 <= count <= 6:
            distribution['several'] += 1
        else:
            distribution['many'] += 1
    
    labels = ['Single Face', '2-3 Faces', '4-6 Faces', '7+ Faces', 'No Face']
    data = [
        distribution['single'],
        distribution['few'],
        distribution['several'],
        distribution['many'],
        distribution['none']
    ]
    
    return labels, data


def get_recent_clusters():
    """Get recent clusters for gallery preview."""
    if not hasattr(st.session_state, 'clusters') or not st.session_state.clusters:
        return []
    
    colors = ['#7C3AED', '#3B82F6', '#10B981', '#F59E0B', '#EC4899', '#06B6D4', '#8B5CF6', '#EF4444']
    
    recent = []
    for i, cluster in enumerate(st.session_state.clusters[:8]):  # First 8 clusters
        recent.append({
            'id': i + 1,
            'faces': len(cluster.embeddings),
            'color': colors[i % len(colors)]
        })
    
    return recent


def get_recent_activity():
    """Get recent activity for timeline."""
    activity = []
    
    # Add processing completion if data exists
    if hasattr(st.session_state, 'processed') and st.session_state.processed:
        total_images = 0
        faces_detected = 0
        
        if hasattr(st.session_state, 'clusters'):
            faces_detected = sum(len(c.embeddings) for c in st.session_state.clusters)
        if hasattr(st.session_state, 'file_handler'):
            try:
                total_images = len(st.session_state.file_handler.image_files) if hasattr(
                    st.session_state.file_handler, 'image_files'
                ) else 0
            except:
                pass
        
        if total_images > 0:
            activity.append({
                'icon': '📤',
                'title': f'Processed {total_images} images',
                'time': 'Just now',
                'type': 'upload'
            })
        
        if faces_detected > 0:
            activity.append({
                'icon': '✅',
                'title': f'Detected {faces_detected} faces',
                'time': 'Just now',
                'type': 'success'
            })
        
        if hasattr(st.session_state, 'clusters') and st.session_state.clusters:
            activity.append({
                'icon': '🔄',
                'title': f'Created {len(st.session_state.clusters)} person clusters',
                'time': 'Just now',
                'type': 'process'
            })
    
    # Add default message if no activity
    if not activity:
        activity.append({
            'icon': 'ℹ️',
            'title': 'No processing activity yet',
            'time': 'Waiting for images',
            'type': 'info'
        })
    
    return activity
