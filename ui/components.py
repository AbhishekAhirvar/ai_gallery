"""
Reusable UI components for VowScan.
"""

import streamlit as st

def apply_custom_css():
    """Apply custom CSS for the application."""
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 1rem 0 2rem 0;
        }
        .main-header h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 700;
        }
        .stExpander {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .summary-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
        }
        .upload-section {
            background: #f8f9fa;
            padding: 2rem;
            border-radius: 15px;
            border: 2px dashed #667eea;
            margin-bottom: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Render the main application header."""
    st.markdown("""
    <div class="main-header">
        <h1>💒 VowScan</h1>
        <p style="font-size: 1.2rem; color: #666;">Wedding Photo Gallery - Organized by Face</p>
    </div>
    """, unsafe_allow_html=True)
