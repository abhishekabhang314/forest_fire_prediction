import streamlit as st
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import io
import os

# === CONFIG ===
MODEL_PATH = "models/unet_alberta.pth"
OUTPUT_PATH = "outputs/streamlit_result.png"

# === NATURE-INSPIRED COLOR THEMES ===
LIGHT_THEME = {
    "primary": "#2E8B57",  # Sea Green
    "secondary": "#4682B4",  # Steel Blue
    "accent": "#32CD32",  # Lime Green
    "background": "#F8FDF8",  # Mint Cream
    "card_bg": "#FFFFFF",
    "text_primary": "#2F4F4F",  # Dark Slate Gray
    "text_secondary": "#708090",  # Slate Gray
    "success": "#228B22",  # Forest Green
    "warning": "#FF8C00",  # Dark Orange
    "danger": "#DC143C",  # Crimson
    "border": "#E0EEE0"  # Honeydew
}

DARK_THEME = {
    "primary": "#20B2AA",  # Light Sea Green
    "secondary": "#5F9EA0",  # Cadet Blue
    "accent": "#00FA9A",  # Medium Spring Green
    "background": "#0A1F1C",  # Dark Forest Green
    "card_bg": "#1A2F2C",
    "text_primary": "#E0F2F1",  # Light Cyan
    "text_secondary": "#B2DFDB",  # Pale Turquoise
    "success": "#90EE90",  # Light Green
    "warning": "#FFA500",  # Orange
    "danger": "#FF6B6B",  # Light Coral
    "border": "#2F4F4F"  # Dark Slate Gray
}

# === THEME TOGGLE ===
def get_theme():
    """Get current theme from session state or default to light"""
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'
    return st.session_state.theme

def toggle_theme():
    """Toggle between light and dark themes"""
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

# === Load Model (cache to avoid reloading every time) ===
@st.cache_resource
def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

def preprocess_image(image: Image.Image):
    image = image.resize((256, 256))
    img_array = np.array(image) / 255.0
    img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return img_tensor

def predict_fire(image: Image.Image, model):
    img_tensor = preprocess_image(image)
    with torch.no_grad():
        preds = model(img_tensor)
        mask = torch.sigmoid(preds)[0, 0].numpy()
        mask = (mask > 0.5).astype(np.uint8)
    return mask

def overlay_mask(image: Image.Image, mask: np.ndarray):
    image = image.convert("RGBA").resize((256, 256))
    overlay = Image.new("RGBA", image.size, (255, 0, 0, 0))
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    # Use nature-inspired alert color (Forest Fire Orange)
    alert_overlay = Image.new("RGBA", image.size, (255, 140, 0, 150))  # Dark Orange with transparency
    overlay = Image.composite(alert_overlay, overlay, mask_img)
    return Image.alpha_composite(image, overlay)

def calculate_fire_coverage(mask: np.ndarray):
    total_pixels = mask.size
    fire_pixels = np.sum(mask)
    coverage_percentage = (fire_pixels / total_pixels) * 100
    return coverage_percentage

# === STREAMLIT APP WITH NATURE THEME ===
st.set_page_config(
    page_title="Wildfire Detection", 
    layout="wide"
)

# Get current theme
current_theme = get_theme()
COLORS = LIGHT_THEME if current_theme == 'light' else DARK_THEME

# === Custom CSS for Nature Theme ===
st.markdown(f"""
<style>
    .main {{
        background-color: {COLORS['background']};
    }}
    .stApp {{
        background: linear-gradient(135deg, {COLORS['background']} 0%, {COLORS['card_bg']} 100%);
    }}
    .stButton>button {{
        background-color: {COLORS['primary']};
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: {COLORS['secondary']};
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(46, 139, 87, 0.3);
    }}
    .uploadedFile {{
        border: 2px dashed {COLORS['secondary']};
        border-radius: 15px;
        padding: 20px;
        background-color: {COLORS['card_bg']};
    }}
    .prediction-card {{
        background-color: {COLORS['card_bg']};
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid {COLORS['primary']};
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid {COLORS['border']};
    }}
    .header-section {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }}
    .feature-icon {{
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }}
    .theme-toggle {{
        position: absolute;
        top: 1rem;
        right: 1rem;
        background: {COLORS['card_bg']};
        border: 1px solid {COLORS['border']};
        border-radius: 20px;
        padding: 0.5rem 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }}
    .theme-toggle:hover {{
        background: {COLORS['primary']};
        color: white;
    }}
    .stat-card {{
        background: {COLORS['card_bg']};
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid {COLORS['border']};
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }}
    .risk-indicator {{
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
    }}
</style>
""", unsafe_allow_html=True)

# === Header Section with Theme Toggle ===
col_header, col_toggle = st.columns([4, 1])
with col_header:
    st.markdown(f"""
    <div class="header-section">
        <h1 style="margin:0; font-size: 2.5rem;">🌲 Forest Fire Detection System</h1>
        <p style="margin:0; font-size: 1.2rem; opacity: 0.9;">AI-powered wildfire detection using nature-inspired design</p>
    </div>
    """, unsafe_allow_html=True)

with col_toggle:
    theme_icon = "🌙" if current_theme == 'light' else "☀️"
    theme_label = "Dark Mode" if current_theme == 'light' else "Light Mode"
    if st.button(f"{theme_icon} {theme_label}", use_container_width=True):
        toggle_theme()
        st.rerun()

# === Introduction Section ===
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div style='text-align: center; padding: 1rem; background: {COLORS['card_bg']}; border-radius: 10px; border: 1px solid {COLORS['border']}'>
        <div class="feature-icon">🛰️</div>
        <h4 style='color: {COLORS["primary"]};'>Satellite Analysis</h4>
        <p style='color: {COLORS["text_secondary"]};'>Real-time processing of satellite imagery</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style='text-align: center; padding: 1rem; background: {COLORS['card_bg']}; border-radius: 10px; border: 1px solid {COLORS['border']}'>
        <div class="feature-icon">🤖</div>
        <h4 style='color: {COLORS["primary"]};'>AI Powered</h4>
        <p style='color: {COLORS["text_secondary"]};'>Advanced deep learning algorithms</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style='text-align: center; padding: 1rem; background: {COLORS['card_bg']}; border-radius: 10px; border: 1px solid {COLORS['border']}'>
        <div class="feature-icon">🌍</div>
        <h4 style='color: {COLORS["primary"]};'>Nature Theme</h4>
        <p style='color: {COLORS["text_secondary"]};'>Eco-friendly green & blue colors</p>
    </div>
    """, unsafe_allow_html=True)

# === Main Content ===
st.markdown("---")

# Create two columns for layout
left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown(f"""
    <div class='stat-card'>
        <h3 style='color: {COLORS["text_primary"]}; margin-top: 0;'>📤 Upload Satellite Image</h3>
        <p style='color: {COLORS["text_secondary"]};'>Upload a satellite image to detect potential fire zones highlighted in alert orange.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a satellite image...", 
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

with right_col:
    if uploaded_file is not None:
        st.markdown(f"""
        <div class='prediction-card'>
            <h3 style='color: {COLORS["text_primary"]}; margin-top: 0;'>📊 Analysis Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Display uploaded image
        st.markdown("<div class='uploadedFile'>", unsafe_allow_html=True)
        st.image(uploaded_file, caption="🌍 Original Satellite Image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Process image
        image = Image.open(uploaded_file).convert("RGB")
        model = load_model()
        
        with st.spinner("🌿 Analyzing vegetation and fire risks... Please wait!"):
            mask = predict_fire(image, model)
            overlay = overlay_mask(image, mask)
            overlay.save(OUTPUT_PATH)
            
            # Calculate statistics
            fire_coverage = calculate_fire_coverage(mask)
            
        # Display results
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            st.image(overlay, caption="🔥 Fire Detection Overlay", use_container_width=True)
        
        with col_result2:
            # Statistics card
            st.markdown(f"""
            <div class='stat-card'>
                <h4 style='color: {COLORS["text_primary"]}; margin-top: 0;'>📈 Detection Statistics</h4>
                <div style='font-size: 2rem; color: {COLORS["warning"]}; font-weight: bold; text-align: center;'>
                    {fire_coverage:.2f}%
                </div>
                <p style='text-align: center; color: {COLORS["text_secondary"]};'>Fire Area Coverage</p>
                <div style='background-color: {COLORS["border"]}; border-radius: 10px; height: 20px; margin: 1rem 0;'>
                    <div style='background: linear-gradient(90deg, {COLORS["success"]}, {COLORS["warning"]}, {COLORS["danger"]}); 
                         height: 100%; width: {min(fire_coverage, 100)}%; border-radius: 10px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk assessment
            if fire_coverage > 10:
                risk_level = "HIGH 🔥"
                risk_color = COLORS["danger"]
                risk_bg = "#FFE4E1"
            elif fire_coverage > 5:
                risk_level = "MEDIUM ⚠️"
                risk_color = COLORS["warning"]
                risk_bg = "#FFF8DC"
            else:
                risk_level = "LOW ✅"
                risk_color = COLORS["success"]
                risk_bg = "#F0FFF0"
                
            st.markdown(f"""
            <div class='stat-card' style='background-color: {risk_bg}; border: 2px solid {risk_color};'>
                <h4 style='color: {COLORS["text_primary"]}; margin: 0; text-align: center;'>
                    Risk Level: <span style='color: {risk_color};'>{risk_level}</span>
                </h4>
                <p style='text-align: center; color: {COLORS["text_secondary"]}; margin: 0.5rem 0 0 0;'>
                    {'Immediate attention required' if fire_coverage > 10 else 'Monitor situation' if fire_coverage > 5 else 'Low risk area'}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        success_color = COLORS["success"] if current_theme == 'light' else COLORS["accent"]
        st.markdown(f"""
        <div style='background-color: {COLORS["card_bg"]}; padding: 1rem; border-radius: 10px; border: 2px solid {success_color}; margin: 1rem 0;'>
            <p style='color: {success_color}; text-align: center; margin: 0; font-weight: bold;'>
                ✅ Analysis complete! Fire zones highlighted in alert orange.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Download button
        with open(OUTPUT_PATH, "rb") as f:
            st.download_button(
                label="📥 Download Fire Analysis",
                data=f,
                file_name="wildfire_analysis.png",
                mime="image/png",
                use_container_width=True
            )

# === Footer ===
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: {COLORS["text_secondary"]}; padding: 1rem;'>
    <p>🌲 Forest Fire Detection System | 🎨 {current_theme.title()} Nature Theme: Forest Greens & Ocean Blues</p>
    <p style='font-size: 0.8rem;'>Current theme: {current_theme.title()} Mode</p>
</div>
""", unsafe_allow_html=True)