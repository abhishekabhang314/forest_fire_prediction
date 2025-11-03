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

# === FOOD COLOR THEME ===
# Fire-inspired food colors: Fiery reds, warm oranges, earthy browns, and fresh greens
COLORS = {
    "primary": "#E74C3C",  # Tomato Red
    "secondary": "#F39C12",  # Orange Pepper
    "accent": "#D35400",  # Pumpkin Orange
    "background": "#FEF9E7",  # Cream/Yellow
    "card_bg": "#FFFFFF",
    "text_primary": "#2C3E50",  # Dark Charcoal
    "text_secondary": "#7F8C8D",  # Gray
    "success": "#27AE60",  # Fresh Green
    "warning": "#F1C40F"  # Golden Yellow
}

# === Custom CSS for Food Color Theme ===
st.markdown(f"""
<style>
    .main {{
        background-color: {COLORS['background']};
    }}
    .stApp {{
        background: linear-gradient(135deg, {COLORS['background']} 0%, #FDEDEC 100%);
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
        background-color: {COLORS['accent']};
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(231, 76, 60, 0.3);
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
    }}
    .header-section {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }}
    .feature-icon {{
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }}
    .stats-container {{
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }}
    .stat-item {{
        text-align: center;
        padding: 1rem;
    }}
</style>
""", unsafe_allow_html=True)

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
    # Use food-inspired red color (Tomato Red)
    red_overlay = Image.new("RGBA", image.size, (231, 76, 60, 120))  # Tomato Red with transparency
    overlay = Image.composite(red_overlay, overlay, mask_img)
    return Image.alpha_composite(image, overlay)

def calculate_fire_coverage(mask: np.ndarray):
    total_pixels = mask.size
    fire_pixels = np.sum(mask)
    coverage_percentage = (fire_pixels / total_pixels) * 100
    return coverage_percentage

# === STREAMLIT APP WITH NEW UI ===
st.set_page_config(
    page_title="🔥 Wildfire Detection", 
    layout="wide",
    page_icon="🔥"
)

# === Header Section ===
st.markdown(f"""
<div class="header-section">
    <h1 style="margin:0; font-size: 2.5rem;">🍅🔥 Wildfire Detection System</h1>
    <p style="margin:0; font-size: 1.2rem; opacity: 0.9;">Using AI to detect fire zones with food-inspired fiery colors</p>
</div>
""", unsafe_allow_html=True)

# === Introduction Section ===
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="text-align: center;">
        <div class="feature-icon">🛰️</div>
        <h4 style="color: #E74C3C;">Satellite Analysis</h4>
        <p>Process satellite imagery in real-time</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center;">
        <div class="feature-icon">🤖</div>
        <h4 style="color: #E74C3C;">AI Powered</h4>
        <p>Deep learning segmentation model</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center;">
        <div class="feature-icon">🎨</div>
        <h4 style="color: #E74C3C;">Food Color Theme</h4>
        <p>Tomato reds & pepper oranges</p>
    </div>
    """, unsafe_allow_html=True)

# === Main Content ===
st.markdown("---")

# Create two columns for layout
left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown(f"""
    <div style='background-color: {COLORS["card_bg"]}; padding: 1.5rem; border-radius: 15px; border-left: 5px solid {COLORS["primary"]};'>
        <h3 style='color: {COLORS["text_primary"]}; margin-top: 0;'>📤 Upload Satellite Image</h3>
        <p style='color: {COLORS["text_secondary"]};'>Upload a satellite image to detect potential fire zones highlighted in tomato red.</p>
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
        
        with st.spinner("🍳 Cooking up analysis... Please wait!"):
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
            <div style='background-color: {COLORS["card_bg"]}; padding: 1.5rem; border-radius: 15px; border: 2px solid {COLORS["warning"]};'>
                <h4 style='color: {COLORS["text_primary"]}; margin-top: 0;'>📈 Detection Statistics</h4>
                <div style='font-size: 2rem; color: {COLORS["primary"]}; font-weight: bold; text-align: center;'>
                    {fire_coverage:.2f}%
                </div>
                <p style='text-align: center; color: {COLORS["text_secondary"]};'>Fire Area Coverage</p>
                <div style='background-color: #ECF0F1; border-radius: 10px; height: 20px; margin: 1rem 0;'>
                    <div style='background-color: {COLORS["primary"]}; height: 100%; width: {min(fire_coverage, 100)}%; border-radius: 10px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk assessment
            if fire_coverage > 10:
                risk_level = "HIGH 🔥"
                risk_color = COLORS["primary"]
            elif fire_coverage > 5:
                risk_level = "MEDIUM ⚠️"
                risk_color = COLORS["warning"]
            else:
                risk_level = "LOW ✅"
                risk_color = COLORS["success"]
                
            st.markdown(f"""
            <div style='background-color: {COLORS["card_bg"]}; padding: 1rem; border-radius: 10px; border: 2px solid {risk_color}; margin-top: 1rem;'>
                <h4 style='color: {COLORS["text_primary"]}; margin: 0; text-align: center;'>Risk Level: <span style='color: {risk_color};'>{risk_level}</span></h4>
            </div>
            """, unsafe_allow_html=True)
        
        st.success("🎉 Analysis complete! Fire zones highlighted in tomato red.")
        
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
    <p>🔥 Wildfire Detection System | 🎨 Food Color Theme: Tomato Reds & Pepper Oranges</p>
</div>
""", unsafe_allow_html=True)