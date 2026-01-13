import streamlit as st
import os
import cv2
import numpy as np
import io
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from PIL import Image, ImageOps

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="EyeCrop Studio",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- GOOGLE MATERIAL CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Google Sans', sans-serif;
        color: #202124;
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding-bottom: 2rem;
    }
    .main-header h1 {
        font-weight: 400;
        color: #202124;
        font-size: 2.5rem;
    }
    .main-header span {
        color: #4285F4;
        font-weight: 700;
    }
    
    /* Card Style */
    .stApp > header {visibility: hidden;}
    .image-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 20px;
        border: 1px solid #dadce0;
        transition: all 0.3s cubic-bezier(.25,.8,.25,1);
    }
    .image-card:hover {
        box-shadow: 0 14px 28px rgba(0,0,0,0.25), 0 10px 10px rgba(0,0,0,0.22);
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #ffffff;
        color: #1a73e8;
        border: 1px solid #dadce0;
        border-radius: 24px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #f8f9fa;
        color: #174ea6;
        border-color: #174ea6;
    }
    
    /* Primary Action Buttons */
    div[data-testid="column"] button:nth-of-type(1) {
        /* Heuristic: Assuming purely first button in col is primary sometimes. 
           Better to specific keys if possible, but st limitation. */
    }

    /* Uploader */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #dadce0;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }
    
</style>
""", unsafe_allow_html=True)

# Force CPU/CUDA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- CONFIGURATION (Relative Paths) ---
# Assuming app is running from the same folder as models
MODEL_PATH_STANDARD = "model_standard.pth"
MODEL_PATH_SLITLAMP = "model_slitlamp.pth"
IMG_SIZE = 768

# --- HEADER ---
st.markdown("""
<div class="main-header">
    <h1><span>Eye</span>Crop Studio</h1>
    <p>Intelligent ROI Detection & Cropping</p>
</div>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    models = {}
    
    # Check current directory
    cwd = os.getcwd()
    path_std = os.path.join(cwd, MODEL_PATH_STANDARD)
    path_slit = os.path.join(cwd, MODEL_PATH_SLITLAMP)

    # 1. Standard Model
    if os.path.exists(path_std):
        try:
            model_std = smp.UnetPlusPlus(encoder_name="timm-efficientnet-b0", in_channels=3, classes=1)
            model_std.load_state_dict(torch.load(path_std, map_location=DEVICE))
            model_std.to(DEVICE)
            model_std.eval()
            models['standard'] = model_std
        except:
             st.error("Failed to load Standard Model")
    
    # 2. Slitlamp Model
    if os.path.exists(path_slit):
        try:
            model_slit = smp.UnetPlusPlus(encoder_name="timm-efficientnet-b0", in_channels=3, classes=1)
            model_slit.load_state_dict(torch.load(path_slit, map_location=DEVICE))
            model_slit.to(DEVICE)
            model_slit.eval()
            models['slitlamp'] = model_slit
        except:
             st.error("Failed to load Slitlamp Model")
        
    return models

with st.spinner("Loading AI Models..."):
    models = load_models()

if not models:
    st.warning("⚠️ No models found in current directory. Please ensure 'model_standard.pth' and 'model_slitlamp.pth' are present.")

# --- HELPER FUNCTIONS ---
def get_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def predict_mask(model, image_rgb):
    transform = get_transform()
    augmented = transform(image=image_rgb)["image"].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(augmented)
        prob = torch.sigmoid(output)[0, 0].cpu().numpy()
        
    return prob

def crop_image_from_prob(image_rgb, prob_map, padding=0):
    h, w = image_rgb.shape[:2]
    mask = (prob_map > 0.5).astype(np.uint8)
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    coords = cv2.findNonZero(mask_resized)
    if coords is not None:
        x, y, rect_w, rect_h = cv2.boundingRect(coords)
        x = max(0, x - padding)
        y = max(0, y - padding)
        rect_w = min(w - x, rect_w + 2*padding)
        rect_h = min(h - y, rect_h + 2*padding)
        crop = image_rgb[y:y+rect_h, x:x+rect_w]
        return crop, mask_resized
    return None, mask_resized

def detect_best_model(image_rgb):
    if 'standard' not in models or 'slitlamp' not in models:
        return 'standard'
    prob_std = predict_mask(models['standard'], image_rgb)
    prob_slit = predict_mask(models['slitlamp'], image_rgb)
    
    # Metric: Mean confidence of top pixels
    score_std = np.mean(prob_std[prob_std > 0.5]) if np.max(prob_std) > 0.5 else 0
    score_slit = np.mean(prob_slit[prob_slit > 0.5]) if np.max(prob_slit) > 0.5 else 0
    
    if score_slit > score_std:
        return 'slitlamp'
    return 'standard'

# --- UI CONTROLS ---
col_mode, col_pad = st.columns([2, 1])
with col_mode:
    mode = st.selectbox("Workflow Mode", ["Auto-Detect (Smart)", "Standard Mode", "Slitlamp Mode"])

uploaded_files = st.file_uploader("Upload Eye Images", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True, label_visibility="collapsed")

if uploaded_files:
    if 'results' not in st.session_state:
        st.session_state.results = {}

    st.write("---")
    
    for uploaded_file in uploaded_files:
        file_key = uploaded_file.name
        
        # Load Image
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image) 
        image = image.convert('RGB')
        image_np = np.array(image)
        
        # CARD CONTAINER
        with st.container():
            st.markdown(f'<div class="image-card"><h3>{file_key}</h3></div>', unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.image(image, use_container_width=True)
            
            # LOGIC
            if file_key not in st.session_state.results:
                selected_model_name = 'standard'
                
                if "Auto" in mode:
                    detected = detect_best_model(image_np)
                    
                    if f"confirm_{file_key}" not in st.session_state:
                        st.session_state[f"confirm_{file_key}"] = False
                    if f"show_options_{file_key}" not in st.session_state:
                        st.session_state[f"show_options_{file_key}"] = False

                    if not st.session_state[f"confirm_{file_key}"]:
                        with c2:
                            st.info(f"✨ AI suggests: **{detected.upper()}**")
                            
                            if not st.session_state[f"show_options_{file_key}"]:
                                b_ok, b_no = st.columns(2)
                                if b_ok.button("✅ Confirm", key=f"ok_{file_key}"):
                                        st.session_state[f"confirm_{file_key}"] = True
                                        st.session_state[f"model_{file_key}"] = detected
                                        st.rerun()
                                if b_no.button("Change...", key=f"bad_{file_key}"):
                                        st.session_state[f"show_options_{file_key}"] = True
                                        st.rerun()
                            else:
                                st.caption("Select manually:")
                                b_std, b_slit = st.columns(2)
                                if b_std.button("Standard", key=f"use_std_{file_key}"):
                                        st.session_state[f"confirm_{file_key}"] = True
                                        st.session_state[f"model_{file_key}"] = 'standard'
                                        st.rerun()
                                if b_slit.button("Slitlamp", key=f"use_slit_{file_key}"):
                                        st.session_state[f"confirm_{file_key}"] = True
                                        st.session_state[f"model_{file_key}"] = 'slitlamp'
                                        st.rerun()
                        continue
                    else:
                        selected_model_name = st.session_state[f"model_{file_key}"]
                elif "Slitlamp" in mode:
                    selected_model_name = 'slitlamp'
                
                # EXECUTE CROP
                with c2:
                    with st.spinner("Processing..."):
                        chosen_model = models.get(selected_model_name)
                        if chosen_model:
                            prob = predict_mask(chosen_model, image_np)
                            crop, _ = crop_image_from_prob(image_np, prob)
                            
                            if crop is not None:
                                st.image(crop, caption=f"Result ({selected_model_name})", use_container_width=True)
                                st.session_state.results[file_key] = crop
                                
                                # Download
                                im_pil = Image.fromarray(crop)
                                buf = io.BytesIO()
                                im_pil.save(buf, format="PNG")
                                byte_im = buf.getvalue()
                                
                                st.download_button(
                                    label="Save Image",
                                    data=byte_im,
                                    file_name=file_key,
                                    mime="image/png",
                                    key=f"dl_{file_key}"
                                )
                            else:
                                st.error("No ROI detected.")
            else:
                # CACHED RESULT
                with c2:
                    crop = st.session_state.results[file_key]
                    st.image(crop, caption="Result", use_container_width=True)
                    
                    im_pil = Image.fromarray(crop)
                    buf = io.BytesIO()
                    im_pil.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="Save Image",
                        data=byte_im,
                        file_name=file_key,
                        mime="image/png",
                        key=f"dl_cached_{file_key}"
                    )
