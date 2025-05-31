import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import requests
import hashlib

# --- Configuration ---
MODEL_URL = "https://github.com/SREESAIARJUN/Alzheimer-s-Disease-Detection/releases/download/v1/alzheimer_cnn_model.pth"
EXPECTED_SHA256 = "4dabbe0229ed2f118674eac834e7c5f5f05c648432fa641ed4e4ec68e363e69d"
LOCAL_MODEL_PATH = "alzheimer_cnn_model.pth"

NUM_CLASSES = 4
CLASS_NAMES = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
IMAGE_SIZE = (128, 128)
MODEL_MEAN = [0.485, 0.456, 0.406]
MODEL_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_IMAGE_DIR = "sample_images"
SAMPLE_IMAGES = {
    "Sample 1 (Mild)": os.path.join(SAMPLE_IMAGE_DIR, "sample_mild.png"),
    "Sample 2 (Moderate)": os.path.join(SAMPLE_IMAGE_DIR, "sample_moderate.png"),
    "Sample 3 (Non-Demented)": os.path.join(SAMPLE_IMAGE_DIR, "sample_non.png"),
    "Sample 4 (Very Mild)": os.path.join(SAMPLE_IMAGE_DIR, "sample_verymild.png"),
}

# --- Model Definition ---
class AlzheimerCNN(nn.Module):
    def __init__(self, num_classes_model):
        super(AlzheimerCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes_model)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- Grad-CAM Implementation ---
class GradCAM:
    def __init__(self, model_gc, target_layer_gc):
        self.model = model_gc
        self.target_layer = target_layer_gc
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input_val, output_val): self.activations = output_val
        def backward_hook(module, grad_in, grad_out): self.gradients = grad_out[0]
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval(); self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None: class_idx = torch.argmax(output, dim=1).item()
        target_score = output[0, class_idx]
        target_score.backward()
        if self.gradients is None or self.activations is None:
            st.error("Error: Gradients/activations not captured for Grad-CAM."); return None, class_idx
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations_weighted = self.activations
        for i in range(activations_weighted.shape[1]): activations_weighted[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations_weighted, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        if heatmap.max() > 0: heatmap /= heatmap.max()
        return heatmap.cpu().detach().numpy(), class_idx

def superimpose_heatmap(heatmap, original_image_pil, alpha=0.6, colormap_val=cv2.COLORMAP_JET):
    original_image_np = np.array(original_image_pil.convert('RGB'))
    heatmap_resized = cv2.resize(heatmap, (original_image_np.shape[1], original_image_np.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap_val)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed_img_np = (alpha * original_image_np + (1 - alpha) * heatmap_colored).astype(np.uint8)
    return Image.fromarray(superimposed_img_np)

# --- Helper Functions ---
def download_file_from_url(url, destination_path, expected_sha256):
    # Using st.empty() for a cleaner progress bar experience
    progress_text_area = st.empty()
    progress_bar_area = st.empty()
    
    progress_text_area.info(f"Model not found locally. Downloading from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 8192
            downloaded_size = 0
            sha256_hash = hashlib.sha256()
            with open(destination_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    f.write(chunk); sha256_hash.update(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0: progress_bar_area.progress(min(1.0, downloaded_size / total_size))
                    else: progress_bar_area.progress(0.5) # Indeterminate if no total size
            progress_bar_area.progress(1.0)
            progress_text_area.success("Download complete!")
        downloaded_file_sha256 = sha256_hash.hexdigest()
        if downloaded_file_sha256 == expected_sha256:
            st.success("Model checksum verified."); return True
        else:
            st.error(f"Checksum mismatch! Expected: {expected_sha256}, Got: {downloaded_file_sha256}"); os.remove(destination_path); return False
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        if os.path.exists(destination_path): os.remove(destination_path)
        return False
    finally: # Clear progress elements
        progress_text_area.empty()
        progress_bar_area.empty()


@st.cache_resource
def load_alzheimer_model(local_model_path_param, model_url_param, expected_sha256_param, num_classes_param, device_param):
    if not os.path.exists(local_model_path_param):
        if not download_file_from_url(model_url_param, local_model_path_param, expected_sha256_param): return None
    else:
        st.info(f"Found local model: {local_model_path_param}. Verifying integrity...")
        sha256_hash = hashlib.sha256()
        with open(local_model_path_param, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b""): sha256_hash.update(byte_block)
        if sha256_hash.hexdigest() != expected_sha256_param:
            st.warning("Local model checksum mismatch. Re-downloading...")
            os.remove(local_model_path_param)
            if not download_file_from_url(model_url_param, local_model_path_param, expected_sha256_param): return None
        else: st.success("Local model integrity verified.")
    try:
        model = AlzheimerCNN(num_classes_model=num_classes_param)
        model.load_state_dict(torch.load(local_model_path_param, map_location=device_param))
        model.to(device_param); model.eval(); return model
    except Exception as e:
        st.error(f"Error loading model: {e}"); return None

def preprocess_image(image_pil, image_size_param, mean_param, std_param):
    transform = transforms.Compose([
        transforms.Resize(image_size_param), transforms.ToTensor(),
        transforms.Normalize(mean=mean_param, std=std_param)])
    return transform(image_pil).unsqueeze(0)

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Alzheimer's Detection Aid")

# --- Load Model (Early to stop if fails) ---
model = load_alzheimer_model(LOCAL_MODEL_PATH, MODEL_URL, EXPECTED_SHA256, NUM_CLASSES, DEVICE)
if model is None:
    st.error("Fatal Error: Model could not be loaded. Application cannot proceed."); st.stop()

# --- Initialize Grad-CAM ---
grad_cam_analyzer = None
try:
    target_layer = model.features[9] # Last Conv2D layer in AlzheimerCNN
    grad_cam_analyzer = GradCAM(model, target_layer)
except Exception as e:
    st.warning(f"Could not initialize Grad-CAM: {e}. Heatmaps might not be available.")


# --- Main App Layout ---
st.title("🧠 Alzheimer's Disease Detection Aid")
st.markdown("Upload an MRI scan or use a sample image. The AI will predict the Alzheimer's stage and show a heatmap indicating important regions for its decision.")
st.markdown("---")

# --- Image Selection (Moved to Main Area) ---
st.subheader("🖼️ Step 1: Select or Upload an MRI Image")
input_source = st.radio(
    "Choose image source:",
    ("Upload an Image", "Use a Sample Image"),
    key="input_source_radio",
    horizontal=True # Makes radio buttons appear inline
)

input_image_pil = None
image_caption = "No image selected"

if input_source == "Upload an Image":
    uploaded_file = st.file_uploader(
        "Click to upload your MRI image...", type=["png", "jpg", "jpeg"], key="file_uploader",
        label_visibility="collapsed" # Hides the default label "Drag and drop file here"
    )
    if uploaded_file:
        try:
            input_image_pil = Image.open(uploaded_file).convert("RGB")
            image_caption = f"Uploaded: {uploaded_file.name}"
        except Exception as e: st.error(f"Error processing uploaded image: {e}")
else: # Use a Sample Image
    missing_samples = [key for key, path in SAMPLE_IMAGES.items() if not os.path.exists(path)]
    if missing_samples:
        st.error(f"Missing samples: {', '.join(missing_samples)}. Check '{SAMPLE_IMAGE_DIR}' folder.")
    else:
        selected_sample_key = st.selectbox(
            "Choose a sample image:", list(SAMPLE_IMAGES.keys()), key="sample_selector",
            label_visibility="collapsed" # Hides the default label
        )
        if selected_sample_key:
            try:
                input_image_pil = Image.open(SAMPLE_IMAGES[selected_sample_key]).convert("RGB")
                image_caption = f"Sample: {selected_sample_key}"
            except Exception as e: st.error(f"Error opening sample '{selected_sample_key}': {e}")

st.markdown("---") # Separator after image selection

# --- Display Image and Analysis Button (if image is selected) ---
if input_image_pil is None:
    st.info("☝️ Please select or upload an image above to begin analysis.")
else:
    st.subheader("🖼️ Step 2: Review Image & Analyze")
    
    # Single column for the image and button, then results below
    col_image_display = st.container() # Use a container to group these
    with col_image_display:
        st.image(input_image_pil, caption=image_caption, use_container_width=True)
        
        if st.button("🔍 Predict Stage & Generate Heatmap", key="predict_button", use_container_width=True, type="primary"):
            with st.spinner("Analyzing image... Please wait."):
                image_tensor = preprocess_image(input_image_pil, IMAGE_SIZE, MODEL_MEAN, MODEL_STD).to(DEVICE)
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    predicted_class = CLASS_NAMES[predicted_idx.item()]
                    confidence_percent = confidence.item() * 100

                # --- Results Display ---
                st.subheader("📊 Analysis Results")
                
                # Prediction and Confidence in a single line if possible, or styled nicely
                st.metric(label="Predicted Stage", value=predicted_class, delta=f"{confidence_percent:.2f}% Confidence", delta_color="off")
                # st.success(f"**Predicted Stage:** {predicted_class}")
                # st.progress(confidence.item(), text=f"Confidence: {confidence_percent:.2f}%")

                # Display class probabilities
                st.markdown("**Prediction Confidence Breakdown:**")
                prob_data = {"Class": [], "Probability (%)": []}
                for i in range(NUM_CLASSES):
                    prob_data["Class"].append(CLASS_NAMES[i])
                    prob_data["Probability (%)"].append(f"{probabilities[0, i].item()*100:.2f}")
                st.dataframe(prob_data, use_container_width=True)
                
                # Grad-CAM
                if grad_cam_analyzer:
                    try:
                        st.markdown("**Explainability Heatmap (Grad-CAM):**")
                        heatmap, _ = grad_cam_analyzer.generate_heatmap(image_tensor, class_idx=predicted_idx.item())
                        if heatmap is not None:
                            superimposed_image = superimpose_heatmap(heatmap, input_image_pil)
                            st.image(superimposed_image, caption=f"Model Focus for '{predicted_class}'", use_container_width=True)
                        else: st.warning("Could not generate Grad-CAM heatmap.")
                    except Exception as e: st.error(f"Grad-CAM generation error: {e}")
                else: st.info("Grad-CAM analyzer not available for heatmap generation.")
        else:
            st.caption("Click the button above to analyze the selected image.") # Prompt if button not clicked yet

st.markdown("---")
st.caption("Disclaimer: This tool is for educational and research purposes only. It is not a substitute for professional medical diagnosis or advice.")
st.caption(f"Model version: {EXPECTED_SHA256[:8]}... | App running on Streamlit.")
