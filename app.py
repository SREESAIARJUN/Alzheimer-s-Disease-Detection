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
MODEL_URL = "https://github.com/SREESAIARJUN/Alzheimer-s-Disease-Detection/releases/download/v1/alzheimer_cnn_model.pth" # Added
EXPECTED_SHA256 = "4dabbe0229ed2f118674eac834e7c5f5f05c648432fa641ed4e4ec68e363e69d" # Added
LOCAL_MODEL_PATH = "alzheimer_cnn_model.pth" # Local path to save/check for the model

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

# --- Model Definition (Must be the same as used for training) ---
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
        def forward_hook(module, input_val, output_val):
            self.activations = output_val
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval()
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        target_score = output[0, class_idx]
        target_score.backward()
        
        if self.gradients is None or self.activations is None:
            st.error("Error: Gradients or activations not captured for Grad-CAM.")
            return None, class_idx

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations_weighted = self.activations
        for i in range(activations_weighted.shape[1]):
            activations_weighted[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations_weighted, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
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
    st.info(f"Model file not found locally. Downloading from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status() # Raise an exception for bad status codes
            total_size = int(r.headers.get('content-length', 0))
            block_size = 8192 # 8KB
            progress_bar = st.progress(0)
            downloaded_size = 0
            sha256_hash = hashlib.sha256()

            with open(destination_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    sha256_hash.update(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress_bar.progress(min(1.0, downloaded_size / total_size))
                    else:
                        # If content-length is not available, show a generic progress
                        progress_bar.progress(0.5) # Or update based on chunks downloaded
            progress_bar.progress(1.0) # Ensure it reaches 100%
            st.success("Download complete!")

        # Verify SHA256 checksum
        downloaded_file_sha256 = sha256_hash.hexdigest()
        if downloaded_file_sha256 == expected_sha256:
            st.success("Model checksum verified successfully.")
            return True
        else:
            st.error(f"Checksum mismatch! Downloaded file is corrupted. "
                     f"Expected: {expected_sha256}, Got: {downloaded_file_sha256}")
            os.remove(destination_path) # Delete corrupted file
            return False
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading model: {e}")
        if os.path.exists(destination_path):
            os.remove(destination_path) # Clean up partially downloaded file
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during download: {e}")
        if os.path.exists(destination_path):
            os.remove(destination_path)
        return False


@st.cache_resource
def load_alzheimer_model(local_model_path_param, model_url_param, expected_sha256_param, num_classes_param, device_param):
    # Check if model exists locally, if not, download
    if not os.path.exists(local_model_path_param):
        if not download_file_from_url(model_url_param, local_model_path_param, expected_sha256_param):
            st.error("Failed to download or verify the model. Please check the URL and try again.")
            return None # Stop if download or verification fails
    else:
        # If file exists, verify its integrity (optional, but good if concerned about local corruption)
        st.info(f"Found local model: {local_model_path_param}. Verifying integrity...")
        sha256_hash = hashlib.sha256()
        with open(local_model_path_param, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        if sha256_hash.hexdigest() != expected_sha256_param:
            st.warning("Local model checksum mismatch. Re-downloading...")
            os.remove(local_model_path_param)
            if not download_file_from_url(model_url_param, local_model_path_param, expected_sha256_param):
                st.error("Failed to re-download or verify the model.")
                return None
        else:
            st.success("Local model integrity verified.")

    try:
        model = AlzheimerCNN(num_classes_model=num_classes_param)
        model.load_state_dict(torch.load(local_model_path_param, map_location=device_param))
        model.to(device_param)
        model.eval()
        return model
    except FileNotFoundError: # Should be handled by download logic, but as a fallback
        st.error(f"Model file not found at {local_model_path_param} even after download attempt.")
        return None
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

def preprocess_image(image_pil, image_size_param, mean_param, std_param):
    transform = transforms.Compose([
        transforms.Resize(image_size_param),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_param, std=std_param)
    ])
    return transform(image_pil).unsqueeze(0)

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Alzheimer's Diagnosis Assistant")

st.title("🧠 Alzheimer's Disease Diagnosis Assistant")
st.markdown("""
    Upload an MRI image or select a sample to predict the stage of Alzheimer's disease.
    The system will also display a Grad-CAM heatmap to highlight areas the model focused on.
""")

# Load Model
# Pass the URL and SHA to the loading function
model = load_alzheimer_model(LOCAL_MODEL_PATH, MODEL_URL, EXPECTED_SHA256, NUM_CLASSES, DEVICE)


if model is None:
    st.error("Model could not be loaded. The application cannot proceed.")
    st.stop()

# Initialize Grad-CAM
try:
    target_layer = model.features[9] # For AlzheimerCNN, last Conv2d is features[9]
    grad_cam_analyzer = GradCAM(model, target_layer)
except Exception as e:
    st.warning(f"Could not initialize Grad-CAM: {e}. Heatmaps might not be available.")
    grad_cam_analyzer = None


st.sidebar.header("Choose Image Source")
input_source = st.sidebar.radio("Select input type:", ("Upload an Image", "Use a Sample Image"))

uploaded_file = None
selected_sample_key = None
input_image_pil = None

if input_source == "Upload an Image":
    uploaded_file = st.sidebar.file_uploader("Upload your MRI image (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        try:
            input_image_pil = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Error opening or processing uploaded image: {e}")
            input_image_pil = None
elif input_source == "Use a Sample Image":
    missing_samples = [key for key, path in SAMPLE_IMAGES.items() if not os.path.exists(path)]
    if missing_samples:
        st.sidebar.error(f"Missing sample images: {', '.join(missing_samples)}. Please check the '{SAMPLE_IMAGE_DIR}' folder.")
    else:
        selected_sample_key = st.sidebar.selectbox("Choose a sample image:", list(SAMPLE_IMAGES.keys()))
        if selected_sample_key:
            try:
                input_image_pil = Image.open(SAMPLE_IMAGES[selected_sample_key]).convert("RGB")
            except Exception as e:
                st.error(f"Error opening sample image '{selected_sample_key}': {e}")
                input_image_pil = None

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Image")
    if input_image_pil:
        st.image(input_image_pil, caption="Selected/Uploaded Image", use_column_width=True)
    else:
        st.info("Please upload an image or select a sample from the sidebar.")

with col2:
    st.subheader("Prediction & Explainability")
    if input_image_pil and st.button("🔍 Predict Stage & Show Heatmap"):
        if model:
            with st.spinner("Processing..."):
                image_tensor = preprocess_image(input_image_pil, IMAGE_SIZE, MODEL_MEAN, MODEL_STD).to(DEVICE)
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    predicted_class = CLASS_NAMES[predicted_idx.item()]
                    confidence_percent = confidence.item() * 100

                st.success(f"**Predicted Stage:** {predicted_class}")
                st.info(f"**Confidence:** {confidence_percent:.2f}%")
                st.markdown("**Class Probabilities:**")
                prob_data = {CLASS_NAMES[i]: f"{probabilities[0, i].item()*100:.2f}%" for i in range(NUM_CLASSES)}
                st.table(prob_data)

                if grad_cam_analyzer:
                    try:
                        heatmap, _ = grad_cam_analyzer.generate_heatmap(image_tensor, class_idx=predicted_idx.item())
                        if heatmap is not None:
                            superimposed_image = superimpose_heatmap(heatmap, input_image_pil)
                            st.image(superimposed_image, caption=f"Grad-CAM Heatmap for '{predicted_class}'", use_column_width=True)
                        else:
                            st.warning("Could not generate Grad-CAM heatmap for this image.")
                    except Exception as e:
                        st.error(f"Error during Grad-CAM generation: {e}")
                else:
                    st.warning("Grad-CAM analyzer not available.")
        else:
            st.error("Model not loaded. Cannot predict.")
    elif input_image_pil:
        st.info("Click the 'Predict Stage & Show Heatmap' button to see the results.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed for early Alzheimer's diagnosis assistance.")
st.sidebar.markdown("Note: This tool is for research/demonstration purposes and not a substitute for professional medical advice.")
