import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

# --- Configuration ---
MODEL_PATH = "alzheimer_cnn_model.pth" # Make sure this model is in the same directory
NUM_CLASSES = 4 # Number of classes your model predicts
CLASS_NAMES = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented'] # IMPORTANT: Order must match training
IMAGE_SIZE = (128, 128)
MODEL_MEAN = [0.485, 0.456, 0.406]
MODEL_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sample Image Paths (assuming they are in a 'sample_images' subdirectory)
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
            nn.Linear(256 * 8 * 8, 512), # Adjust if image size or architecture changes (128 -> 8x8 for 4 maxpools)
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
        def forward_hook(module, input_val, output_val): # Renamed input to input_val
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

def superimpose_heatmap(heatmap, original_image_pil, alpha=0.6, colormap_val=cv2.COLORMAP_JET): # Renamed colormap
    original_image_np = np.array(original_image_pil.convert('RGB'))
    heatmap_resized = cv2.resize(heatmap, (original_image_np.shape[1], original_image_np.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap_val)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) # OpenCV BGR to PIL RGB

    superimposed_img_np = (alpha * original_image_np + (1 - alpha) * heatmap_colored).astype(np.uint8)
    return Image.fromarray(superimposed_img_np)

# --- Helper Functions ---
@st.cache_resource # Cache the model loading
def load_alzheimer_model(model_path, num_classes_param, device_param): # Renamed parameters
    try:
        model = AlzheimerCNN(num_classes_model=num_classes_param)
        # Load state_dict, ensuring map_location for CPU if model saved on GPU
        model.load_state_dict(torch.load(model_path, map_location=device_param))
        model.to(device_param)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please ensure it's in the correct location.")
        return None
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

def preprocess_image(image_pil, image_size_param, mean_param, std_param): # Renamed parameters
    transform = transforms.Compose([
        transforms.Resize(image_size_param),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_param, std=std_param)
    ])
    return transform(image_pil).unsqueeze(0) # Add batch dimension

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Alzheimer's Diagnosis Assistant")

st.title("🧠 Alzheimer's Disease Diagnosis Assistant")
st.markdown("""
    Upload an MRI image or select a sample to predict the stage of Alzheimer's disease.
    The system will also display a Grad-CAM heatmap to highlight areas the model focused on.
""")

# Load Model
model = load_alzheimer_model(MODEL_PATH, NUM_CLASSES, DEVICE)

if model is None:
    st.stop() # Stop execution if model fails to load

# Initialize Grad-CAM
# Target the last convolutional layer in model.features. For AlzheimerCNN, this is features[9].
try:
    target_layer = model.features[9] # nn.Conv2d(128, 256, ...)
    grad_cam_analyzer = GradCAM(model, target_layer)
except Exception as e:
    st.warning(f"Could not initialize Grad-CAM: {e}. Heatmaps might not be available.")
    grad_cam_analyzer = None


# Image Input
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
    # Check if sample images exist
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
                # Preprocess image
                image_tensor = preprocess_image(input_image_pil, IMAGE_SIZE, MODEL_MEAN, MODEL_STD).to(DEVICE)

                # Get prediction
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    predicted_class = CLASS_NAMES[predicted_idx.item()]
                    confidence_percent = confidence.item() * 100

                st.success(f"**Predicted Stage:** {predicted_class}")
                st.info(f"**Confidence:** {confidence_percent:.2f}%")

                # Display class probabilities
                st.markdown("**Class Probabilities:**")
                prob_data = {CLASS_NAMES[i]: f"{probabilities[0, i].item()*100:.2f}%" for i in range(NUM_CLASSES)}
                st.table(prob_data)


                # Generate and display Grad-CAM
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
