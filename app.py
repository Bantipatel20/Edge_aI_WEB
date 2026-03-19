import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms, models

INPUT_SIZE = (512, 512)
CLASS_NAMES = ["Mixed", "Not Mixed"]


# ----------------------------------------------------------
# MODEL SETUP
# ----------------------------------------------------------
def get_last_conv_layer(model):
    """Get the last Conv2d layer from model.features for Grad-CAM."""
    last_conv = None
    for module in model.features.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


@st.cache_resource
def load_model():
    """Load trained DenseNet model, target layer, and device."""

    def modify_densenet_input_channels(model, in_channels=9):
        conv = model.features.conv0
        new_conv = nn.Conv2d(
            in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=False,
        )

        with torch.no_grad():
            new_conv.weight[:, :3] = conv.weight
            for i in range(3, in_channels):
                new_conv.weight[:, i : i + 1] = conv.weight[:, i % 3 : i % 3 + 1]

        model.features.conv0 = new_conv
        return model

    # Load DenseNet121 and modify for 9 channels
    model = models.densenet121(weights=None)
    model = modify_densenet_input_channels(model, in_channels=9)
    model.classifier = nn.Linear(model.classifier.in_features, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained weights
    model_path = "Code/best_densenet_9ch.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    target_layer = get_last_conv_layer(model)
    if target_layer is None:
        raise RuntimeError("Could not find a Conv2d layer for Grad-CAM.")

    return model, target_layer, device


def preprocess_image(image):
    """Preprocess image to 9-channel format (RGB + HSV + LAB)."""
    # Resize image
    transform = transforms.Resize(INPUT_SIZE)
    img = transform(image)
    img = np.array(img)

    # Convert RGB to HSV and LAB
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

    # Combine all channels (9 channels total)
    combined = np.concatenate([img, hsv, lab], axis=2)
    combined = combined.transpose(2, 0, 1) / 255.0

    # Convert to tensor
    tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)
    return tensor


def predict(model, image_tensor, device):
    """Make prediction on preprocessed image."""
    with torch.no_grad():
        output = model(image_tensor.to(device))
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prediction = output.argmax(1).item()
        confidence = probabilities[0][prediction].item()

    return prediction, confidence, probabilities[0].detach().cpu().numpy()


def generate_gradcam_overlay(model, target_layer, image_tensor, original_rgb, class_index, device):
    """Generate Grad-CAM heatmap overlay for a single image."""
    activations = []
    gradients = []

    def forward_hook(_, __, output):
        activations.append(output)

    def backward_hook(_, grad_input, grad_output):
        del grad_input
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        input_tensor = image_tensor.to(device)
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            output = model(input_tensor)
            score = output[:, class_index].sum()
            score.backward()

        if not activations or not gradients:
            return original_rgb, original_rgb

        activation_map = activations[0]
        gradient_map = gradients[0]

        weights = gradient_map.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation_map).sum(dim=1, keepdim=True))
        cam = cam.squeeze().detach().cpu().numpy()

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        height, width = original_rgb.shape[:2]
        cam = cv2.resize(cam, (width, height))
        heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(original_rgb, 0.62, heatmap, 0.38, 0)
        return overlay, heatmap
    finally:
        forward_handle.remove()
        backward_handle.remove()


def extract_feature_maps(model, image_tensor, device, num_features=16):
    """Extract feature maps from the first convolutional layer."""
    feature_maps = []

    def hook(_module, _input, output):
        feature_maps.append(output)

    # Register hook on first conv layer
    handle = model.features.conv0.register_forward_hook(hook)

    try:
        with torch.no_grad():
            _ = model(image_tensor.to(device))

        if feature_maps:
            fmaps = feature_maps[0][0].detach().cpu().numpy()  # [C, H, W]
            # Take first num_features channels
            fmaps = fmaps[: min(num_features, fmaps.shape[0])]
            return fmaps
    finally:
        handle.remove()

    return None


def create_feature_map_grid(feature_maps, cols=4):
    """Create a grid visualization of feature maps."""
    if feature_maps is None:
        return None

    num_maps = len(feature_maps)
    rows = (num_maps + cols - 1) // cols

    # Normalize each feature map
    normalized_maps = []
    for fm in feature_maps:
        fm_norm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
        fm_colored = cv2.applyColorMap(np.uint8(fm_norm * 255), cv2.COLORMAP_VIRIDIS)
        fm_colored = cv2.cvtColor(fm_colored, cv2.COLOR_BGR2RGB)
        normalized_maps.append(fm_colored)

    # Get dimensions
    h, w = normalized_maps[0].shape[:2]

    # Create grid
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, fm in enumerate(normalized_maps):
        r, c = idx // cols, idx % cols
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = fm

    return grid


# ----------------------------------------------------------
# STREAMLIT APP
# ----------------------------------------------------------
def main():
    st.set_page_config(
        page_title="EDGE AI - Drug Classification",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Custom CSS for professional light theme
    st.markdown(
        """
    <style>
        /* Remove top padding */
        .block-container {
            padding-top: 1rem !important;
        }

        /* Light theme base */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        }

        /* Navbar styling */
        .navbar {
            background: linear-gradient(90deg, #1a365d 0%, #2c5282 100%);
            padding: 1.2rem 2rem;
            border-radius: 0 0 12px 12px;
            margin: -1rem -1rem 2rem -1rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-align: center;
        }
        .navbar h1 {
            color: white;
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: 3px;
        }
        .navbar p {
            color: #a0aec0;
            font-size: 0.9rem;
            margin: 0.5rem 0 0 0;
        }

        /* Hide sidebar */
        [data-testid="stSidebar"] {
            display: none;
        }

        /* Card styling */
        .card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            margin-bottom: 1rem;
        }

        /* Result cards */
        .result-mixed {
            background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%);
            border-left: 4px solid #e53e3e;
            padding: 1rem 1.5rem;
            border-radius: 8px;
        }
        .result-notmixed {
            background: linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%);
            border-left: 4px solid #38a169;
            padding: 1rem 1.5rem;
            border-radius: 8px;
        }

        /* Headers */
        h2, h3 {
            color: #2d3748;
        }

        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem;
            color: #2c5282;
        }

        /* Button styling */
        .stButton > button {
            background: linear-gradient(90deg, #2c5282 0%, #1a365d 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 2rem;
            font-weight: 600;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #1a365d 0%, #2c5282 100%);
        }

        /* File uploader */
        [data-testid="stFileUploader"] {
            background: white;
            border-radius: 12px;
            padding: 1rem;
        }

        /* Info boxes */
        .stAlert {
            border-radius: 8px;
        }

        /* Hide default header, deploy button, and menu */
        header[data-testid="stHeader"] {
            display: none;
        }
        [data-testid="stToolbar"] {
            display: none;
        }
        .stDeployButton {
            display: none;
        }
        #MainMenu {
            display: none;
        }

        /* Divider */
        .divider {
            height: 2px;
            background: linear-gradient(90deg, transparent, #cbd5e0, transparent);
            margin: 1.5rem 0;
        }
    </style>

    <div class="navbar">
        <h1>🧠 EDGE AI</h1>
        <p>Drug Classification System | Powered by DenseNet121</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    try:
        with st.spinner("Loading model..."):
            model, target_layer, device = load_model()
    except Exception as exc:
        st.error(f"Model loading failed: {exc}")
        st.stop()

    # Main content area
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_input, _col_spacer = st.columns([3, 1])
    with col_input:
        input_mode = st.radio(
            "📷 Select Input Method",
            options=["Upload Image", "Camera"],
            horizontal=True,
        )

    if input_mode == "Upload Image":
        input_file = st.file_uploader(
            "Choose an image...",
            type=["png", "jpg", "jpeg"],
            help="Upload a drug image for classification",
        )
    else:
        input_file = st.camera_input("Capture an image")

    if input_file is not None:
        image = Image.open(input_file).convert("RGB")
        original_rgb = np.array(image)

        with st.spinner("Analyzing image and generating visualizations..."):
            image_tensor = preprocess_image(image)
            prediction, confidence, probabilities = predict(model, image_tensor, device)

            # Generate Grad-CAM for predicted class
            gradcam_overlay, _raw_heatmap = generate_gradcam_overlay(
                model=model,
                target_layer=target_layer,
                image_tensor=image_tensor,
                original_rgb=original_rgb,
                class_index=prediction,
                device=device,
            )

            # Generate Grad-CAM for both classes
            gradcam_mixed, heatmap_mixed = generate_gradcam_overlay(
                model=model,
                target_layer=target_layer,
                image_tensor=image_tensor,
                original_rgb=original_rgb,
                class_index=0,  # Mixed
                device=device,
            )

            gradcam_notmixed, heatmap_notmixed = generate_gradcam_overlay(
                model=model,
                target_layer=target_layer,
                image_tensor=image_tensor,
                original_rgb=original_rgb,
                class_index=1,  # Not Mixed
                device=device,
            )

            # Extract feature maps
            feature_maps = extract_feature_maps(model, image_tensor, device, num_features=16)
            feature_grid = create_feature_map_grid(feature_maps)

        predicted_class = CLASS_NAMES[prediction]

        # Image display section - Row 1: Input + Predicted Grad-CAM
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🖼️ Input & Predicted Class Heatmap")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📤 Input Image")
            st.image(original_rgb, use_container_width=True)
        with col2:
            st.markdown(f"#### 🔥 Grad-CAM ({predicted_class})")
            st.image(gradcam_overlay, use_container_width=True)

        # Row 2: Grad-CAM for both classes
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🎯 Grad-CAM Comparison (Both Classes)")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### 🔴 Mixed Class Attention")
            st.image(gradcam_mixed, use_container_width=True)
        with col4:
            st.markdown("#### 🟢 Not Mixed Class Attention")
            st.image(gradcam_notmixed, use_container_width=True)

        # Row 3: Raw Heatmaps
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🌡️ Raw Heatmaps (Without Overlay)")
        col5, col6 = st.columns(2)
        with col5:
            st.markdown("#### Mixed Heatmap")
            st.image(heatmap_mixed, use_container_width=True)
        with col6:
            st.markdown("#### Not Mixed Heatmap")
            st.image(heatmap_notmixed, use_container_width=True)

        # Row 4: Feature Maps
        if feature_grid is not None:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("### 🧩 Feature Maps (First Conv Layer)")
            st.image(feature_grid, use_container_width=True)
            st.caption("Visualization of 16 feature maps from the first convolutional layer")

        # Results section
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 Classification Result")

        if predicted_class == "Mixed":
            st.markdown(
                """
                <div class="result-mixed">
                    <h2 style="margin:0; color:#c53030;">🔴 MIXED</h2>
                    <p style="margin:0.5rem 0 0 0; color:#742a2a;">Drug mixture detected</p>
                </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-notmixed">
                    <h2 style="margin:0; color:#276749;">🟢 NOT MIXED</h2>
                    <p style="margin:0.5rem 0 0 0; color:#22543d;">Single drug detected</p>
                </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Metrics in cards
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("🎯 Confidence", f"{confidence * 100:.1f}%")
        with metric_col2:
            st.metric("📁 Predicted", predicted_class)
        with metric_col3:
            st.metric("🧪 Classes", "2")

        # Probability chart
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("#### 📈 Class Probabilities")
        prob_dict = {
            "Mixed": probabilities[0] * 100,
            "Not Mixed": probabilities[1] * 100,
        }
        st.bar_chart(prob_dict)

    else:
        # Empty state with guidance
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(
            """
        <div style="text-align: center; padding: 3rem; background: white; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
            <h3 style="color: #2d3748; margin-bottom: 1rem;">Ready for Analysis</h3>
            <p style="color: #718096; font-size: 1.1rem;">Upload an image or capture one using your camera to begin classification</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        """
    <div style="text-align: center; padding: 1rem; color: #718096; font-size: 0.85rem;">
        EDGE AI | Built for Edge Inference | DenseNet121 + Grad-CAM
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
