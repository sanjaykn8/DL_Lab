# app.py
import io
import time
from datetime import datetime

import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

import streamlit as st
import matplotlib.pyplot as plt

# -----------------------
# Configuration / Palette
# -----------------------
PALETTE = np.array([
    [0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128],
    [128,0,128], [0,128,128], [128,128,128], [64,0,0], [192,0,0],
    [64,128,0], [192,128,0], [64,0,128], [192,0,128], [64,128,128],
    [192,128,128], [0,64,0], [128,64,0], [0,192,0], [128,192,0],
], dtype=np.uint8)

# -----------------------
# Utilities
# -----------------------
def preprocess(img_pil, short_size=520):
    transform = T.Compose([
        T.Resize(short_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return transform(img_pil).unsqueeze(0)

def postprocess(output_tensor):
    # output_tensor is model(inp)
    out = output_tensor['out'][0].argmax(0).cpu().numpy().astype(np.uint8)
    return out

def colorize_mask(mask, palette=PALETTE):
    h,w = mask.shape
    color_mask = np.zeros((h,w,3), dtype=np.uint8)
    unique = np.unique(mask)
    for cls in unique:
        color_mask[mask==cls] = palette[int(cls) % len(palette)]
    return color_mask

def overlay_image(img_np, color_mask, alpha=0.5):
    # img_np: HWC uint8
    overlay = (img_np.astype(np.float32) * (1.0-alpha) + color_mask.astype(np.float32) * alpha)
    return overlay.astype(np.uint8)

def pil_to_tboard(img_pil):
    # returns torch.Tensor (C,H,W) float32 in range [0,1]
    arr = np.array(img_pil).astype(np.float32) / 255.0
    if arr.ndim == 2:  # grayscale
        arr = np.stack([arr]*3, axis=-1)
    arr = arr.transpose(2,0,1)
    return torch.from_numpy(arr)

def npimg_to_tboard(arr_uint8):
    # arr_uint8: HWC uint8 -> returns torch.Tensor (C,H,W)
    arr = arr_uint8.astype(np.float32) / 255.0
    arr = arr.transpose(2,0,1)
    return torch.from_numpy(arr)

# -----------------------
# Model loader (cached)
# -----------------------
@st.cache_resource(ttl=3600)
def load_deeplab(pretrained=True, device_name='cpu'):
    device = torch.device(device_name)
    try:
        # try legacy API (works for many torchvision versions)
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=pretrained)
    except TypeError:
        # fallback to newer weights API
        # note: if this fails, user should update torchvision or adjust accordingly
        try:
            weights = torchvision.models.segmentation.DeeplabV3_ResNet101_Weights.DEFAULT
            model = torchvision.models.segmentation.deeplabv3_resnet101(weights=weights)
        except Exception:
            model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=pretrained)
    model.to(device).eval()
    return model, device

# -----------------------
# TensorBoard writer 
# -----------------------
def make_writer():
    t = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = f"runs/seg_{t}"
    writer = SummaryWriter(log_dir=logdir)
    return writer, logdir

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(layout="wide")
st.title("DeepLab Segmentation — Streamlit + TensorBoard")

# Sidebar controls
st.sidebar.header("Settings")
device_options = ["auto", "cpu"]
if torch.cuda.is_available():
    device_options.append("cuda")
device_choice = st.sidebar.selectbox("Device", device_options, index=0)
alpha = st.sidebar.slider("Overlay alpha", 0.0, 1.0, 0.5, step=0.05)
resize_to = st.sidebar.selectbox("Resize model input (short side)", [320, 420, 520, 720], index=2)

if 'writer' not in st.session_state:
    st.session_state.writer, st.session_state.logdir = make_writer()
    st.session_state.step = 0

st.sidebar.markdown(f"TensorBoard logdir: `{st.session_state.logdir}`")
st.sidebar.markdown("To view TensorBoard:\n```\ntensorboard --logdir {logdir}\n```")

# File uploader
uploaded = st.file_uploader("Upload an image (JPEG/PNG). Or use sample image.", type=['jpg','jpeg','png'])
sample_col, main_col = st.columns([1,3])

with sample_col:
    st.subheader("Sample")
    if st.button("Use sample dog image"):
        uploaded = io.BytesIO()
        # generate a small sample (a plain colored placeholder) if user didn't include sample file
        # but try to fetch a packaged sample if present; here we create a simple placeholder
        sample_img = Image.new("RGB", (640,480), color=(120,120,200))
        buffer = io.BytesIO()
        sample_img.save(buffer, format="JPEG")
        buffer.seek(0)
        uploaded = buffer

if uploaded is None:
    st.info("Upload an image to run segmentation.")
    st.stop()

# Read image
image = Image.open(uploaded).convert("RGB")
orig_w, orig_h = image.size

# Model loading
if device_choice == "auto":
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device_name = device_choice
model, device = load_deeplab(pretrained=True, device_name=device_name)

st.sidebar.write(f"Model: deeplabv3_resnet101 (eval)  •  Device: {device}")

# Run inference button
if st.button("Run segmentation"):
    with st.spinner("Running model..."):
        inp = preprocess(image, short_size=resize_to).to(device)
        t0 = time.time()
        with torch.no_grad():
            out = model(inp)
        t1 = time.time()
        inf_time = t1 - t0

        mask = postprocess(out)  # H x W (resized)
        # resize original image to mask size for overlay display
        img_resized = image.resize((mask.shape[1], mask.shape[0]))
        img_np = np.array(img_resized).astype(np.uint8)
        color_mask = colorize_mask(mask)
        overlay = overlay_image(img_np, color_mask, alpha=alpha)

        # display
        st.subheader("Results")
        c1, c2, c3 = st.columns(3)
        c1.image(image, caption="Original (full size)", use_column_width=True)
        c2.image(color_mask, caption="Segmentation mask (colorized)", use_column_width=True)
        c3.image(overlay, caption=f"Overlay (alpha={alpha:.2f}) — inference {inf_time*1000:.1f} ms", use_column_width=True)

        # Classes present
        unique = np.unique(mask)
        st.markdown(f"**Classes present (indices):** {unique.tolist()}")

        # Offer downloads
        mask_pil = Image.fromarray(mask)  # single-channel
        color_pil = Image.fromarray(color_mask)
        overlay_pil = Image.fromarray(overlay)

        buf_mask = io.BytesIO()
        mask_pil.save(buf_mask, format="PNG")
        buf_mask.seek(0)
        st.download_button("Download mask (PNG, class indices)", data=buf_mask, file_name="mask.png", mime="image/png")

        buf_overlay = io.BytesIO()
        overlay_pil.save(buf_overlay, format="PNG")
        buf_overlay.seek(0)
        st.download_button("Download overlay (PNG)", data=buf_overlay, file_name="overlay.png", mime="image/png")

        # Log to TensorBoard
        step = st.session_state.step + 1
        writer = st.session_state.writer

        # input image (original resized to mask for easy comparison)
        writer.add_image('input/original_resized', pil_to_tboard(img_resized), step)

        # mask (color) and overlay
        writer.add_image('output/mask_color', npimg_to_tboard(color_mask), step)
        writer.add_image('output/overlay', pil_to_tboard(overlay_pil), step)

        # raw mask as single channel (0..N) -> convert to 3-channel for TB visualization
        mask_color_for_tb = color_mask  # already colorized
        writer.add_image('output/mask_color_for_tb', npimg_to_tboard(mask_color_for_tb), step)

        # scalar stats
        writer.add_scalar('inference/time_ms', inf_time * 1000.0, step)
        writer.add_scalar('inference/width', mask.shape[1], step)
        writer.add_scalar('inference/height', mask.shape[0], step)
        writer.add_text('meta/classes_present', str(unique.tolist()), step)

        writer.flush()
        st.session_state.step = step

        st.success(f"Logged to TensorBoard under run `{st.session_state.logdir}` (step {step}).")

# show small footer with instructions
st.markdown("---")
st.markdown(
    "**Run locally**\n\n"
    "- Start the Streamlit app: `streamlit run app.py`\n"
    "- View TensorBoard: `tensorboard --logdir runs/` (then open displayed URL in browser)\n\n"
    "The app logs images and timing to a timestamped folder under `runs/`."
)