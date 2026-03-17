# marine_debris_yolov5_dashboard.py

import os
import glob
import yaml
import torch
import cv2
import streamlit as st
import numpy as np

from collections import Counter
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------

DATA_YAML = "data.yaml"
WEIGHTS = "yolov5/runs/train/exp11/weights/best.pt"

IMG_SIZE = 640
CONF_THRES = 0.25
BATCH_SIZE = 8

SAVE_DIR = "runs/test_results"
LOGDIR = "logs/marine_debris"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOGDIR, exist_ok=True)

writer = SummaryWriter(LOGDIR)

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

st.title("Marine Debris Detection — YOLOv5")

run_button = st.button("Run Test Inference")

progress_bar = st.progress(0)

stats_box = st.empty()
chart_area = st.empty()
image_area = st.empty()

# --------------------------------------------------
# LOAD DATA CONFIG
# --------------------------------------------------

with open(DATA_YAML) as f:
    data = yaml.safe_load(f)

dataset_root = data["path"]
test_folder = data["test"]

TEST_DIR = os.path.join(dataset_root, test_folder)

st.write("Test folder:", TEST_DIR)

# --------------------------------------------------
# COLLECT TEST IMAGES
# --------------------------------------------------

img_paths = []

for ext in ["*.jpg","*.png","*.jpeg","*.bmp"]:
    img_paths += glob.glob(os.path.join(TEST_DIR, ext))

img_paths = sorted(img_paths)

st.write("Test images found:", len(img_paths))

if len(img_paths) == 0:
    st.error("No test images found")
    st.stop()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():

    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=WEIGHTS,
        force_reload=False
    )

    model.conf = CONF_THRES
    model.to(device)

    return model

model = load_model()

st.write("Model running on:", device)

# --------------------------------------------------
# RUN INFERENCE
# --------------------------------------------------

if run_button:

    class_counter = Counter()
    total_det = 0

    example_img = None

    total_batches = len(img_paths) // BATCH_SIZE + 1

    for i in range(0, len(img_paths), BATCH_SIZE):

        batch_paths = img_paths[i:i+BATCH_SIZE]

        results = model(batch_paths, size=IMG_SIZE)

        for j, det in enumerate(results.xyxy):

            img_path = batch_paths[j]
            img = cv2.imread(img_path)

            for *xyxy, conf, cls in det.cpu().numpy():

                x1,y1,x2,y2 = map(int, xyxy)

                cls = int(cls)
                name = model.names[cls]

                label = f"{name} {conf:.2f}"

                cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)

                cv2.putText(
                    img,
                    label,
                    (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,0,255),
                    1
                )

                class_counter[name]+=1
                total_det+=1

            out_path = os.path.join(SAVE_DIR, os.path.basename(img_path))
            cv2.imwrite(out_path,img)

            if example_img is None:
                example_img = img

        torch.cuda.empty_cache()

        progress_bar.progress((i/BATCH_SIZE+1)/total_batches)

        # TensorBoard logging
        writer.add_scalar("detections/total", total_det, i)

    # --------------------------------------------------
    # STATS
    # --------------------------------------------------

    stats_box.write(f"Total detections: {total_det}")

    for cls,count in class_counter.items():
        writer.add_scalar(f"class_counts/{cls}", count)

    # --------------------------------------------------
    # BAR CHART
    # --------------------------------------------------

    if len(class_counter)>0:

        fig = plt.figure()

        plt.bar(class_counter.keys(), class_counter.values())
        plt.title("Detections per Class")

        chart_area.pyplot(fig)

    # --------------------------------------------------
    # SHOW EXAMPLE
    # --------------------------------------------------

    if example_img is not None:

        example_img = cv2.cvtColor(example_img,cv2.COLOR_BGR2RGB)

        image_area.image(
            example_img,
            caption="Example Detection",
            use_container_width=True
        )

    writer.close()