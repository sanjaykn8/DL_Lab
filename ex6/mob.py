# mobilenet_ip102_dashboard.py

import torch
import torch.nn as nn
import time
import numpy as np
import os
import streamlit as st

from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


# --------------------------------------------------
# PATHS
# --------------------------------------------------

DATA_ROOT = "ip102"

TRAIN_TXT = os.path.join(DATA_ROOT, "train.txt")
VAL_TXT   = os.path.join(DATA_ROOT, "val.txt")

TRAIN_DIR = os.path.join(DATA_ROOT, "classification/train")
VAL_DIR   = os.path.join(DATA_ROOT, "classification/val")

NUM_CLASSES = 102
BATCH_SIZE = 8
EPOCHS = 5
WORKERS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOGDIR = "logs/ip102"

writer = SummaryWriter(LOGDIR)


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

st.title("IP102 Insect Classifier — MobileNetV2")

if "training" not in st.session_state:
    st.session_state.training = False

if st.button("Start Training"):
    st.session_state.training = True

progress_bar = st.progress(0)

loss_chart = st.empty()
stats_box = st.empty()
speed_box = st.empty()


# --------------------------------------------------
# TRANSFORMS
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


# --------------------------------------------------
# DATASET
# --------------------------------------------------

class IP102Dataset(Dataset):

    def __init__(self, txt_file, img_root, transform=None):

        self.samples=[]
        self.img_root=img_root
        self.transform=transform

        with open(txt_file) as f:
            lines=f.readlines()

        for line in lines:

            img_name,label=line.strip().split()
            label=int(label)

            img_path=os.path.join(
                img_root,
                str(label),
                img_name
            )

            self.samples.append((img_path,label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):

        img_path,label=self.samples[idx]

        image=Image.open(img_path).convert("RGB")

        if self.transform:
            image=self.transform(image)

        return image,label


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    train_ds = IP102Dataset(TRAIN_TXT, TRAIN_DIR, transform)
    val_ds   = IP102Dataset(VAL_TXT, VAL_DIR, transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True
    )

    return train_loader,val_loader,len(train_ds),len(val_ds)

train_loader,val_loader,train_size,val_size = load_data()

st.write("Train samples:", train_size)
st.write("Validation samples:", val_size)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

def load_model():

    model=models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT
    )

    model.classifier[1]=nn.Linear(
        model.last_channel,
        NUM_CLASSES
    )

    return model.to(device)

model = load_model()

st.write("Device:", device)


# --------------------------------------------------
# TRAINING SETUP
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=7,
    gamma=0.1
)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate(model, loader):

    model.eval()

    preds_all=[]
    labels_all=[]

    with torch.no_grad():

        for imgs,labels in loader:

            imgs=imgs.to(device)

            outputs=model(imgs)

            preds=outputs.argmax(dim=1).cpu().numpy()

            preds_all.extend(preds)
            labels_all.extend(labels.numpy())

    acc=np.mean(
        np.array(preds_all)==np.array(labels_all)
    )

    return acc


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

if st.session_state.training:

    losses=[]

    for epoch in range(EPOCHS):

        model.train()

        running_loss=0
        t0=time.time()

        for batch_idx,(imgs,labels) in enumerate(train_loader):

            imgs=imgs.to(device)
            labels=labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs=model(imgs)

            loss=criterion(outputs,labels)

            loss.backward()

            optimizer.step()

            running_loss+=loss.item()

        scheduler.step()

        losses.append(running_loss)

        val_acc = evaluate(model,val_loader)

        # Streamlit updates
        progress_bar.progress((epoch+1)/EPOCHS)

        stats_box.write(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"loss={running_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"time={time.time()-t0:.1f}s"
        )

        # loss chart
        import matplotlib.pyplot as plt

        fig=plt.figure()

        plt.plot(losses,label="train_loss")
        plt.legend()
        plt.title("Training Loss")

        loss_chart.pyplot(fig)

        # TensorBoard
        writer.add_scalar("Loss/train",running_loss,epoch)
        writer.add_scalar("Accuracy/val",val_acc,epoch)

    writer.close()

    # --------------------------------------------------
    # MODEL SIZE
    # --------------------------------------------------

    param_count = sum(p.numel() for p in model.parameters())

    st.write("Parameters:", f"{param_count:,}")

    # --------------------------------------------------
    # INFERENCE SPEED TEST
    # --------------------------------------------------

    model.eval()

    dummy = torch.randn(1,3,224,224).to(device)

    for _ in range(10):
        model(dummy)

    N = 100

    t0 = time.time()

    for _ in range(N):
        model(dummy)

    avg = (time.time()-t0)/N

    speed_box.write(
        f"Inference time: {avg*1000:.2f} ms"
    )

    st.success("Training and evaluation complete.")
    st.session_state.training = False