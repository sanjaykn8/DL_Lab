# efficientnet_cow_dashboard.py

import torch
import torch.nn as nn
import time
import streamlit as st
import matplotlib.pyplot as plt

from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter


# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------

DATA_DIR = "cow"
NUM_CLASSES = 5
BATCH_SIZE = 8
EPOCHS = 5
VAL_SPLIT = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOGDIR = "logs/cow_classifier"

writer = SummaryWriter(LOGDIR)


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

st.title("EfficientNet Cow Breed Classifier")

train_button = st.button("Start Training")

progress_bar = st.progress(0)

loss_chart = st.empty()
acc_chart = st.empty()

stats_box = st.empty()


# --------------------------------------------------
# TRANSFORMS
# --------------------------------------------------

train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# --------------------------------------------------
# DATASET
# --------------------------------------------------

full_dataset = datasets.ImageFolder(DATA_DIR)

train_size = int((1-VAL_SPLIT)*len(full_dataset))
val_size   = len(full_dataset) - train_size

indices = torch.randperm(len(full_dataset)).tolist()

train_indices = indices[:train_size]
val_indices   = indices[train_size:]

train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tf)
val_dataset   = datasets.ImageFolder(DATA_DIR, transform=val_tf)

train_ds = torch.utils.data.Subset(train_dataset, train_indices)
val_ds   = torch.utils.data.Subset(val_dataset, val_indices)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

st.write("Classes:", full_dataset.classes)
st.write("Train samples:", train_size)
st.write("Validation samples:", val_size)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

def load_model():

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, NUM_CLASSES)
    )

    return model.to(device)

model = load_model()


# --------------------------------------------------
# TRAINING SETUP
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.1
)

import gc

gc.collect()
torch.cuda.empty_cache()
# --------------------------------------------------
# TRAINING
# --------------------------------------------------

if train_button:

    train_losses,val_losses = [],[]
    train_accs,val_accs = [],[]

    for epoch in range(EPOCHS):

        t0 = time.time()

        # ---------------- TRAIN ----------------
        model.train()

        running_loss=0
        preds_all=[]
        labels_all=[]

        for imgs,labels in train_loader:

            imgs=imgs.to(device)
            labels=labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs=model(imgs)

            loss=criterion(outputs,labels)

            loss.backward()

            optimizer.step()

            running_loss+=loss.item()*imgs.size(0)

            preds=outputs.argmax(1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

        train_loss=running_loss/train_size
        train_acc=accuracy_score(labels_all,preds_all)

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ---------------- VALIDATION ----------------
        model.eval()

        running_loss=0
        preds_all=[]
        labels_all=[]

        with torch.no_grad():

            for imgs,labels in val_loader:

                imgs=imgs.to(device)
                labels=labels.to(device)

                outputs=model(imgs)

                loss=criterion(outputs,labels)

                running_loss+=loss.item()*imgs.size(0)

                preds=outputs.argmax(1)

                preds_all.extend(preds.cpu().numpy())
                labels_all.extend(labels.cpu().numpy())

        val_loss=running_loss/val_size
        val_acc=accuracy_score(labels_all,preds_all)

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step()

        # ---------------- STREAMLIT UI ----------------

        progress_bar.progress((epoch+1)/EPOCHS)

        stats_box.write(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # Loss chart
        fig1=plt.figure()

        plt.plot(train_losses,label="train_loss")
        plt.plot(val_losses,label="val_loss")

        plt.title("Loss")
        plt.legend()

        loss_chart.pyplot(fig1)

        # Accuracy chart
        fig2=plt.figure()

        plt.plot(train_accs,label="train_acc")
        plt.plot(val_accs,label="val_acc")

        plt.title("Accuracy")
        plt.legend()

        acc_chart.pyplot(fig2)

        # ---------------- TensorBoard ----------------

        writer.add_scalar("Loss/train",train_loss,epoch)
        writer.add_scalar("Loss/val",val_loss,epoch)

        writer.add_scalar("Accuracy/train",train_acc,epoch)
        writer.add_scalar("Accuracy/val",val_acc,epoch)

    writer.close()

    st.success("Training completed.")