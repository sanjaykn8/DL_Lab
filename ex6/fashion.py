# googlenet_fashion_dashboard.py

import torch
import torch.nn as nn
import streamlit as st
import time
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------

BATCH_SIZE = 8
EPOCHS = 5
NUM_CLASSES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOGDIR = "logs/fashion_googlenet"

writer = SummaryWriter(LOGDIR)

torch.backends.cudnn.benchmark = True


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

st.title("Fashion-MNIST Classifier — GoogLeNet")

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
    transforms.Grayscale(3),  # convert 1-channel → 3-channel
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
])


# --------------------------------------------------
# DATASETS
# --------------------------------------------------

train_ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=train_tf
)

val_ds = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=val_tf
)

train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
val_loader   = DataLoader(val_ds,batch_size=BATCH_SIZE)

st.write("Train samples:", len(train_ds))
st.write("Validation samples:", len(val_ds))


# --------------------------------------------------
# MODEL
# --------------------------------------------------

def load_model():

    model = models.googlenet(
        weights=models.GoogLeNet_Weights.DEFAULT
    )

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model.to(device)

model = load_model()

st.write("Device:", device)


# --------------------------------------------------
# TRAINING SETUP
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate():

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for imgs,labels in val_loader:

            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)

            preds = outputs.argmax(1)

            correct += (preds==labels).sum().item()
            total += labels.size(0)

    return correct/total


# --------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------

if train_button:

    train_losses = []
    val_accs = []

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0
        t0 = time.time()

        for imgs,labels in train_loader:

            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(imgs)

            loss = criterion(outputs,labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss)

        val_acc = evaluate()

        val_accs.append(val_acc)

        # ---------------- STREAMLIT ----------------

        progress_bar.progress((epoch+1)/EPOCHS)

        stats_box.write(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"loss={running_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"time={time.time()-t0:.1f}s"
        )

        # Loss plot
        fig1 = plt.figure()

        plt.plot(train_losses,label="train_loss")
        plt.legend()
        plt.title("Training Loss")

        loss_chart.pyplot(fig1)

        # Accuracy plot
        fig2 = plt.figure()

        plt.plot(val_accs,label="val_accuracy")
        plt.legend()
        plt.title("Validation Accuracy")

        acc_chart.pyplot(fig2)

        # ---------------- TensorBoard ----------------

        writer.add_scalar("Loss/train",running_loss,epoch)
        writer.add_scalar("Accuracy/val",val_acc,epoch)

    writer.close()

    st.success("Training finished.")