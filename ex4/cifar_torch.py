# app.py
import io
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------
# Basic config / device
# -----------------------
st.set_page_config(layout="wide", page_title="CIFAR10 Trainer")
device_auto = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Model definition
# -----------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128*4*4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.pool(x)
        x = F.relu(self.conv2(x)); x = self.pool(x)
        x = F.relu(self.conv3(x)); x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------
# Helpers
# -----------------------
@st.cache_resource
def load_datasets(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, testloader, trainset, testset

def evaluate(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loss_total += loss.item() * labels.size(0)
    acc = correct / total
    avg_loss = loss_total / total
    return acc, avg_loss

def plot_history(train_acc, val_acc, train_loss, val_loss):
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    axes[0].plot(train_acc, label='train_acc'); axes[0].plot(val_acc, label='val_acc')
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(train_loss, label='train_loss'); axes[1].plot(val_loss, label='val_loss')
    axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(True)
    st.pyplot(fig)

def img_tensor_to_numpy(img_tensor):
    # unnormalize for display: input assumed normalized with mean=0.5, std=0.5
    img = img_tensor.cpu().numpy().transpose(1,2,0)
    img = (img * 0.5) + 0.5  # undo normalization
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

# -----------------------
# UI layout
# -----------------------
st.title("CIFAR-10 Trainer")
left, right = st.columns([1,2])

with left:
    st.header("Hyperparameters")
    epochs = st.number_input("Epochs", min_value=1, max_value=200, value=8, step=1)
    batch_size = st.selectbox("Batch size", options=[32,64,128,256], index=2)
    lr = st.number_input("Learning rate", value=1e-3, format="%.6f")
    momentum = st.number_input("Adam betas (momentum-like) - only beta1", value=0.9, format="%.3f")
    use_gpu = st.checkbox(f"Use GPU (detected: {device_auto})", value=(device_auto=="cuda"))
    seed = st.number_input("Random seed", value=42, step=1)
    st.markdown("---")
    start_btn = st.button("Start training")
    st.markdown("After training completes you can download the trained model.")

with right:
    st.header("Live outputs")
    status_text = st.empty()
    progress_bar = st.progress(0.0)
    metrics_col = st.empty()
    sample_preds_col = st.empty()

# -----------------------
# Training action
# -----------------------
if start_btn:
    st.session_state.setdefault("training_running", True)
    torch.manual_seed(int(seed))

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(momentum, 0.999))

    trainloader, testloader, trainset, testset = load_datasets(batch_size)

    train_acc_hist = []
    val_acc_hist = []
    train_loss_hist = []
    val_loss_hist = []

    total_epochs = int(epochs)
    status_text.info("Training started.")
    start_time = time.time()

    for epoch in range(total_epochs):
        epoch_start = time.time()
        model.train()
        running_correct = 0
        running_total = 0
        running_loss = 0.0

        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = outputs.max(1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item() * labels.size(0)

        epoch_train_acc = running_correct / running_total
        epoch_train_loss = running_loss / running_total
        train_acc_hist.append(epoch_train_acc)
        train_loss_hist.append(epoch_train_loss)

        # validation
        val_acc, val_loss = evaluate(model, testloader, criterion, device)
        val_acc_hist.append(val_acc)
        val_loss_hist.append(val_loss)

        # UI updates
        elapsed = time.time() - start_time
        status_text.info(f"Epoch {epoch+1}/{total_epochs} — train_acc={epoch_train_acc:.4f} val_acc={val_acc:.4f} — elapsed {int(elapsed)}s")
        progress_bar.progress((epoch+1)/total_epochs)

        # show metrics and plots
        with metrics_col:
            st.subheader("Metrics")
            st.write({
                "epoch": epoch+1,
                "train_acc": f"{epoch_train_acc:.4f}",
                "val_acc": f"{val_acc:.4f}",
                "train_loss": f"{epoch_train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}"
            })
            plot_history(train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist)

        # show a few sample predictions from test set
        with sample_preds_col:
            st.subheader("Sample predictions")
            images, labels = next(iter(testloader))
            images_cpu = images[:8]
            labels_cpu = labels[:8]
            images_device = images_cpu.to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(images_device)
                _, preds = outputs.max(1)
            preds = preds.cpu().numpy()
            fig, axs = plt.subplots(2,4, figsize=(12,6))
            class_names = trainset.classes
            for idx, ax in enumerate(axs.flatten()):
                if idx >= len(images_cpu): break
                img_np = img_tensor_to_numpy(images_cpu[idx])
                ax.imshow(img_np)
                ax.set_title(f"pred: {class_names[preds[idx]]}\ntrue: {class_names[int(labels_cpu[idx])]}")
                ax.axis('off')
            st.pyplot(fig)

        # small delay to allow UI to render smoothly
        time.sleep(0.2)

    total_time = time.time() - start_time
    status_text.success(f"Training finished in {int(total_time)}s. Final val_acc={val_acc_hist[-1]:.4f}")
    st.session_state["trained_model_state"] = model.state_dict()
    st.session_state["trained_model_classnames"] = trainset.classes

    # model download
    buffer = io.BytesIO()
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": trainset.classes,
        "hyperparams": {"epochs": total_epochs, "batch_size": batch_size, "lr": lr}
    }, buffer)
    buffer.seek(0)
    st.download_button("Download trained model (.pt)", data=buffer, file_name="simplecnn_cifar10.pt", mime="application/octet-stream")

    # Final plots
    st.subheader("Final training curves")
    plot_history(train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist)
    st.balloons()