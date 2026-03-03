# app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import time

st.set_page_config(layout="wide", page_title="Keras CIFAR10 Trainer")

# -------------------------
# Model
# -------------------------
def build_model(input_shape=(32,32,3), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


# -------------------------
# Plot helper
# -------------------------
def plot_history_from_lists(train_loss, val_loss, train_acc, val_acc):
    fig, axes = plt.subplots(1,2, figsize=(12,4))

    axes[0].plot(train_loss, label='train loss')
    if len(val_loss) > 0:
        axes[0].plot(val_loss, label='val loss')
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(train_acc, label='train acc')
    if len(val_acc) > 0:
        axes[1].plot(val_acc, label='val acc')
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    return fig


# -------------------------
# First layer visualization
# -------------------------
def visualize_first_layer(model, x_sample):

    first_conv = None
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            first_conv = layer
            break

    activation_model = models.Model(
        inputs=model.inputs,
        outputs=first_conv.output
    )

    activations = activation_model.predict(
        np.expand_dims(x_sample, 0),
        verbose=0
    )[0]

    n_maps = activations.shape[-1]
    cols = 8
    rows = int(np.ceil(n_maps / cols))

    fig = plt.figure(figsize=(cols * 1.5, rows * 1.5))

    for i in range(n_maps):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(activations[:, :, i], cmap="viridis")
        plt.axis("off")

    plt.tight_layout()
    st.pyplot(fig)


# -------------------------
# Streamlit Callback
# -------------------------
class StreamlitCallback(tf.keras.callbacks.Callback):

    def __init__(self, progress_bar, status_box, metrics_area, total_epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_box = status_box
        self.metrics_area = metrics_area
        self.total_epochs = int(total_epochs)

        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        acc = logs.get("accuracy")
        val_acc = logs.get("val_accuracy")

        if loss is not None:
            self.train_loss.append(loss)
        if val_loss is not None:
            self.val_loss.append(val_loss)
        if acc is not None:
            self.train_acc.append(acc)
        if val_acc is not None:
            self.val_acc.append(val_acc)

        progress_value = float(epoch + 1) / float(self.total_epochs)
        self.progress_bar.progress(progress_value)

        status_text = f"Epoch {epoch+1}/{self.total_epochs}"
        if loss is not None:
            status_text += f" | loss={loss:.4f}"
        if val_loss is not None:
            status_text += f" | val_loss={val_loss:.4f}"
        if acc is not None:
            status_text += f" | acc={acc:.4f}"
        if val_acc is not None:
            status_text += f" | val_acc={val_acc:.4f}"

        self.status_box.info(status_text)

        fig = plot_history_from_lists(
            self.train_loss,
            self.val_loss,
            self.train_acc,
            self.val_acc
        )

        with self.metrics_area:
            st.subheader("Live Metrics")
            st.pyplot(fig)


# -------------------------
# UI Layout
# -------------------------
st.title("Keras CIFAR-10 CNN Trainer (Live Training)")

left, right = st.columns([1,2])

with left:
    st.header("Hyperparameters")
    epochs = st.number_input("Epochs", 1, 100, 10)
    batch_size = st.selectbox("Batch Size", [32, 64, 128], index=1)
    val_split = st.slider("Validation Split", 0.05, 0.3, 0.1)
    start_training = st.button("Start Training")

with right:
    status = st.empty()
    progress = st.progress(0.0)
    metrics_area = st.empty()


# -------------------------
# Training Block
# -------------------------
if start_training:

    status.info("Loading CIFAR-10 dataset...")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = build_model()
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    status.info("Training started...")
    start_time = time.time()

    callback = StreamlitCallback(
        progress_bar=progress,
        status_box=status,
        metrics_area=metrics_area,
        total_epochs=epochs
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=int(epochs),
        batch_size=int(batch_size),
        validation_split=float(val_split),
        verbose=0,
        callbacks=[callback]
    )

    total_time = time.time() - start_time
    status.success(f"Training completed in {int(total_time)} seconds")

    # Final curves
    st.subheader("Final Training Curves")
    final_fig = plot_history_from_lists(
        history.history.get("loss", []),
        history.history.get("val_loss", []),
        history.history.get("accuracy", []),
        history.history.get("val_accuracy", [])
    )
    st.pyplot(final_fig)

    # Evaluation
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    st.write({
        "Test Accuracy": f"{test_acc:.4f}",
        "Test Loss": f"{test_loss:.4f}"
    })

    # Activation visualization
    st.subheader("First Conv Layer Activations (Test Image 0)")
    visualize_first_layer(model, x_test[0])

    # Sample prediction
    st.subheader("Sample Prediction")

    sample = x_test[0]
    pred = model.predict(np.expand_dims(sample, 0), verbose=0)
    predicted_class = np.argmax(pred)

    class_names = [
        "airplane","automobile","bird","cat","deer",
        "dog","frog","horse","ship","truck"
    ]

    st.image(sample, caption=f"Predicted: {class_names[predicted_class]}", width=200)