# app.py

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import datetime
import os

# -----------------------
# Config
# -----------------------
IMG_SIZE = (224,224)
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

st.set_page_config(layout="wide", page_title="Transfer Learning vs Scratch")

# -----------------------
# Models
# -----------------------
def build_transfer_model(input_shape=IMG_SIZE+(3,)):
    base = tf.keras.applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base.trainable = False

    x = layers.Flatten()(base.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base.input, outputs=out)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_small_cnn(input_shape=IMG_SIZE+(3,)):
    model = models.Sequential([
        layers.Conv2D(32,3,activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64,3,activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128,3,activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# -----------------------
# Data
# -----------------------
@st.cache_resource
def get_generators(batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=20
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='binary'
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_gen, val_gen

# -----------------------
# Plot
# -----------------------
def plot_both(hist1, hist2):
    fig, axes = plt.subplots(1,2, figsize=(12,4))

    axes[0].plot(hist1.history['accuracy'], label='TL train')
    axes[0].plot(hist1.history['val_accuracy'], label='TL val')
    axes[0].set_title("Transfer Learning Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(hist2.history['accuracy'], label='Scratch train')
    axes[1].plot(hist2.history['val_accuracy'], label='Scratch val')
    axes[1].set_title("Scratch Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    st.pyplot(fig)

# -----------------------
# UI
# -----------------------
st.title("Transfer Learning (VGG16) vs Scratch CNN")

left, right = st.columns([1,2])

with left:
    st.header("Settings")
    batch_size = st.selectbox("Batch Size", [8,16,32], index=1)
    epochs = st.number_input("Epochs", 1, 50, 5)
    start = st.button("Start Training")

with right:
    status = st.empty()
    progress = st.progress(0.0)

# -----------------------
# Training
# -----------------------
if start:

    log_root = "logs"
    os.makedirs(log_root, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    status.info("Loading data...")
    train_gen, val_gen = get_generators(batch_size)

    # TensorBoard callbacks
    tb_tl = TensorBoard(
        log_dir=f"{log_root}/transfer_{run_id}",
        histogram_freq=1
    )

    tb_scratch = TensorBoard(
        log_dir=f"{log_root}/scratch_{run_id}",
        histogram_freq=1
    )

    # -------------------
    # Transfer Learning
    # -------------------
    status.info("Training Transfer Learning model...")
    tl_model = build_transfer_model()

    hist_tl = tl_model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        verbose=1,
        callbacks=[tb_tl]
    )

    progress.progress(0.5)

    # -------------------
    # Scratch Model
    # -------------------
    status.info("Training Scratch model...")
    scratch = build_small_cnn()

    hist_scratch = scratch.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        verbose=1,
        callbacks=[tb_scratch]
    )

    progress.progress(1.0)

    status.success("Training completed.")

    st.subheader("Accuracy Comparison")
    plot_both(hist_tl, hist_scratch)

    st.markdown("### TensorBoard")
    st.code("tensorboard --logdir logs/")
    st.write("Transfer logs:", f"logs/transfer_{run_id}")
    st.write("Scratch logs:", f"logs/scratch_{run_id}")