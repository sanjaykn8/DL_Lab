# heartbeat_dashboard.py

import os
import time
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard, Callback
import seaborn as sns


# ----------------------------------
# PARAMETERS
# ----------------------------------

DATA_ROOT = "data/heartbeat"
IMG_SIZE = (128,128)
SR = 22050
LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)


# ----------------------------------
# STREAMLIT UI
# ----------------------------------

st.title("Heartbeat Sound Classification")

train_button = st.button("Start Training")

progress_bar = st.progress(0)

loss_chart = st.empty()
acc_chart = st.empty()
cm_chart = st.empty()

metrics_text = st.empty()


# ----------------------------------
# LOAD DATA
# ----------------------------------

@st.cache_data
def load_dataset():

    df_a = pd.read_csv(os.path.join(DATA_ROOT,"set_a.csv"))
    df_b = pd.read_csv(os.path.join(DATA_ROOT,"set_b.csv"))

    df = pd.concat([df_a,df_b], ignore_index=True)
    df = df[df["label"].notna()]

    df["binary_label"] = df["label"].apply(
        lambda x: 0 if x=="normal" else 1
    )

    all_files = {}

    for root, dirs, files in os.walk(DATA_ROOT):
        for f in files:
            if f.endswith(".wav"):
                all_files[f] = os.path.join(root,f)

    paths = []
    labels = []

    for i,row in df.iterrows():

        fname = os.path.basename(row["fname"])

        if fname in all_files:
            paths.append(all_files[fname])
            labels.append(row["binary_label"])

    return paths,labels


# ----------------------------------
# AUDIO → MEL
# ----------------------------------

def audio_to_mel(path):

    signal, sr = librosa.load(path, sr=SR)

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )

    mel = librosa.power_to_db(mel)

    mel = np.resize(mel, IMG_SIZE)

    return mel


# ----------------------------------
# CUSTOM STREAMLIT CALLBACK
# ----------------------------------

class StreamlitCallback(Callback):

    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.acc=[]
        self.val_acc=[]
        self.loss=[]
        self.val_loss=[]

    def on_epoch_end(self, epoch, logs=None):

        self.acc.append(logs["accuracy"])
        self.val_acc.append(logs["val_accuracy"])
        self.loss.append(logs["loss"])
        self.val_loss.append(logs["val_loss"])

        progress_bar.progress((epoch+1)/self.epochs)

        # accuracy plot
        fig1 = plt.figure()
        plt.plot(self.acc)
        plt.plot(self.val_acc)
        plt.legend(["Train","Val"])
        plt.title("Accuracy")
        acc_chart.pyplot(fig1)

        # loss plot
        fig2 = plt.figure()
        plt.plot(self.loss)
        plt.plot(self.val_loss)
        plt.legend(["Train","Val"])
        plt.title("Loss")
        loss_chart.pyplot(fig2)


# ----------------------------------
# TRAIN PIPELINE
# ----------------------------------

if train_button:

    paths,labels = load_dataset()

    data=[]
    y=[]

    for i,path in enumerate(paths):

        mel = audio_to_mel(path)

        data.append(mel)
        y.append(labels[i])

    X=np.array(data)
    y=np.array(y)

    X = X.reshape(-1,128,128,1)

    X = X/(np.max(X)+1e-6)

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )


    # -------------------------
    # MODEL
    # -------------------------

    model=models.Sequential([

        layers.Conv2D(16,(3,3),activation='relu',input_shape=(128,128,1)),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(32,(3,3),activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),

        layers.Dense(64,activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(1,activation='sigmoid')
    ])


    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )


    # -------------------------
    # CALLBACKS
    # -------------------------

    tensorboard = TensorBoard(
        log_dir=LOG_DIR,
        histogram_freq=1
    )

    streamlit_callback = StreamlitCallback(epochs=10)


    # -------------------------
    # TRAIN
    # -------------------------

    history=model.fit(

        X_train,
        y_train,

        epochs=10,
        batch_size=32,

        validation_split=0.2,

        callbacks=[
            tensorboard,
            streamlit_callback
        ]
    )


    # -------------------------
    # EVALUATION
    # -------------------------

    y_pred=(model.predict(X_test)>0.5).astype(int)

    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)

    metrics_text.write(
        f"Accuracy: {acc:.3f}  |  F1 Score: {f1:.3f}"
    )


    # -------------------------
    # CONFUSION MATRIX
    # -------------------------

    cm=confusion_matrix(y_test,y_pred)

    fig=plt.figure()

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap="Blues",
        xticklabels=["Normal","Abnormal"],
        yticklabels=["Normal","Abnormal"]
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_chart.pyplot(fig)


    # -------------------------
    # CLASSIFICATION REPORT
    # -------------------------

    report = classification_report(y_test,y_pred)

    st.text(report)
