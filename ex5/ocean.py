# ocean_listener_dashboard.py

import os
import numpy as np
import librosa
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from skimage.transform import resize


# -------------------------------------------------
# PARAMETERS
# -------------------------------------------------

DATASET_PATH = "ocean_dataset"
SAMPLE_RATE = 22050
SEGMENT_DURATION = 3
SAMPLES_PER_SEGMENT = SAMPLE_RATE * SEGMENT_DURATION
IMG_SIZE = (128,128)

LOGDIR = "logs/ocean_listener"


# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------

st.title("Ocean Listener — Underwater Bioacoustic Classifier")

train_button = st.button("Start Training")

progress_bar = st.progress(0)

loss_chart = st.empty()
acc_chart = st.empty()
cm_chart = st.empty()

metrics_text = st.empty()


# -------------------------------------------------
# LABEL MAPPING
# -------------------------------------------------

classes = sorted(os.listdir(DATASET_PATH))
class_to_index = {c:i for i,c in enumerate(classes)}

st.write("Detected classes:", classes)


# -------------------------------------------------
# FEATURE EXTRACTION
# -------------------------------------------------

def extract_segments(file_path):

    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    segments=[]

    for start in range(0,len(signal),SAMPLES_PER_SEGMENT):

        end=start+SAMPLES_PER_SEGMENT

        if len(signal[start:end])==SAMPLES_PER_SEGMENT:

            mel=librosa.feature.melspectrogram(
                y=signal[start:end],
                sr=sr,
                n_mels=128,
                n_fft=1024,
                hop_length=512
            )

            mel_db=librosa.power_to_db(mel,ref=np.max)

            mel_db=(mel_db-mel_db.min())/(mel_db.max()-mel_db.min()+1e-6)

            mel_img=resize(mel_db,IMG_SIZE)

            segments.append(mel_img)

    return segments


# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------

@st.cache_data
def load_dataset():

    X=[]
    y=[]

    for cls in classes:

        folder=os.path.join(DATASET_PATH,cls)

        for file in os.listdir(folder):

            path=os.path.join(folder,file)

            try:

                segs=extract_segments(path)

                for s in segs:

                    X.append(s)
                    y.append(class_to_index[cls])

            except:
                continue

    X=np.array(X)
    y=np.array(y)

    X=X.reshape(-1,128,128,1)

    return X,y


# -------------------------------------------------
# DILATION CNN
# -------------------------------------------------

def dilation_block(x,filters,dilation):

    x=layers.Conv2D(
        filters,
        (3,3),
        padding="same",
        dilation_rate=dilation,
        activation="relu")(x)

    x=layers.BatchNormalization()(x)

    return x


def build_model():

    inp=layers.Input(shape=(128,128,1))

    x=dilation_block(inp,32,1)
    x=dilation_block(x,32,2)
    x=layers.MaxPool2D()(x)

    x=dilation_block(x,64,1)
    x=dilation_block(x,64,4)
    x=layers.MaxPool2D()(x)

    x=dilation_block(x,96,1)

    x=layers.GlobalAveragePooling2D()(x)

    x=layers.Dense(192,activation="relu")(x)
    x=layers.Dropout(0.4)(x)

    out=layers.Dense(3,activation="softmax")(x)

    model=models.Model(inp,out)

    return model


# -------------------------------------------------
# STREAMLIT CALLBACK
# -------------------------------------------------

class StreamlitCallback(tf.keras.callbacks.Callback):

    def __init__(self,epochs):

        self.epochs=epochs

        self.acc=[]
        self.val_acc=[]
        self.loss=[]
        self.val_loss=[]

    def on_epoch_end(self,epoch,logs=None):

        self.acc.append(logs["accuracy"])
        self.val_acc.append(logs["val_accuracy"])

        self.loss.append(logs["loss"])
        self.val_loss.append(logs["val_loss"])

        progress_bar.progress((epoch+1)/self.epochs)

        # Accuracy plot
        fig1=plt.figure()

        plt.plot(self.acc)
        plt.plot(self.val_acc)

        plt.legend(["Train","Validation"])
        plt.title("Accuracy")

        acc_chart.pyplot(fig1)

        # Loss plot
        fig2=plt.figure()

        plt.plot(self.loss)
        plt.plot(self.val_loss)

        plt.legend(["Train","Validation"])
        plt.title("Loss")

        loss_chart.pyplot(fig2)


# -------------------------------------------------
# TRAINING
# -------------------------------------------------

if train_button:

    X,y=load_dataset()

    st.write("Dataset shape:",X.shape)

    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,random_state=42,stratify=y
    )

    model=build_model()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # TensorBoard
    tb=tf.keras.callbacks.TensorBoard(LOGDIR)

    streamlit_cb=StreamlitCallback(epochs=50)

    history=model.fit(

        X_train,
        y_train,

        epochs=50,
        batch_size=32,

        validation_split=0.2,

        callbacks=[
            tb,
            streamlit_cb,
            tf.keras.callbacks.EarlyStopping(
                patience=8,
                restore_best_weights=True
            )
        ]
    )

    # -------------------------------------------------
    # EVALUATION
    # -------------------------------------------------

    preds=np.argmax(model.predict(X_test),axis=1)

    cm=confusion_matrix(y_test,preds)

    fig=plt.figure()

    disp=ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes
    )

    disp.plot(cmap="Blues")

    cm_chart.pyplot(fig)

    metrics_text.write("Testing completed.")