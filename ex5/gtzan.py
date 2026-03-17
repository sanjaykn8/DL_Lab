# gtzan_streamlit_dashboard.py

import os
import numpy as np
import librosa
import tensorflow as tf
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
from skimage.transform import resize


# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------

DATA_ROOT = "data/gtzan/genres_original/"
SR = 22050
SEGMENT_DURATION = 3
IMG_SIZE = (128,128)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
LOGDIR = "logs/gtzan"

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

st.title("GTZAN Music Genre Classifier")

train_button = st.button("Start Training")

progress_bar = st.progress(0)

loss_chart = st.empty()
acc_chart = st.empty()
cm_chart = st.empty()

metrics_text = st.empty()

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_files():

    genres = sorted(os.listdir(DATA_ROOT))
    genre_to_index = {g:i for i,g in enumerate(genres)}

    filepaths=[]
    labels=[]

    for genre in genres:

        genre_folder=os.path.join(DATA_ROOT,genre)

        for file in os.listdir(genre_folder):

            if file.endswith(".wav"):

                filepaths.append(os.path.join(genre_folder,file))
                labels.append(genre_to_index[genre])

    return np.array(filepaths),np.array(labels),genres


filepaths,labels,genres = load_files()

X_train,X_temp,y_train,y_temp = train_test_split(
    filepaths,labels,
    test_size=0.3,
    stratify=labels,
    random_state=42
)

X_val,X_test,y_val,y_test = train_test_split(
    X_temp,y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)

# --------------------------------------------------
# AUDIO → MEL SEGMENTS
# --------------------------------------------------

def wav_to_segments(path,label):

    segments=[]
    labels_out=[]

    path=path.numpy().decode("utf-8")

    try:
        y,sr=librosa.load(path,sr=SR)
    except:
        return np.zeros((0,128,128),dtype=np.float32),np.zeros((0,),dtype=np.int64)

    samples_per_segment = SR*SEGMENT_DURATION
    num_segments = int(len(y)//samples_per_segment)

    for i in range(num_segments):

        start=i*samples_per_segment
        end=start+samples_per_segment

        segment=y[start:end]

        if len(segment)<samples_per_segment:
            continue

        try:

            mel=librosa.feature.melspectrogram(
                y=segment,
                sr=sr,
                n_mels=128,
                n_fft=1024,
                hop_length=512
            )

            mel_db=librosa.power_to_db(mel,ref=np.max)

            mel_db=(mel_db-mel_db.min())/(mel_db.max()-mel_db.min()+1e-6)

            mel_resized=resize(
                mel_db,
                IMG_SIZE,
                mode="reflect",
                anti_aliasing=True
            )

            segments.append(mel_resized.astype(np.float32))
            labels_out.append(label.numpy())

        except:
            continue

    if len(segments)==0:
        return np.empty((0,128,128),dtype=np.float32),np.empty((0,),dtype=np.int64)

    return np.array(segments),np.array(labels_out)

# --------------------------------------------------
# TF DATA PIPELINE
# --------------------------------------------------

def create_dataset(paths,labels):

    ds=tf.data.Dataset.from_tensor_slices((paths,labels))

    def process(path,label):

        segs,labs=tf.py_function(
            wav_to_segments,
            [path,label],
            [tf.float32,tf.int64]
        )

        segs.set_shape([None,128,128])
        labs.set_shape([None])

        segs=tf.expand_dims(segs,-1)

        return tf.data.Dataset.from_tensor_slices((segs,labs))

    ds=ds.flat_map(process)
    ds=ds.shuffle(2000)
    ds=ds.batch(BATCH_SIZE)
    ds=ds.prefetch(AUTOTUNE)

    return ds


# --------------------------------------------------
# MODEL
# --------------------------------------------------

def build_model():

    inputs=tf.keras.layers.Input(shape=(128,128,1))

    x=tf.keras.layers.Conv2D(32,3,padding='same',activation='relu')(inputs)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.MaxPool2D()(x)
    x=tf.keras.layers.Dropout(0.2)(x)

    x=tf.keras.layers.Conv2D(64,3,padding='same',activation='relu')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.MaxPool2D()(x)
    x=tf.keras.layers.Dropout(0.25)(x)

    x=tf.keras.layers.Conv2D(128,3,padding='same',activation='relu')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.MaxPool2D()(x)
    x=tf.keras.layers.Dropout(0.3)(x)

    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    x=tf.keras.layers.Dense(256,activation='relu')(x)
    x=tf.keras.layers.Dropout(0.4)(x)

    outputs=tf.keras.layers.Dense(10,activation='softmax')(x)

    return tf.keras.Model(inputs,outputs)

# --------------------------------------------------
# STREAMLIT TRAINING CALLBACK
# --------------------------------------------------

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

        # accuracy plot
        fig1=plt.figure()
        plt.plot(self.acc)
        plt.plot(self.val_acc)
        plt.legend(["Train","Val"])
        plt.title("Accuracy")
        acc_chart.pyplot(fig1)

        # loss plot
        fig2=plt.figure()
        plt.plot(self.loss)
        plt.plot(self.val_loss)
        plt.legend(["Train","Val"])
        plt.title("Loss")
        loss_chart.pyplot(fig2)

# --------------------------------------------------
# TRAIN
# --------------------------------------------------

if train_button:

    train_ds=create_dataset(X_train,y_train)
    val_ds=create_dataset(X_val,y_val)
    test_ds=create_dataset(X_test,y_test)

    model=build_model()

    tensorboard=tf.keras.callbacks.TensorBoard(LOGDIR)

    streamlit_cb=StreamlitCallback(epochs=10)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[
            tensorboard,
            streamlit_cb,
            tf.keras.callbacks.EarlyStopping(patience=7)
        ]
    )

    # --------------------------------------------------
    # TEST EVALUATION
    # --------------------------------------------------

    y_true=[]
    y_pred=[]

    for x,y in test_ds:

        preds=model.predict(x)

        y_pred.extend(np.argmax(preds,axis=1))
        y_true.extend(y.numpy())

    cm=confusion_matrix(y_true,y_pred)

    fig=plt.figure()

    disp=ConfusionMatrixDisplay(cm,display_labels=genres)
    disp.plot(cmap="Blues",xticks_rotation=45)

    cm_chart.pyplot(fig)

    metrics_text.write("Testing completed.")
