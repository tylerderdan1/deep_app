import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import cv2

st.set_page_config(page_title="Fashion MNIST Classifier", layout="wide")

# Label names
labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Load Model
@st.cache_resource
def load_cnn_model():
    model = load_model("fashion_model.h5")
    return model

model = load_cnn_model()

# Preprocess image
def preprocess(img):
    img = img.resize((28, 28)).convert("L")
    img = ImageOps.invert(img)  # invert for better prediction
    img = np.array(img).astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

st.title("👗 Fashion MNIST – Deep Learning App (All in One)")
st.write("Upload an image, draw an image, view dataset samples, train model, and predict – all inside one app.")

tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Image", "✍️ Draw", "📊 Dataset Samples", "🧠 Train Model"])

# ----------------------------
# TAB 1: UPLOAD IMAGE
# ----------------------------
with tab1:
    st.header("📤 Upload an Image for Prediction")

    uploaded = st.file_uploader("Upload a 28x28 clothing item (jpg/png)", type=["jpg", "png", "jpeg"])

    if uploaded is not None:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", width=200)

        processed = preprocess(img)
        pred = model.predict(processed)
        label = labels[np.argmax(pred)]
        st.success(f"Prediction: **{label}**")

# ----------------------------
# TAB 2: DRAW DIGIT
# ----------------------------
with tab2:
    st.header("✍️ Draw an Item (like a shirt, shoe, bag)")

    canvas = st.canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        width=300,
        height=300
    )

    if st.button("Predict Drawing"):
        if canvas.image_data is not None:
            img = canvas.image_data
            img = cv2.resize(img, (28, 28))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = Image.fromarray(img)
            processed = preprocess(img)
            pred = model.predict(processed)
            label = labels[np.argmax(pred)]
            st.success(f"Prediction: **{label}**")

# ----------------------------
# TAB 3: VIEW DATASET
# ----------------------------
with tab3:
    st.header("📊 Fashion MNIST Samples")

    (X_train, y_train), _ = fashion_mnist.load_data()
    
    col1, col2, col3, col4 = st.columns(4)

    for i, col in zip(range(4), [col1, col2, col3, col4]):
        col.image(X_train[i], caption=labels[y_train[i]], use_column_width=True)

# ----------------------------
# TAB 4: TRAIN MODEL
# ----------------------------
with tab4:
    st.header("🧠 Train a New Model (Optional)")

    if st.button("Start Training (Takes 1–2 mins)"):
        st.write("Training model... please wait")

        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        
        X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
        X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

        new_model = Sequential([
            Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax")
        ])

        new_model.compile(optimizer="adam",
                          loss="categorical_crossentropy",
                          metrics=["accuracy"])

        history = new_model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

        new_model.save("fashion_model.h5")

        st.success("🎉 Training Completed! Model saved as fashion_model.h5")

