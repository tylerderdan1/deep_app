import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Fashion MNIST Classifier", layout="wide")

# Labels
labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Load Model
@st.cache_resource
def load_cnn_model():
    return load_model("fashion_model.h5")

model = load_cnn_model()

# Preprocess Image
def preprocess(img):
    img = img.resize((28, 28)).convert("L")
    img = ImageOps.invert(img)
    img = np.array(img).astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

st.title("👗 Fashion MNIST – Deep Learning App (All in One)")

tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Image", "✍️ Draw", "📊 Samples", "🧠 Train Model"])

# ----------------------
# TAB 1: UPLOAD IMAGE
# ----------------------
with tab1:
    st.header("Upload an Image")

    uploaded = st.file_uploader("Upload a clothing image (png/jpg)", type=["png", "jpg", "jpeg"])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, width=200)

        processed = preprocess(img)
        pred = model.predict(processed)
        label = labels[np.argmax(pred)]

        st.success(f"Prediction: **{label}**")

# ----------------------
# TAB 2: DRAW CANVAS
# ----------------------
with tab2:
    st.header("Draw Clothing Item")

    canvas = st_canvas(
        fill_color="#000000",
        stroke_color="#FFFFFF",
        stroke_width=10,
        width=300,
        height=300,
        drawing_mode="freedraw",
        key="canvas"
    )

    if st.button("Predict Drawing"):
        if canvas.image_data is not None:
            # Convert canvas → PIL Image
            img = Image.fromarray(canvas.image_data.astype("uint8"))
            img = img.convert("L")
            img = img.resize((28, 28))

            processed = preprocess(img)
            pred = model.predict(processed)
            label = labels[np.argmax(pred)]

            st.success(f"Prediction: **{label}**")

# ----------------------
# TAB 3: SAMPLES
# ----------------------
with tab3:
    st.header("Dataset Samples")

    (X_train, y_train), _ = fashion_mnist.load_data()

    cols = st.columns(4)
    for i, col in zip(range(4), cols):
        col.image(X_train[i], caption=labels[y_train[i]], use_column_width=True)

# ----------------------
# TAB 4: TRAIN MODEL
# ----------------------
with tab4:
    st.header("Train New Model")

    if st.button("Start Training"):
        st.write("Training... please wait")

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
            Dropout(0.25),
            Dense(10, activation="softmax")
        ])

        new_model.compile(optimizer="adam",
                          loss="categorical_crossentropy",
                          metrics=["accuracy"])

        history = new_model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

        new_model.save("fashion_model.h5")

        st.success("Model Training Complete! New model saved.")
