import streamlit as st
from PIL import Image

from model import ImageClassifier


# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🤖",
    layout="centered"
)


# --------------------------------
# Load Model
# --------------------------------

@st.cache_resource
def load_model():

    return ImageClassifier()


classifier = load_model()


# --------------------------------
# Title
# --------------------------------

st.title("🤖 AI Image Classification")

st.write(
    "Upload an image and let "
    "EfficientNet-B0 classify it."
)


# --------------------------------
# Upload Image
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# --------------------------------
# Prediction
# --------------------------------

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )


    # Predict
    if st.button("🔍 Predict"):

        with st.spinner(
            "Running Deep Learning inference..."
        ):

            results = classifier.predict(
                image,
                top_k=5
            )


        # --------------------------------
        # Main Prediction
        # --------------------------------

        top_prediction = results[0]

        st.success(
            f"Prediction: "
            f"{top_prediction['class']}"
        )

        st.metric(
            "Confidence",
            f"{top_prediction['confidence']:.2f}%"
        )


        # --------------------------------
        # Top 5 Predictions
        # --------------------------------

        st.subheader(
            "Top 5 Predictions"
        )


        for i, result in enumerate(
            results,
            start=1
        ):

            st.write(
                f"**{i}. {result['class']}**"
            )

            st.progress(
                int(result["confidence"])
            )

            st.caption(
                f"{result['confidence']:.2f}%"
            )