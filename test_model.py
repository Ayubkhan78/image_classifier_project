from PIL import Image

from model import ImageClassifier


# Load model
classifier = ImageClassifier()


# Change this to your image name
image = Image.open(
    "sample_images/sample.jpg"
).convert("RGB")


# Prediction
prediction, confidence = classifier.predict(image)


print("\nPrediction:", prediction)
print("Confidence:", f"{confidence:.2f}%")