import torch
from torchvision.models import efficientnet_b0
from torchvision.models import EfficientNet_B0_Weights


class ImageClassifier:

    def __init__(self):

        print("Loading EfficientNet-B0...")

        # Pretrained weights
        self.weights = EfficientNet_B0_Weights.DEFAULT

        # Load model
        self.model = efficientnet_b0(
            weights=self.weights
        )

        # Inference mode
        self.model.eval()

        # Preprocessing
        self.transform = self.weights.transforms()

        # ImageNet class names
        self.categories = self.weights.meta["categories"]

        print("Model loaded successfully!")


    def predict(self, image, top_k=5):

        # -------------------------------
        # Preprocessing
        # -------------------------------

        image_tensor = self.transform(image)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)


        # -------------------------------
        # Inference
        # -------------------------------

        with torch.no_grad():

            output = self.model(image_tensor)


        # -------------------------------
        # Convert logits to probabilities
        # -------------------------------

        probabilities = torch.softmax(
            output,
            dim=1
        )


        # -------------------------------
        # Top K predictions
        # -------------------------------

        values, indices = torch.topk(
            probabilities,
            top_k
        )


        results = []

        for value, index in zip(
            values[0],
            indices[0]
        ):

            class_name = self.categories[
                index.item()
            ]

            confidence = value.item() * 100

            results.append(
                {
                    "class": class_name,
                    "confidence": confidence
                }
            )


        return results