from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io

from model import ImageClassifier


# ==========================================
# Create FastAPI Application
# ==========================================

app = FastAPI(
    title="AI Image Classification API",
    description="Deep Learning Image Classification using EfficientNet-B0",
    version="1.0.0"
)


# ==========================================
# Load Model
# ==========================================

classifier = ImageClassifier()


# ==========================================
# Home Endpoint
# ==========================================

@app.get("/")
def home():

    return {
        "message": "AI Image Classification API is running",
        "model": "EfficientNet-B0",
        "framework": "PyTorch"
    }


# ==========================================
# Health Check
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# ==========================================
# Prediction Endpoint
# ==========================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------
    # Validate file type
    # --------------------------------------

    allowed_types = [
        "image/jpeg",
        "image/png"
    ]

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are allowed."
        )


    # --------------------------------------
    # Read uploaded file
    # --------------------------------------

    contents = await file.read()


    # --------------------------------------
    # Convert bytes to PIL Image
    # --------------------------------------

    try:

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )


    # --------------------------------------
    # Run Deep Learning inference
    # --------------------------------------

    results = classifier.predict(
        image,
        top_k=5
    )


    # --------------------------------------
    # Return response
    # --------------------------------------

    return {

        "filename": file.filename,

        "prediction": results[0]["class"],

        "confidence": round(
            results[0]["confidence"],
            2
        ),

        "top_5_predictions": [

            {
                "class": result["class"],

                "confidence": round(
                    result["confidence"],
                    2
                )

            }

            for result in results

        ]

    }