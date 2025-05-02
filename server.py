from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from src.image_frame_processor import process_image_frame
from PIL import Image
import numpy as np
import cv2
import io

app = FastAPI()

@app.get("/atten-sense/api/v1/is-up/")
async def isup():
    return {"message": "Server is up and running"}

@app.post("/atten-sense/api/v1/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Read image bytes from file
        contents = await file.read()

        # Open image with PIL
        # io.BytesIO(contents) converts raw bytes into a file like object
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Convert to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        is_focused, prediction = process_image_frame(frame)

        return {"is_focused": is_focused, "prediction": prediction}

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
