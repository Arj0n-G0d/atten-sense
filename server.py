from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from src.image_frame_processor import process_image_frame
from src.video_frame_processor import process_video_frame
from db.db_utils import insert_focus_session, insert_focus_log
from PIL import Image
import tempfile
import os
import time
import numpy as np
import cv2
import io

def group_focus_log(focus_logs):
    if not focus_logs:
        return []
    
    grouped_logs = []
    start_time, state = focus_logs[0]  # Initialize with first entry

    for i in range(1, len(focus_logs)):
        current_time, current_state = focus_logs[i]

        # If state changes, store the previous segment
        if current_state != state:
            grouped_logs.append((start_time, current_time, state))  # (start, end, state)
            start_time = current_time  # Update start time for the new state
            state = current_state  # Update state

    # Append last segment
    grouped_logs.append((start_time, focus_logs[-1][0] + 0.05, state))
    
    return grouped_logs

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

        return {"message": "Image processed successfully", "is_focused": is_focused, "prediction": prediction}

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/atten-sense/api/v1/upload-video/")
async def upload_video(name: str = Form(...), file: UploadFile = File(...)):
    # Save the uploaded file to a temporary location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Open the video file using OpenCV
        cap = cv2.VideoCapture(tmp_path)
        frame_count = 0

        start_time = time.time()
        focus_logs = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame and get focus status
            frame, is_focused = process_video_frame(frame)
            current_time = time.time() - start_time
            focus_logs.append((current_time, is_focused))

        cap.release()
        os.remove(tmp_path)  # Clean up the temp file

        # Generate session ID when ending analysis
        current_session_id, current_date = insert_focus_session(name)
        # Insert focus logs
        grouped_focus_logs = group_focus_log(focus_logs)
        for (start, end, focus_state) in grouped_focus_logs:
            insert_focus_log(current_session_id, start, end, focus_state)

        # Convert tuples into list of dictionaries
        json_logs = [
            {
                "start": start,
                "end": end,
                "is_focused": focus_state
            } for (start, end, focus_state) in grouped_focus_logs
        ]

        return {
            "message": "Video processed successfully",
            "focus_logs": json_logs
        }

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

