import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mlops.ml_logic.registry import load_multimodal_model
from mlops.interface.main import fast_pred_multimodal
import uuid

from mlops.params import *

app = FastAPI()
genres = ['Action',
  'Adventure',
  'Animation',
  'Biography',
  'Comedy',
  'Crime',
  'Horror',
  'Romance',
  'Thriller',
  'War']
model = load_multimodal_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/predict/")
async def create_upload_file(sypnosis: str, file: UploadFile = File(...)):

    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()

    # save the file
    with open(f"{SAVEIMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)

    prediction = fast_pred_multimodal(
        model=model,
        image_file_path=f"{SAVEIMAGEDIR}{file.filename}",
        text=sypnosis,
        genres=genres,
        threshold=0.3
        )

    if os.path.exists(f"{SAVEIMAGEDIR}{file.filename}"):
        os.remove(f"{SAVEIMAGEDIR}{file.filename}")
        print("deleted image from local")
    else:
        print(f"{SAVEIMAGEDIR}{file.filename} does not exist")

    # return {"filename": file.filename, "prediction": ",".join(prediction)}
    return {"filename": file.filename, "length": len(sypnosis), "prediction": prediction}

@app.get("/")
def root():
    return {'greeting': 'Hello'}
