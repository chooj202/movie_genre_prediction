import os
import numpy as np

##################  VARIABLES  ##################
IMAGE_WIDTH = os.environ.get("IMAGE_WIDTH")
IMAGE_HEIGHT = os.environ.get("IMAGE_HEIGHT")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
BQ_DATASET = os.environ.get("BQ_DATASET")
MODEL_NAME= os.environ.get("MODEL_NAME")
DATA_SIZE = os.environ.get("DATA_SIZE")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
SAVEIMAGEDIR = os.environ.get("SAVEIMAGEDIR")

##################  CONSTANTS  #####################
GENRE_NAMES = ["action", "adventure", "animation", "biography", "comedy", "crime", "drama", "family", "fantasy", "film-noir", "history", "horror", "music", "musical", "mystery", "romance", "scifi", "sport", "thriller", "war", "western"]
LOCAL_DATA_PATH = "data"
LOCAL_REGISTRY_PATH =  "training_outputs"


env_valid_options = dict(
    MODEL_TARGET=["local", "gcs"],
)

def validate_env_value(env, valid_options):
    env_value = os.environ[env]
    if env_value not in valid_options:
        raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")

for env, valid_options in env_valid_options.items():
    validate_env_value(env, valid_options)
