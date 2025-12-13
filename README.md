#We use this code to download keras model for our face detection.

import tensorflow as tf
from tensorflow.keras.utils import get_file

MODEL_URL = "https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"

model_path = get_file("emotion_model.hdf5", MODEL_URL, cache_subdir="models")
print("Downloaded emotion model to:", model_path)
