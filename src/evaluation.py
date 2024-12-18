import config
from preprocessing import data
import tensorflow as tf
from tensorflow.keras.models import load_model

model=load_model(config.MODEL_PATH)

dt=data()
dt.get_data()
model.evaluate(dt.scaled_test)

# got 93% accuracy on test data