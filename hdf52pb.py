import os
from tensorflow import keras

model_path = 'D:/dataset/models/crack_detection.hdf5'
model = keras.models.load_model(model_path, compile=False)

model_dir = os.path.dirname(model_path)
model_name = os.path.basename(model_path)
export_path = os.path.join(model_dir, model_name.split('.')[0])
model.save(export_path, save_format='tf')