import numpy as np
from tensorflow import keras
from tensorflow.keras import layers         
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
import matplotlib.image as Image

KERAS_MODEL_NAME = "Model/keras.hdf5"
model = keras.models.load_model(KERAS_MODEL_NAME)

img = Image.imread("imgs\\slika.jpg")
img = img[:, :, 0]
print(img.shape)

img_reshaped = np.reshape(img, (1, img.shape[0]*img.shape[1]))

img_pred = model.predict(img_reshaped)
img_pred = np.argmax(img_pred, axis=1)

print("Broj na slici:", img_pred)
