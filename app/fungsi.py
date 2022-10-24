import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout,LeakyReLU, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2

def make_model():
    model = Sequential()
    model.add(MobileNetV2(input_shape=(224, 224, 3),include_top=False))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='sigmoid',name='preds'))
    model.layers[0].trainable= False
    # show model summary
    # model.summary()
    
    return model
