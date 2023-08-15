import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,BatchNormalization,Dropout

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")



def initialize_model(input_shape: tuple, output_shape: int) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    model = Sequential()
    model.add(Conv2D(16,kernel_size=(3,3),activation='relu',input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.3))

    model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.3))


    model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(output_shape,activation='sigmoid'))

    return model

def compile_model(model: Model) -> Model:
    """
    Compile the Neural Network
    """
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    print("✅ Model compiled")

    return model

def train_model(
        model: Model,
        X_train,
        y_train,
        X_val,
        y_val,
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    history = model.fit(X_train,y_train,epochs=5,validation_data=(X_val,y_val))

    print(f"✅ Model trained on {len(X_train)} rows with accuracy of: {round(history.history['accuracy'][-1], 2)}")

    return model, history
