import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow.keras import Sequential, Model
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,BatchNormalization,Dropout,Input
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
from transformers import TFBertModel
from tensorflow.keras.callbacks import EarlyStopping

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



def initialize_compile_multimodal(num_classes) -> Model:
    """
    RESNET and BERT model
    """
    # RESNET
    def load_model():
        model = ResNet50(weights="imagenet", include_top=False, input_shape=(256, 256, 3))
        return model
    def set_nontrainable_layers(model):
        model.trainable = False
        return model
    def add_last_layers(model):
        '''Take a pre-trained model, set its parameters as non-trainable, and add additional trainable layers on top'''
        base_model = load_model()
        base_model = set_nontrainable_layers(base_model)
        flatten_layer = layers.Flatten()
        dense_layer = layers.Dense(500, activation='relu')
        prediction_layer = layers.Dense(512, activation='relu')

        model = models.Sequential([
            base_model,
            flatten_layer,
            dense_layer,
            prediction_layer
        ])
        return model

    # define the BERT-based text feature extractor
    def build_text_model():
        bert_model = TFBertModel.from_pretrained('bert-base-uncased')
        inputs = Input(shape=(None,), dtype=tf.int32, name='input_word_ids')
        outputs = bert_model(inputs)[1]
        text_model = Model(inputs=inputs, outputs=outputs)
        return text_model

    # define the multimodal document classification model
    def build_multimodal_model(num_classes):
        model = load_model()
        model = set_nontrainable_layers(model)
        img_model = add_last_layers(model)

        print(Fore.BLUE + "\nDone adding layers to RESNET" + Style.RESET_ALL)

        text_model = build_text_model()

        print(Fore.BLUE + "\nDone buidling text model" + Style.RESET_ALL)

        img_input = Input(shape=(256, 256, 3), name='img_input')
        text_input = Input(shape=(None,), dtype=tf.int32, name='text_input')
        img_features = img_model(img_input)
        text_features = text_model(text_input)
        concat_features = tf.keras.layers.concatenate([img_features, text_features])
        x = tf.keras.layers.Dense(512, activation='relu')(concat_features)
        x = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)
        multimodal_model = tf.keras.Model(inputs=[img_input, text_input], outputs=x)
        return multimodal_model

    # build the multimodal model
    multimodal_model = build_multimodal_model(num_classes)

    print(multimodal_model.summary())

    # compile the model and train on the train set
    multimodal_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC', 'binary_accuracy', 'categorical_accuracy'])

    return multimodal_model


def compile_model(model: Model) -> Model:
    """
    Compile the Neural Network
    """
    legacy_adam = tf.keras.optimizers.legacy.Adam()
    model.compile(optimizer=legacy_adam,loss='binary_crossentropy',metrics=['accuracy'])

    print("✅ Model compiled")

    return model



def fit_multimodal(
    model,
    X_train_img,
    X_train_text,
    y_train,
    X_val_img,
    X_val_text,
    y_val
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """

    print(Fore.BLUE + "\nTraining multimodal model..." + Style.RESET_ALL)

    es = EarlyStopping(monitor = 'val_accuracy',
                    mode = 'max',
                    patience = 5,
                    verbose = 1,
                    restore_best_weights = True)

    history = model.fit([(X_train_img, X_train_text)], tf.convert_to_tensor(y_train), epochs=1, batch_size=64, validation_data=([(X_val_img, X_val_text)], tf.convert_to_tensor(y_val)), callbacks = [es])

    print(f"✅ Model trained on {len(X_train_img)} rows with accuracy of: {round(history.history['accuracy'][-1], 2)}")

    return (model, history)


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
