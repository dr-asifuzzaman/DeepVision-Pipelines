# tf_record_pipeline/model_builder.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import IMG_SIZE, CHANNELS

def build_model(num_classes: int) -> keras.Model:
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, CHANNELS),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inp = keras.Input(shape=(*IMG_SIZE, CHANNELS))
    x = keras.applications.mobilenet_v2.preprocess_input(inp * 255.0)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inp, out, name="mobilenetv2_classifier")
    return model

def fine_tune(model: keras.Model, lr=1e-4):
    # Unfreeze backbone (layer index 1 is backbone in our assembly)
    model.get_layer(index=1).trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
