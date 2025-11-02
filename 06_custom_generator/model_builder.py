# model_builder.py
import tensorflow as tf
from tensorflow.keras import layers, models
from config import Config

def build_model(cfg, num_classes: int) -> tf.keras.Model:
    base = tf.keras.applications.EfficientNetB3(
        input_shape=(*cfg.img_size, cfg.channels),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False  # freeze for transfer learning

    inputs = tf.keras.Input(shape=(*cfg.img_size, cfg.channels))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="EfficientNetB3_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.base_learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=cfg.label_smoothing),
        metrics=["accuracy"]
    )
    return model
