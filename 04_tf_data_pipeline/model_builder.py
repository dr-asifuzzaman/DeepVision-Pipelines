# model_builder.py
import os, json, tensorflow as tf
from config import IMG_SIZE, BASE_LR

def build_model(class_names, model_dir):
    num_classes = len(class_names)
    normalization = tf.keras.layers.Rescaling(1./255)
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = normalization(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x*255.0)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_classifier")

    model.compile(optimizer=tf.keras.optimizers.Adam(BASE_LR),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # Save label map
    with open(os.path.join(model_dir, "label_map.json"), "w") as f:
        json.dump({i: c for i, c in enumerate(class_names)}, f, indent=2)

    return model
