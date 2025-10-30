
import argparse
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import NPZ_PATH, TARGET_SIZE, BATCH_SIZE, EPOCHS, AUGMENT, AUG_PARAMS, RANDOM_STATE, TEST_SIZE, VAL_SIZE_WITHIN_TEST
from data_npz import load_npz, split_and_encode

def build_model(input_shape, n_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train(npz_path=NPZ_PATH, epochs=EPOCHS, batch_size=BATCH_SIZE, augment=AUGMENT, aug_params=AUG_PARAMS):
    X, y = load_npz(npz_path)
    X_train, X_val, X_test, y_train_enc, y_val_enc, y_test_enc, class_names = split_and_encode(
        X, y, TEST_SIZE, VAL_SIZE_WITHIN_TEST, RANDOM_STATE
    )
    model = build_model((TARGET_SIZE[0], TARGET_SIZE[1], 3), len(class_names))

    if augment:
        datagen = ImageDataGenerator(**aug_params)
        datagen.fit(X_train)
        history = model.fit(
            datagen.flow(X_train, y_train_enc, batch_size=batch_size),
            validation_data=(X_val, y_val_enc),
            epochs=epochs
        )
    else:
        history = model.fit(
            X_train, y_train_enc,
            validation_data=(X_val, y_val_enc),
            batch_size=batch_size,
            epochs=epochs
        )
    return model, history, (X_train, X_val, X_test, y_train_enc, y_val_enc, y_test_enc, class_names)

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', default=NPZ_PATH)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--augment', type=int, default=1 if AUGMENT else 0)
    parsed = parser.parse_args(args)

    model, history, data_tuple = train(parsed.npz_path, parsed.epochs, parsed.batch_size, bool(parsed.augment))
    print("Training complete. You can now evaluate using evaluate.py")

if __name__ == '__main__':
    main()
