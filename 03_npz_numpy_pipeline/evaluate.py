
import argparse
import numpy as np

from config import NPZ_PATH, TARGET_SIZE, RANDOM_STATE, TEST_SIZE, VAL_SIZE_WITHIN_TEST
from data_npz import load_npz, split_and_encode
from train import build_model
from viz import plot_dataset_distribution, show_class_grid, visualize_augmentations, plot_training_curves, plot_confusion_matrix, print_classification_report, plot_roc_auc_all, show_predictions

def evaluate(npz_path=NPZ_PATH, epochs=5):
    # Load and split
    X, y = load_npz(npz_path)
    X_train, X_val, X_test, y_train_enc, y_val_enc, y_test_enc, class_names = split_and_encode(
        X, y, TEST_SIZE, VAL_SIZE_WITHIN_TEST, RANDOM_STATE
    )

    # Train a small model quickly (for demo when run standalone)
    model = build_model((TARGET_SIZE[0], TARGET_SIZE[1], 3), len(class_names))
    history = model.fit(X_train, y_train_enc, validation_data=(X_val, y_val_enc), batch_size=32, epochs=epochs, verbose=0)

    # Predictions
    y_prob = model.predict(X_test, verbose=0)

    # Visuals & Reports
    plot_dataset_distribution(y, class_names[y_train_enc], class_names[y_val_enc], class_names[y_test_enc])
    show_class_grid(X_train, y_train_enc, class_names, samples_per_class=4)
    visualize_augmentations(X_train, n=5)
    plot_training_curves(history)
    plot_confusion_matrix(y_test_enc, y_prob, class_names, cmap='Blues')
    print_classification_report(y_test_enc, y_prob, class_names)
    plot_roc_auc_all(y_test_enc, y_prob, class_names)
    show_predictions(X_test, y_test_enc, class_names, model, rows=2, cols=5)

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', default=NPZ_PATH)
    parser.add_argument('--epochs', type=int, default=5)
    parsed = parser.parse_args(args)
    evaluate(parsed.npz_path, parsed.epochs)

if __name__ == '__main__':
    main()
