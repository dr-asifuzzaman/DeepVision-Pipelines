
import argparse
from config import IMAGE_DIR, NPZ_PATH, STRUCTURED_NPZ_PATH, TARGET_SIZE, TEST_SIZE, VAL_SIZE_WITHIN_TEST, RANDOM_STATE, BATCH_SIZE, EPOCHS, AUGMENT, AUG_PARAMS
from data_npz import load_and_preprocess_images, save_npz, load_npz, split_and_encode, memory_report, save_structured_npz
from train import build_model, train
from viz import plot_dataset_distribution, show_class_grid, visualize_augmentations, plot_training_curves, plot_confusion_matrix, print_classification_report, plot_roc_auc_all, show_predictions

def main(args=None):
    parser = argparse.ArgumentParser(description='End-to-end NPZ pipeline')
    parser.add_argument('--image_dir', default=IMAGE_DIR, help='Path to images folder')
    parser.add_argument('--npz_path', default=NPZ_PATH, help='Path to save/load NPZ')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--augment', type=int, default=1 if AUGMENT else 0)
    parser.add_argument('--create_npz', type=int, default=1, help='Create NPZ from images if 1')
    parsed = parser.parse_args(args)

    # Step 1: Create NPZ (optional)
    if parsed.create_npz:
        X, y = load_and_preprocess_images(parsed.image_dir, TARGET_SIZE)
        save_npz(X, y, parsed.npz_path)
        memory_report(parsed.npz_path)
    else:
        X, y = load_npz(parsed.npz_path)

    # Step 2: Split + encode
    X_train, X_val, X_test, y_train_enc, y_val_enc, y_test_enc, class_names = split_and_encode(
        X, y, TEST_SIZE, VAL_SIZE_WITHIN_TEST, RANDOM_STATE
    )
    save_structured_npz(X_train, X_val, X_test, y_train_enc, y_val_enc, y_test_enc, class_names, STRUCTURED_NPZ_PATH)

    # Step 3: Train
    model, history, _ = train(parsed.npz_path, parsed.epochs, parsed.batch_size, bool(parsed.augment))

    # Step 4: Visuals + Evaluate
    from numpy import unique
    plot_dataset_distribution(y, class_names[y_train_enc], class_names[y_val_enc], class_names[y_test_enc])
    show_class_grid(X_train, y_train_enc, class_names, samples_per_class=4)
    if bool(parsed.augment):
        visualize_augmentations(X_train, AUG_PARAMS, n=5)
    plot_training_curves(history)

    # Predict on test for metrics
    import numpy as np
    y_prob = model.predict(X_test, verbose=0)
    plot_confusion_matrix(y_test_enc, y_prob, class_names, cmap='Blues')
    print_classification_report(y_test_enc, y_prob, class_names)
    plot_roc_auc_all(y_test_enc, y_prob, class_names)
    show_predictions(X_test, y_test_enc, class_names, model, rows=2, cols=5)

if __name__ == '__main__':
    main()
