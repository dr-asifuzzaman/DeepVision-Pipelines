
import argparse
from config import NPZ_PATH, TARGET_SIZE, RANDOM_STATE, TEST_SIZE, VAL_SIZE_WITHIN_TEST
from data_npz import load_npz, split_and_encode
from train import build_model
from viz import show_predictions

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', default=NPZ_PATH)
    parser.add_argument('--rows', type=int, default=2)
    parser.add_argument('--cols', type=int, default=5)
    parsed = parser.parse_args(args)

    X, y = load_npz(parsed.npz_path)
    X_train, X_val, X_test, y_train_enc, y_val_enc, y_test_enc, class_names = split_and_encode(X, y)
    model = build_model((TARGET_SIZE[0], TARGET_SIZE[1], 3), len(class_names))
    model.fit(X_train, y_train_enc, validation_data=(X_val, y_val_enc), epochs=3, batch_size=32, verbose=0)

    show_predictions(X_test, y_test_enc, class_names, model, rows=parsed.rows, cols=parsed.cols)

if __name__ == '__main__':
    main()
