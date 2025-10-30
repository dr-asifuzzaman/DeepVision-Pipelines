
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib.patches import Rectangle

# 1) Dataset distribution
def plot_dataset_distribution(y_all, y_train, y_val, y_test):
    def _bar(labels, title):
        u, c = np.unique(labels, return_counts=True)
        plt.figure(figsize=(8,4))
        plt.bar(u, c)
        plt.title(title)
        plt.xlabel('Class'); plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout(); plt.show()
    _bar(y_all, 'Dataset Distribution (All)')
    _bar(y_train, 'Train Distribution')
    _bar(y_val, 'Validation Distribution')
    _bar(y_test, 'Test Distribution')

# 2) Class image grid
def show_class_grid(X_data, y_enc, class_names, samples_per_class=5):
    rows = len(class_names); cols = samples_per_class
    plt.figure(figsize=(cols*2.2, rows*2.2))
    idx = 1
    for ci, cname in enumerate(class_names):
        indices = np.where(y_enc == ci)[0]
        if len(indices) == 0: continue
        choose = np.random.choice(indices, size=min(samples_per_class, len(indices)), replace=False)
        for k in choose:
            ax = plt.subplot(rows, cols, idx)
            ax.imshow(X_data[k]); ax.set_title(cname, fontsize=9); ax.axis('off')
            idx += 1
    plt.tight_layout(); plt.show()

# 3) Augmentation visualization
def visualize_augmentations(X_train, aug_params=None, n=5):
    if aug_params is None:
        aug_params = dict(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, horizontal_flip=True)
    datagen = ImageDataGenerator(**aug_params)
    datagen.fit(X_train)
    idx = np.random.randint(0, len(X_train))
    sample = X_train[idx:idx+1]
    plt.figure(figsize=(n*2.0, 2.4))
    plt.subplot(1, n+1, 1); plt.imshow(sample[0]); plt.title('Original'); plt.axis('off')
    it = datagen.flow(sample, batch_size=1)
    for i in range(2, n+2):
        aug = next(it)[0]
        plt.subplot(1, n+1, i); plt.imshow(aug); plt.title(f'Aug {i-1}'); plt.axis('off')
    plt.tight_layout(); plt.show()

# 4) Training curves side-by-side
def plot_training_curves(history):
    h = history.history
    epochs = range(1, len(h.get('loss', [])) + 1)
    acc_key = 'accuracy' if 'accuracy' in h else ('acc' if 'acc' in h else None)
    val_acc_key = 'val_accuracy' if 'val_accuracy' in h else ('val_acc' if 'val_acc' in h else None)
    fig, axes = plt.subplots(1, 2, figsize=(12,4))

    axes[0].plot(epochs, h.get('loss', []), label='Train Loss')
    if 'val_loss' in h: axes[0].plot(epochs, h['val_loss'], label='Val Loss')
    axes[0].set_title('Training & Validation Loss'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.3)

    if acc_key is not None:
        axes[1].plot(epochs, h.get(acc_key, []), label='Train Acc')
        if val_acc_key is not None and val_acc_key in h:
            axes[1].plot(epochs, h[val_acc_key], label='Val Acc')
        axes[1].set_title('Training & Validation Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].legend(); axes[1].grid(True, linestyle='--', alpha=0.3)
    else:
        axes[1].axis('off'); axes[1].text(0.5,0.5,'No accuracy metric', ha='center', va='center')
    plt.tight_layout(); plt.show()

# 5) Confusion matrix with colormap
def plot_confusion_matrix(y_true_enc, y_prob, class_names, cmap='Blues'):
    y_pred = np.argmax(y_prob, axis=1)
    cm = confusion_matrix(y_true_enc, y_pred)
    plt.figure(figsize=(6,5))
    im = plt.imshow(cm, interpolation='nearest', cmap=getattr(plt.cm, cmap) if isinstance(cmap, str) else cmap)
    plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=45)
    plt.yticks(ticks=np.arange(len(class_names)), labels=class_names)
    thresh = cm.max()/2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i,j], ha='center', va='center', color='white' if cm[i,j]>thresh else 'black', fontsize=10)
    plt.tight_layout(); plt.show()

# 6) Classification report
def print_classification_report(y_true_enc, y_prob, class_names):
    from sklearn.metrics import classification_report
    y_pred = np.argmax(y_prob, axis=1)
    print('Classification Report:')
    print(classification_report(y_true_enc, y_pred, target_names=class_names))

# 7) ROC–AUC (single figure for all classes + micro/macro)
def plot_roc_auc_all(y_true_enc, y_prob, class_names):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true_enc, classes=list(range(n_classes)))
    if y_prob.shape[1] == 1 and n_classes == 2:
        y_prob = np.hstack([1 - y_prob, y_prob])

    fpr = {}; tpr = {}; roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:,i], y_prob[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr['micro'], tpr['micro'], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    roc_auc['macro'] = np.mean([roc_auc[i] for i in range(n_classes)])

    plt.figure(figsize=(7,6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})")
    plt.plot(fpr['micro'], tpr['micro'], linestyle='--', label=f"Micro-avg (AUC={roc_auc['micro']:.3f})")
    plt.plot([0,1], [0,1], linestyle=':')
    plt.title('ROC Curves (All Classes)')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.xlim([0.0,1.0]); plt.ylim([0.0,1.05]); plt.legend(loc='lower right'); plt.tight_layout(); plt.show()

# 8) Prediction gallery with misclassifications in red
def show_predictions(X_data, y_true_enc, class_names, model, rows=2, cols=5):
    idxs = np.random.choice(len(X_data), size=min(rows*cols, len(X_data)), replace=False)
    y_prob = model.predict(X_data[idxs], verbose=0)
    preds = np.argmax(y_prob, axis=1)
    confs = np.max(y_prob, axis=1)

    plt.figure(figsize=(cols*2.2, rows*2.2))
    for i, (img_idx, pred, conf) in enumerate(zip(idxs, preds, confs)):
        true_idx = y_true_enc[img_idx]
        true_label = class_names[true_idx]
        pred_label = class_names[pred]
        is_correct = (pred == true_idx)

        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(X_data[img_idx]); ax.axis('off')
        title_color = 'green' if is_correct else 'red'
        ax.set_title(f"P: {pred_label} ({conf*100:.1f}%)\nT: {true_label}", fontsize=9, color=title_color)

        h, w = X_data[img_idx].shape[:2]
        rect = Rectangle((0,0), w-1, h-1, fill=False, linewidth=2.0, edgecolor=title_color)
        ax.add_patch(rect)
    plt.tight_layout(); plt.show()
