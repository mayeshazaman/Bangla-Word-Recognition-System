"""
train.py — BanglaLekha-Isolated OCR Training Script
=====================================================
Dataset  : BanglaLekha-Isolated (84 classes, ~166 K images)
Model    : MobileNetV2 transfer-learning (fast, accurate on small images)
Tracking : MLflow — two named runs are always recorded:
             Run 1  "mobilenetv2_warmup"   – frozen base, head only
             Run 2  "mobilenetv2_finetune" – top-50 base layers unfrozen
Outputs  : models/model.keras  — best checkpoint (val accuracy)
           labels.json         — {index: unicode_character} mapping
           artifacts/mlflow/   — MLflow tracking directory

Usage
-----
1. Put the dataset in the project root as:
       BanglaLekha-Isolated/<class_folder>/*.png
   or pass your own path with --data_dir.

2. Install deps:
       pip install -r requirements.txt

3. Run:
       python train.py [--data_dir BanglaLekha-Isolated]
                       [--img_size 64]
                       [--batch_size 64]
                       [--epochs 5]
                       [--fine_tune_epochs 2]
                       [--lr 1e-3]
                       [--fine_tune_lr 1e-4]
                       [--val_split 0.15]
                       [--test_split 0.05]
                       [--experiment_name bangla_ocr]
"""

import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train Bangla OCR classifier")
    p.add_argument("--data_dir",         default="BanglaLekha-Isolated",
                   help="Root folder whose sub-folders are class labels")
    p.add_argument("--img_size",         type=int,   default=64)
    p.add_argument("--batch_size",       type=int,   default=64)
    p.add_argument("--epochs",           type=int,   default=5,
                   help="Epochs for warm-up phase (Run 1)")
    p.add_argument("--fine_tune_epochs", type=int,   default=2,
                   help="Epochs for fine-tuning phase (Run 2); 0 = skip")
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--fine_tune_lr",     type=float, default=1e-4)
    p.add_argument("--val_split",        type=float, default=0.15)
    p.add_argument("--test_split",       type=float, default=0.05)
    p.add_argument("--experiment_name",  default="bangla_ocr")
    p.add_argument("--model_out",        default="models/model.keras")
    p.add_argument("--labels_out",       default="labels.json")
    p.add_argument("--mlflow_uri",       default="./artifacts/mlflow")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Bangla Unicode label map
# ---------------------------------------------------------------------------

BANGLA_BASIC_CONSONANTS = [
    "ক", "খ", "গ", "ঘ", "ঙ",
    "চ", "ছ", "জ", "ঝ", "ঞ",
    "ট", "ঠ", "ড", "ঢ", "ণ",
    "ত", "থ", "দ", "ধ", "ন",
    "প", "ফ", "ব", "ভ", "ম",
    "য", "র", "ল", "শ", "ষ",
    "স", "হ", "ড়", "ঢ়", "য়",
    "ৎ", "ং", "ঃ", "ঁ",
    # independent vowels
    "অ", "আ", "ই", "ঈ", "উ",
    "ঊ", "ঋ", "এ", "ঐ", "ও",   # 50 total
]

BANGLA_NUMERALS = ["০", "১", "২", "৩", "৪", "৫", "৬", "৭", "৮", "৯"]

BANGLA_COMPOUNDS = [
    "ক্ষ", "জ্ঞ", "ত্র", "ঞ্চ", "ঞ্ছ",
    "ঞ্জ", "ঞ্ঝ", "ট্ট", "ণ্ট", "ণ্ঠ",
    "ণ্ড", "ত্ত", "ত্থ", "দ্দ", "দ্ধ",
    "দ্ব", "দ্ভ", "ন্ত", "ন্থ", "ন্দ",
    "ন্ধ", "ন্ন", "ন্ব", "ন্ম",          # 24 total
]

ALL_CHARS = BANGLA_BASIC_CONSONANTS + BANGLA_NUMERALS + BANGLA_COMPOUNDS


def build_char_map(class_names: list[str]) -> dict[str, str]:
    """Map folder names → Bangla Unicode characters."""
    mapping: dict[str, str] = {}
    for name in sorted(class_names):
        stripped = name.strip()
        if stripped.isdigit():
            idx = int(stripped) - 1
            mapping[name] = ALL_CHARS[idx] if 0 <= idx < len(ALL_CHARS) else stripped
        else:
            mapping[name] = stripped
    return mapping


# ---------------------------------------------------------------------------
# Dataset collection
# ---------------------------------------------------------------------------

def resolve_dataset_dir(data_dir: str) -> Path:
    """Resolve the dataset folder from common project-relative locations."""
    candidates = [
        Path(data_dir),
        Path.cwd() / data_dir,
        Path(__file__).resolve().parent / data_dir,
        Path(__file__).resolve().parent / "data" / data_dir,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    return Path(data_dir)


def collect_paths(data_dir: str) -> tuple[list[str], list[str]]:
    """Return (image_paths, class_name_per_image) lists."""
    root = resolve_dataset_dir(data_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {root.resolve()}\n"
            "Place your dataset under the project root as BanglaLekha-Isolated/Images/<class>/*.png "
            "or pass --data_dir /path/to/your/dataset."
        )

    dataset_root = root / "Images" if (root / "Images").exists() else root

    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    paths, labels = [], []
    for img_file in sorted(dataset_root.rglob("*")):
        if img_file.is_file() and img_file.suffix.lower() in exts:
            paths.append(str(img_file))
            labels.append(img_file.parent.name)

    if not paths:
        raise RuntimeError(
            f"No images found under {dataset_root.resolve()}. "
            "Expected PNG/JPG files in <dataset>/Images/<class>/*.png"
        )

    print(f"Found {len(paths):,} images across {len(set(labels))} classes.")
    return paths, labels


# ---------------------------------------------------------------------------
# tf.data pipeline
# ---------------------------------------------------------------------------

def make_tf_dataset(paths, labels_encoded, class_count,
                    img_size, batch_size, augment=False):
    import tensorflow as tf
    AUTO = tf.data.AUTOTUNE

    def load_and_preprocess(path, label):
        raw   = tf.io.read_file(path)
        img   = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img   = tf.image.resize(img, [img_size, img_size])
        img   = tf.cast(img, tf.float32) / 255.0
        label = tf.one_hot(label, class_count)
        return img, label

    def augment_fn(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels_encoded))
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTO)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=AUTO)
    ds = ds.shuffle(2048).batch(batch_size).prefetch(AUTO)
    return ds


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(img_size: int, num_classes: int, learning_rate: float):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs  = keras.Input(shape=(img_size, img_size, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )
    return model, base


# ---------------------------------------------------------------------------
# MLflow metric-logging callback
# ---------------------------------------------------------------------------

def make_mlflow_callback():
    from tensorflow import keras

    class MlflowCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs:
                mlflow.log_metrics(
                    {k: float(v) for k, v in logs.items()},
                    step=epoch,
                )
    return MlflowCallback()


# ---------------------------------------------------------------------------
# Training script — runs top-level when executed directly
# ---------------------------------------------------------------------------

import tensorflow as tf
from tensorflow import keras

args = parse_args()

# Ensure output directories exist
os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
os.makedirs(args.mlflow_uri, exist_ok=True)

print(f"TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")

# ── collect data ──────────────────────────────────────────────────────────
all_paths, all_labels_str = collect_paths(args.data_dir)
class_names = sorted(set(all_labels_str))
num_classes = len(class_names)

le = LabelEncoder().fit(class_names)
all_labels_int = le.transform(all_labels_str).tolist()

# ── labels.json ───────────────────────────────────────────────────────────
char_map = build_char_map(class_names)
labels_json = {
    "index_to_char": {str(i): char_map[c] for i, c in enumerate(le.classes_)},
    "folder_to_char": char_map,
    "class_names": list(le.classes_),
}
with open(args.labels_out, "w", encoding="utf-8") as f:
    json.dump(labels_json, f, ensure_ascii=False, indent=2)
print(f"Saved label mapping → {args.labels_out}  ({num_classes} classes)")

# ── splits ────────────────────────────────────────────────────────────────
test_frac = args.test_split
val_frac  = args.val_split / (1 - test_frac)

paths_tv, paths_test, labels_tv, labels_test = train_test_split(
    all_paths, all_labels_int,
    test_size=test_frac, stratify=all_labels_int, random_state=42,
)
paths_train, paths_val, labels_train, labels_val = train_test_split(
    paths_tv, labels_tv,
    test_size=val_frac, stratify=labels_tv, random_state=42,
)
print(f"Split → train: {len(paths_train):,}  "
      f"val: {len(paths_val):,}  test: {len(paths_test):,}")

# ── tf.data pipelines ─────────────────────────────────────────────────────
IMG, BS = args.img_size, args.batch_size
train_ds = make_tf_dataset(paths_train, labels_train, num_classes, IMG, BS, augment=True)
val_ds   = make_tf_dataset(paths_val,   labels_val,   num_classes, IMG, BS)
test_ds  = make_tf_dataset(paths_test,  labels_test,  num_classes, IMG, BS)

# ── MLflow setup ──────────────────────────────────────────────────────────
mlflow.set_tracking_uri(args.mlflow_uri)
mlflow.set_experiment(args.experiment_name)

best_ckpt = "best_model.keras"

# ══════════════════════════════════════════════════════════════════════════
# MLflow Run 1 — Warm-up: frozen MobileNetV2 base, train head only
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MLflow Run 1 — mobilenetv2_warmup (frozen base)")
print("=" * 60)

with mlflow.start_run(run_name="mobilenetv2_warmup") as run1:
    mlflow.log_params({
        "model":         "MobileNetV2",
        "phase":         "warmup",
        "img_size":      IMG,
        "batch_size":    BS,
        "epochs":        args.epochs,
        "lr":            args.lr,
        "val_split":     args.val_split,
        "test_split":    args.test_split,
        "num_classes":   num_classes,
        "train_samples": len(paths_train),
        "val_samples":   len(paths_val),
        "test_samples":  len(paths_test),
        "augmentation":  True,
        "base_frozen":   True,
    })

    model, base_model = build_model(IMG, num_classes, args.lr)
    model.summary()

    callbacks_r1 = [
        keras.callbacks.ModelCheckpoint(
            best_ckpt, monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7,
            restore_best_weights=True, verbose=1,
        ),
        make_mlflow_callback(),
    ]

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks_r1,
    )

    # Evaluate after warm-up
    model.load_weights(best_ckpt)
    wu_loss, wu_acc, wu_top3 = model.evaluate(test_ds, verbose=1)
    mlflow.log_metrics({
        "test_loss":     wu_loss,
        "test_accuracy": wu_acc,
        "test_top3_acc": wu_top3,
    })
    print(f"Run 1 test → loss: {wu_loss:.4f}  acc: {wu_acc:.4f}  top3: {wu_top3:.4f}")
    print(f"Run 1 ID: {run1.info.run_id}")

# ══════════════════════════════════════════════════════════════════════════
# MLflow Run 2 — Fine-tuning: unfreeze top-50 base layers
# ══════════════════════════════════════════════════════════════════════════
if args.fine_tune_epochs > 0:
    print("\n" + "=" * 60)
    print("MLflow Run 2 — mobilenetv2_finetune (top-50 layers unfrozen)")
    print("=" * 60)

    with mlflow.start_run(run_name="mobilenetv2_finetune") as run2:
        mlflow.log_params({
            "model":              "MobileNetV2",
            "phase":              "finetune",
            "img_size":           IMG,
            "batch_size":         BS,
            "fine_tune_epochs":   args.fine_tune_epochs,
            "fine_tune_lr":       args.fine_tune_lr,
            "unfrozen_layers":    50,
            "warmup_run_id":      run1.info.run_id,
            "num_classes":        num_classes,
            "train_samples":      len(paths_train),
            "val_samples":        len(paths_val),
            "test_samples":       len(paths_test),
        })

        # Load best warm-up weights, then unfreeze
        model.load_weights(best_ckpt)
        base_model.trainable = True
        for layer in base_model.layers[:-50]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(args.fine_tune_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy",
                     keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
        )

        warmup_epochs_done = len(history.history["loss"])
        callbacks_r2 = [
            keras.callbacks.ModelCheckpoint(
                best_ckpt, monitor="val_accuracy",
                save_best_only=True, verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=5,
                restore_best_weights=True, verbose=1,
            ),
            make_mlflow_callback(),
        ]

        model.fit(
            train_ds,
            initial_epoch=warmup_epochs_done,
            epochs=warmup_epochs_done + args.fine_tune_epochs,
            validation_data=val_ds,
            callbacks=callbacks_r2,
        )

        # Evaluate after fine-tuning
        model.load_weights(best_ckpt)
        ft_loss, ft_acc, ft_top3 = model.evaluate(test_ds, verbose=1)
        mlflow.log_metrics({
            "test_loss":     ft_loss,
            "test_accuracy": ft_acc,
            "test_top3_acc": ft_top3,
        })
        print(f"Run 2 test → loss: {ft_loss:.4f}  acc: {ft_acc:.4f}  top3: {ft_top3:.4f}")

        # Save final model + log to MLflow
        model.save(args.model_out)
        mlflow.log_artifact(args.model_out,  artifact_path="model")
        mlflow.log_artifact(args.labels_out, artifact_path="model")
        mlflow.keras.log_model(model, artifact_path="keras_model")
        print(f"Run 2 ID: {run2.info.run_id}")
else:
    # No fine-tuning: save best warm-up model
    model.load_weights(best_ckpt)
    model.save(args.model_out)
    with mlflow.start_run(run_name="mobilenetv2_finetune_skipped"):
        mlflow.log_param("note", "fine_tune_epochs=0; skipped")
        mlflow.log_artifact(args.model_out,  artifact_path="model")
        mlflow.log_artifact(args.labels_out, artifact_path="model")

# ── Cleanup ───────────────────────────────────────────────────────────────
if os.path.exists(best_ckpt):
    os.remove(best_ckpt)

print(f"\nFinal model saved → {args.model_out}")
print(f"Labels saved      → {args.labels_out}")
print(f"Tracking URI      → {mlflow.get_tracking_uri()}")
print("Done ✓")
