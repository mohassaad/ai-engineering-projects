# ============================================================
# CHEST X-RAY CLASSIFIER (4 classes) — LEAKAGE-RESISTANT
#   COVID19 / NORMAL / PNEUMONIA / TURBERCULOSIS
#
# THE REAL PROBLEM this version fixes:
#   Earlier the model was cheating — it learned "which dataset a
#   picture came from" (scanner/contrast/crop style) instead of the
#   actual lung disease. That's why a TB scan got called Normal and a
#   Normal scan got called COVID: the *style* didn't match what it had
#   memorised for that label. This is called SOURCE / DATASET LEAKAGE.
#
# WHAT THIS VERSION DOES ABOUT IT:
#   1) GRAYSCALE input. X-rays are grayscale; different datasets have
#      subtle colour/tint casts that the model was using as a shortcut.
#      Converting to grayscale (then back to 3 channels for MobileNetV2)
#      removes that cue and FORCES the model onto real lung structure.
#   2) NO STRETCHING. Images used to be squashed into a 128x128 square,
#      so a wide pneumonia scan (one dataset) looked different from a
#      square 512x512 TB scan (another dataset) — the SHAPE became a
#      shortcut. Now dark borders are trimmed and the centre square is
#      cut out instead. multimodal_ui.py runs the exact same
#      prepare_xray() on uploads, so the app sees what training saw.
#   3) MILD (sqrt) class weighting, so it doesn't over-predict rare
#      classes.
#   4) Honest evaluation on the big test/ folder + confusion matrix
#      + per-class accuracy. Run remove_duplicates.py first so no test
#      image is also sitting in train/.
#
# NOTE: the biggest remaining fix is DATA, not code — you must mix
#   image sources for every class (especially add ADULT NORMAL images:
#   every current NORMAL image is a child's X-ray), split into train AND
#   test in the same ratio (add_data.py does this).
#
# EDUCATIONAL DEMO ONLY — not a medical diagnostic tool.
# ============================================================

import os
import shutil
import numpy as np
from PIL import Image

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
# STEP 1: SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "chest_xray")
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
TEST_DIR  = os.path.join(DATASET_PATH, "test")

MODEL_PATH       = os.path.join(BASE_DIR, "cat_dog_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.txt")
APP_DIR          = os.path.join(BASE_DIR, "..", "Week4_multimodal_agent_proj")  # gets a copy of the model

IMG_SIZE        = (128, 128)   # keep 128 — matches the app; simple and stable
BATCH_SIZE      = 16
SEED            = 42
VAL_SPLIT       = 0.2
HEAD_EPOCHS     = 10
FINETUNE_EPOCHS = 12
FINETUNE_LAYERS = 40

# dark-border trimming — must match multimodal_ui.py
BORDER_LEVEL    = 20.0   # pixels darker than this (0-255) count as border
BORDER_FRACTION = 0.05   # a row/column with fewer than 5% brighter pixels is border

print(" Chest X-Ray Classification — leakage-resistant (grayscale, no stretching)")
print(f"Dataset: {DATASET_PATH}")

for name, path in [("train", TRAIN_DIR), ("test", TEST_DIR)]:
    if not os.path.isdir(path):
        raise SystemExit(
            f"Could not find '{name}' at: {path}\n"
            f"Make sure chest_xray/ (with train/ and test/) sits next to this script."
        )


# ============================================================
# STEP 2: CLEAN BROKEN IMAGES
# ============================================================

print("\n Checking images...")
broken = []
for split_dir in [TRAIN_DIR, TEST_DIR]:
    for disease in os.listdir(split_dir):
        folder = os.path.join(split_dir, disease)
        if not os.path.isdir(folder):
            continue
        for fn in os.listdir(folder):
            fp = os.path.join(folder, fn)
            try:
                with Image.open(fp) as im:
                    im.load()
                    if im.mode != "RGB":
                        im.convert("RGB").save(fp, "JPEG", quality=95)
            except Exception as e:
                print(f" Broken: {fp} ({e})")
                broken.append(fp)
for fp in broken:
    try:
        os.remove(fp)
    except Exception:
        pass
print(f" Cleaned ({len(broken)} removed).")


# ============================================================
# STEP 3: LOAD DATA (val split from train; test = honest report)
# ============================================================

# ---- ONE preprocessing function, shared with the app ----
# multimodal_ui.py has an IDENTICAL copy of prepare_xray(). If the app
# prepares images even slightly differently from training, predictions
# silently get worse — so change both or neither.
def prepare_xray(rgb):
    """RGB image (h, w, 3) -> grayscale -> dark borders trimmed -> centre square
    (no stretching) -> IMG_SIZE, as 3 identical channels in [0, 255]."""
    # grayscale kills colour/tint leakage; MobileNetV2 still needs 3 channels
    gray = tf.image.rgb_to_grayscale(tf.cast(rgb, tf.float32))

    # trim near-black bars (screenshot letterboxing, scanner margins)
    content = tf.cast(gray[..., 0] > BORDER_LEVEL, tf.float32)
    rows = tf.where(tf.reduce_mean(content, axis=1) > BORDER_FRACTION)[:, 0]
    cols = tf.where(tf.reduce_mean(content, axis=0) > BORDER_FRACTION)[:, 0]
    big_enough = tf.logical_and(tf.size(rows) * 3 >= tf.shape(gray)[0],   # only crop if at least
                                tf.size(cols) * 3 >= tf.shape(gray)[1])   # 1/3 of the picture is left
    trimmed = tf.cond(big_enough,
                      lambda: gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1],
                      lambda: gray)

    # cut out the centre square instead of stretching, then shrink.
    # (not padding: black bars would appear on the wide child X-rays only,
    #  and the model would learn "bars = Normal/Pneumonia" — a new shortcut)
    side = tf.minimum(tf.shape(trimmed)[0], tf.shape(trimmed)[1])
    square = tf.image.resize_with_crop_or_pad(trimmed, side, side)
    small = tf.image.resize(square, IMG_SIZE, antialias=True)
    return tf.image.grayscale_to_rgb(small)


print("\n Loading images...")
class_names = sorted(d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d)))
NUM_CLASSES = len(class_names)
print(f" Classes: {class_names}")

IMAGE_EXTS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")   # formats TensorFlow can decode

def list_images(split_dir):
    paths, labels = [], []
    for i, cls in enumerate(class_names):
        folder = os.path.join(split_dir, cls)
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith(IMAGE_EXTS):
                paths.append(os.path.join(folder, fn))
                labels.append(i)
    return np.array(paths), np.array(labels)

def load_image(path, label):
    rgb = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    rgb.set_shape([None, None, 3])
    return prepare_xray(rgb), tf.one_hot(label, NUM_CLASSES)

AUTOTUNE = tf.data.AUTOTUNE

def make_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    # cache(): the slow decode/trim/pad work only happens in the first epoch
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE).cache()
    if shuffle:
        ds = ds.shuffle(1000, seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

all_paths, all_labels = list_images(TRAIN_DIR)
order = np.random.RandomState(SEED).permutation(len(all_paths))
n_val = int(len(order) * VAL_SPLIT)
val_idx, train_idx = order[:n_val], order[n_val:]
test_paths, test_labels = list_images(TEST_DIR)

train_ds = make_dataset(all_paths[train_idx], all_labels[train_idx], shuffle=True)
val_ds   = make_dataset(all_paths[val_idx], all_labels[val_idx])
test_ds  = make_dataset(test_paths, test_labels)
print(f" train: {len(train_idx)}  |  val: {n_val}  |  test: {len(test_paths)}")


# ============================================================
# STEP 4: MILD (SQRT) CLASS WEIGHTS
# ============================================================

print("\n Computing sqrt class weights...")
counts = []
for cls in class_names:
    folder = os.path.join(TRAIN_DIR, cls)
    n = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
    counts.append(n)
    print(f"   {cls}: {n} train images")
counts = np.array(counts, dtype=float)
raw = counts.sum() / (NUM_CLASSES * counts)
soft = np.sqrt(raw); soft = soft / soft.mean()
class_weight = {i: float(soft[i]) for i in range(NUM_CLASSES)}
print(f"   weights: {class_weight}")


# ============================================================
# STEP 5: AUGMENTATION (upright X-rays: no horizontal flip)
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.05, 0.05),
])


# ============================================================
# STEP 6: MODEL
# ============================================================

print("\n Building model...")
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights="imagenet")
base_model.trainable = False

inputs = layers.Input(shape=(128, 128, 3))
x = data_augmentation(inputs)
x = layers.Lambda(preprocess_input)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs)

loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss_fn, metrics=["accuracy"])
model.summary()

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
]


# ============================================================
# STEP 7: PHASE 1 — head only
# ============================================================
print("\n Phase 1: training the head...")
h1 = model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS,
               class_weight=class_weight, callbacks=callbacks)


# ============================================================
# STEP 8: PHASE 2 — fine-tune top layers
# ============================================================
print(f"\n Phase 2: fine-tuning last {FINETUNE_LAYERS} layers...")
base_model.trainable = True
for layer in base_model.layers[:-FINETUNE_LAYERS]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=loss_fn, metrics=["accuracy"])
h2 = model.fit(train_ds, validation_data=val_ds, epochs=FINETUNE_EPOCHS,
               class_weight=class_weight, callbacks=callbacks)


# ============================================================
# STEP 9: SAVE (same filename the app loads) + copy into the app
# ============================================================
model.save(MODEL_PATH)
with open(CLASS_NAMES_PATH, "w") as f:
    f.write("\n".join(class_names))
print("\n Saved -> 'cat_dog_model.keras'  |  order -> 'class_names.txt'")

# The app loads ITS OWN copy of the model, so copy the new one over —
# otherwise the app keeps using an old model after you retrain.
if os.path.isdir(APP_DIR):
    try:
        shutil.copy2(MODEL_PATH, APP_DIR)
        shutil.copy2(CLASS_NAMES_PATH, APP_DIR)
        print(" Copied both into Week4_multimodal_agent_proj/ — restart multimodal_ui.py to load it.")
    except OSError as e:
        print(f" Could not copy into Week4_multimodal_agent_proj/ ({e}) — copy cat_dog_model.keras by hand.")


# ============================================================
# STEP 10: HONEST TEST REPORT
# ============================================================
print("\n Final evaluation on TEST:")
tl, ta = model.evaluate(test_ds)
print(f"   Test accuracy: {ta*100:.2f}%   loss: {tl:.4f}")

y_true, y_pred = [], []
for images, labels in test_ds:
    p = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(p, axis=1))
y_true, y_pred = np.array(y_true), np.array(y_pred)

cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t][p] += 1

print("\n Confusion matrix (rows=actual, cols=predicted):")
print("actual\\pred  " + "  ".join(f"{c[:6]:>6}" for c in class_names))
for i, c in enumerate(class_names):
    print(f"{c[:10]:<11} " + "  ".join(f"{cm[i][j]:>6}" for j in range(NUM_CLASSES)))

print("\n Per-class accuracy (TEST):")
for i, c in enumerate(class_names):
    tot = cm[i].sum(); ok = cm[i][i]
    print(f"   {c}: {(ok/tot*100 if tot else 0):.1f}%  ({ok}/{tot})")


# ============================================================
# STEP 11: PLOT
# ============================================================
acc  = h1.history["accuracy"] + h2.history["accuracy"]
vacc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
plt.figure(figsize=(8, 5))
plt.plot(acc, label="Training Accuracy", marker="o")
plt.plot(vacc, label="Validation Accuracy", marker="s")
plt.axvline(len(h1.history["accuracy"]) - 0.5, color="grey", linestyle="--", alpha=0.6,
            label="fine-tuning starts")
plt.title("Chest X-Ray Training Progress"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(BASE_DIR, "training_accuracy.png"), dpi=150)
print("\n Graph saved -> 'training_accuracy.png'")
print(" DONE. Educational demo only — not for real diagnosis.")
