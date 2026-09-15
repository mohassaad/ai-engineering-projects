# ============================================================
# LIBRARIES WE NEED
# ============================================================

import os                     # files and folders: os.listdir(), os.path.join(),
                              # os.remove(), os.getcwd()
from PIL import Image         # open and process images

import tensorflow as tf       # main deep learning library
import matplotlib.pyplot as plt  # draw graphs

from tensorflow.keras import layers, models  # build the network (the brain)
from tensorflow.keras.applications import MobileNetV2  # pretrained on 1.4M+ pics (edges, textures — the eyes)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # MobileNet input prep

# ============================================================
# STEP 1: SETUP OUR SETTINGS
# ============================================================

# script folder + "PetImages"
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), # __file__ is a preimplemented variable
    "PetImages"
)

IMG_SIZE = (128, 128)  # resize everything to 128x128
BATCH_SIZE = 16        # process 16 images at a time
SEED = 42              # da raqam used for shuffle, mixes cats and dogs together — the exact value doesn't matter

print(" Starting Cat vs Dog Classification!")
print(f"Looking for images in: {DATASET_PATH}")


# ============================================================
# STEP 2: CLEAN UP BROKEN IMAGES
# ============================================================

print("\n🔍 Checking images for problems...")

# any broken images we find
broken_images = []

# check both Cat and Dog folders
for animal_type in ["Cat", "Dog"]: # loop through the whole folder

    # full path to the folder (e.g. "PetImages/Cat")
    folder_path = os.path.join(DATASET_PATH, animal_type)

    # go through every file in this folder
    for filename in os.listdir(folder_path):

        # full path to the image file # ba3deeha geeb el file nafso el path
        file_path = os.path.join(folder_path, filename)

        try:
            # try to open the image
            with Image.open(file_path) as img:
                img.load()  # force it to read the data — must be in try bcz it can throw

                # make sure it's RGB
                if img.mode != "RGB":
                    img = img.convert("RGB") # momken yeb2a masalan png or L (grayscale), etc..

                # save it back as a clean JPEG
                img.save(file_path, "JPEG", quality=95)

        except Exception as error:
            # couldn't open it -> broken
            print(f" Broken image found: {file_path}")
            print(f"   Reason: {error}") # the exception type in py — IndexError, NameError, etc..
            broken_images.append(file_path)

# delete all broken images
print(f"\n🗑️  -  Found {len(broken_images)} broken images. Deleting them...")

for file_path in broken_images: # haye3mal da law la2a ay 7aga broken
    try:
        os.remove(file_path)  # delete the file
        print(f"   Deleted: {file_path}")
    except Exception as error:
        print(f"   Could not delete: {error}")

print("✅ Dataset cleaning complete!")


# ============================================================
# STEP 3: LOAD THE IMAGES INTO TENSORFLOW
# ============================================================

print("\n📥 Loading images into TensorFlow...")

# training images (80%) — hena it understands the image
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,              # where images live
    validation_split=0.2,      # 20% for validation
    subset="training",         # this is the training portion
    seed=SEED,                 # reproducibility — heya dee el random shuffling
    image_size=IMG_SIZE,       # resize to 128x128
    batch_size=BATCH_SIZE,     # 16 at a time
    label_mode="binary"        # 0 = Cat, 1 = Dog
)

# validation images (20%) — hena is where the test happens after the understanding
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",       # this is the validation portion
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# class names (should be ["Cat", "Dog"])
class_names = train_ds.class_names
print(f" -  Classes: {class_names}")

# CPU + GPU overlap for speed: while training on a batch, load the next one in the background

AUTOTUNE = tf.data.AUTOTUNE # autotune = tf picks how many batches to preload based on my laptop's cpu / memory
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# ============================================================
# STEP 4: DATA AUGMENTATION (Create fake new images)
# ============================================================

# helps the model learn better by slightly changing images — 3ashan beye3raf el angles
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),  # flip left-right
    layers.RandomRotation(0.1),       # rotate a little
    layers.RandomZoom(0.1)            # zoom in/out a little
])


# ============================================================
# STEP 5: BUILD THE MODEL
# ============================================================

print("\n Building the neural network...")

# MobileNetV2 — already trained on ~1.4M images
base_model = MobileNetV2(
    input_shape=(128, 128, 3),  # (height, width, colors) — ya3ny expects rgb and 128x128
    include_top=False,          # drop the final classification layer — 3ashan we only need 2 categories not 1000
    weights="imagenet"          # use ImageNet weights — the ~1.4M image knowledge
)



base_model.trainable = False    # freeze the base so we don't retrain it.
                                # True  -> trained on my data only (usually worse).
                                # False -> keeps its years of expertise from 1.4M images.

# full model
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),  # input layer

    data_augmentation,                  # augment data

    layers.Lambda(preprocess_input),    # math function that translates the image for the model

    base_model,                         # MobileNetV2 reads the translated image


    # HENA THE ENTIRE CLASSIFIER
    layers.GlobalAveragePooling2D(),    # average all features, ready for the final decision

    layers.Dropout(0.2),                # during training, randomly switch off 20% of neurons per batch —
    # prevents the 100% accuracy trap and overfitting to the 20% it never trained on

    layers.Dense(1, activation="sigmoid")  # output: 0=Cat, 1=Dog
])

# compile (prepare for training)
model.compile(
    optimizer="adam",          # a7san optimizer for adjusting the model weights
    loss="binary_crossentropy", # loss for binary classification
    metrics=["accuracy"]       # track accuracy
)

model.summary()  # show the architecture, da


# ============================================================
# STEP 6: PLOTTING THE GRAPH
# ============================================================

print("\n🏋️  -  Starting training...")

# 3 epochs (passes through the data)
history = model.fit( # this is used for plotting the graph
    train_ds,         # training data
    validation_data=val_ds,  # validation data
    epochs=3          # number of passes
)


# ============================================================
# STEP 7: EVALUATE THE MODEL
# ============================================================

print("\n Evaluating model...")

# test on validation data
loss, accuracy = model.evaluate(val_ds)

print(f"\n - Validation Accuracy: {accuracy * 100:.2f}%")
print(f" -  Validation Loss: {loss:.4f}")


# ============================================================
# STEP 8: SAVE THE MODEL
# ============================================================

model.save("cat_dog_model.keras") # save it so every run can start from it
print("\n💾 - Model saved as 'cat_dog_model.keras'")


# ============================================================
# STEP 9: TEST ON 5 RANDOM IMAGES
# ============================================================

print("\nTesting on 5 sample images...")

# grab one batch from validation
test_images = []
test_labels = []

for images, labels in val_ds.take(1):  # just 1 batch
    # up to 5 images from this batch
    for i in range(min(5, len(images))):
        test_images.append(images[i])
        test_labels.append(int(labels[i].numpy().item()))

# predictions
predictions = model.predict(tf.stack(test_images), verbose=0)

# check each prediction
correct_predictions = 0

print("\nResults:")
print("-" * 50)

for i in range(len(test_images)):
    # prediction value between 0 and 1
    probability = float(predictions[i][0])

    # >0.5 -> Dog (1), else Cat (0) — 3ashan e7na 2asemna dog as 1
    predicted_label = 1 if probability >= 0.5 else 0
    actual_label = test_labels[i]

    # class names
    predicted_class = class_names[predicted_label]
    actual_class = class_names[actual_label]

    # correct?
    if predicted_label == actual_label:
        result = "✅ CORRECT"
        correct_predictions += 1
    else:
        result = "❌ WRONG"

    print(f"Image {i+1}:")
    print(f"   Predicted: {predicted_class} (confidence: {probability:.2%})")
    print(f"   Actual:    {actual_class}")
    print(f"   Result:    {result}")
    print("-" * 50)

# summary
print(f"\n📊 Test Results: {correct_predictions}/{len(test_images)} correct")
print(f"🎯 Test Accuracy: {(correct_predictions/len(test_images)) * 100:.2f}%")


# ============================================================
# STEP 10: PLOT TRAINING PROGRESS
# ============================================================

print("\n📈 Creating training graph...")

# graph
plt.figure(figsize=(8, 5))

# training accuracy
plt.plot(
    history.history["accuracy"],
    label="Training Accuracy",
    marker='o'
)

# validation accuracy
plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy",
    marker='s'
)

# labels and title
plt.title("🐱 vs 🐶 Training Progress", fontsize=16)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# save and show
plt.savefig("training_accuracy.png", dpi=150)
#plt.show() # REMOVE THE COMMENT LAW 3AYEZ TSHOOF EL GRAPH FEL VS CODE TERMINAL

print(" -  Graph saved as 'training_accuracy.png'")
print("\n🎉 ALL DONE! Your cat vs dog classifier is ready!")

# ============================================================
# STEP 11: GRADIO WEB INTERFACE FOR THE WEBSITE LAUNCH
# ============================================================

import gradio as gr

def predict_image(img):
    img = img.resize((128, 128))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # batch dim from 16 to 1 at position 0
    prediction = model.predict(img_array, verbose=0) # number between 0 and 1
    probability = float(prediction[0][0])
    if probability >= 0.5:
        label = "Dog"
        confidence = probability
    else:
        label = "Cat"
        confidence = 1 - probability
    return f"{label} ({confidence:.2%} confidence)" # format to percent with 2 decimals


# web UI — python to HTML conversion 3alatool
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Text(label="Prediction"),
    title="Cat vs Dog Classifier",
    description="Upload an image to classify it as a Cat or Dog"
)

interface.launch()
