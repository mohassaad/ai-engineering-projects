
# ============================================================
# LIBRARIES WE NEED
# ============================================================

import os                     # For working with files and folders, 
                                #used for tools zay  Read folders → os.listdir() Build file paths → os.path.join()
                                # Delete files → os.remove()Get current directory → os.getcwd()



from PIL import Image         # For opening and processing images

import tensorflow as tf       # Main deep learning library
import matplotlib.pyplot as plt  # For drawing graphs

from tensorflow.keras import layers, models  # For building neural networks , the brain of the model
from tensorflow.keras.applications import MobileNetV2  # Pre-trained model,trained more than 1.4 mil pics for edges and textures (the eyes)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # For MobileNet model 

# ============================================================
# STEP 1: SETUP OUR SETTINGS
# ============================================================

# Find where this script is located, then add "PetImages" folder
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), #__file__ preimplemented variable
    "PetImages"
)

IMG_SIZE = (128, 128)  # All images will be resized to 128x128 pixels
BATCH_SIZE = 16        # Process 16 images at a time
SEED = 42              # da raqam used for shuffle in a mathematical formula, so it can mix both dogs and cats together and it doesn't matter the number 

print(" Starting Cat vs Dog Classification!")
print(f"Looking for images in: {DATASET_PATH}")


# ============================================================
# STEP 2: CLEAN UP BROKEN IMAGES
# ============================================================

print("\n🔍 Checking images for problems...")

# List to store any broken images we find
broken_images = []

# Check both Cat and Dog folders
for animal_type in ["Cat", "Dog"]: #loop among the entire folder until it ends
    
    # Get the full path to the folder (e.g., "PetImages/Cat")
    folder_path = os.path.join(DATASET_PATH, animal_type)
    
    # Go through every file in this folder
    for filename in os.listdir(folder_path):
        
        # Get full path to the image file #ba3deeha geeb el file nafso el path 
        file_path = os.path.join(folder_path, filename)
        
        try:
            # Try to open the image
            with Image.open(file_path) as img:
                img.load()  # Force it to actually read the image data, must do it with try bcz it can give exceptions
                
                # Make sure it's in RGB format (color images)
                if img.mode != "RGB":
                    img = img.convert("RGB") #momken yeb2a masalan png or L (grayscale), etc..
                
                # Save it back as a clean JPEG
                img.save(file_path, "JPEG", quality=95)
                
        except Exception as error:
            # If we couldn't open it, it's a broken image
            print(f" Broken image found: {file_path}")
            print(f"   Reason: {error}") #the exception class type in py, maybe IndexError, NameError, or etc ..
            broken_images.append(file_path)

# Delete all broken images
print(f"\n🗑️  -  Found {len(broken_images)} broken images. Deleting them...")

for file_path in broken_images: # haye3mal da law la2a ay 7aga broken
    try:
        os.remove(file_path)  # Delete the file
        print(f"   Deleted: {file_path}")
    except Exception as error:
        print(f"   Could not delete: {error}")

print("✅ Dataset cleaning complete!")


# ============================================================
# STEP 3: LOAD THE IMAGES INTO TENSORFLOW
# ============================================================

print("\n📥 Loading images into TensorFlow...")

# Load training images (80% of data) hena it understands the image 
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,              # Where images are stored
    validation_split=0.2,      # Use 20% for validation
    subset="training",         # This is the training portion
    seed=SEED,                 # For reproducibility, heya dee el random shuffiling used 
    image_size=IMG_SIZE,       # Resize all images to 128x128
    batch_size=BATCH_SIZE,     # Process 16 at a time
    label_mode="binary"        # 0 = Cat, 1 = Dog
)

# Load validation images (20% of data) hena is where the test happens after the understanding
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",       # This is the validation portion
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# Get the class names (should be ["Cat", "Dog"])
class_names = train_ds.class_names
print(f" -  Classes: {class_names}")

# Make loading faster where cpu and gpu works together for better performance , Load batch → while training on it, already loading the next batch in the background


AUTOTUNE = tf.data.AUTOTUNE #autotune means tensforlow bey7aded ad eh batches to preload based on my laptop's cpu , memory
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# ============================================================
# STEP 4: DATA AUGMENTATION (Create fake new images)
# ============================================================

# This helps the model learn better by slightly changing images 3ahsan beye3raf el angles 
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),  # Flip left-right
    layers.RandomRotation(0.1),       # Rotate a little
    layers.RandomZoom(0.1)            # Zoom in/out a little
])


# ============================================================
# STEP 5: BUILD THE MODEL
# ============================================================

print("\n Building the neural network...")

# Load MobileNetV2 - a model already trained on millions of images ~ 1.4m mil image
base_model = MobileNetV2(
    input_shape=(128, 128, 3),  # Input size (height, width, colors) - ya3ny expect rgb and 128x128 size 
    include_top=False,          # Remove the final classification layer, 3ashan we onlyu need 2 categroies not 1000
    weights="imagenet"          # Use weights trained on ImageNet, loads the images that it learned  ~ el 1.4 m image 
)



base_model.trainable = False    # Freeze the base model so we don't retrain it ,   making it TRUE will be  based on my data only (worse results conventionally), 
                                # making it FALSE will be based on his expertise too which is years of training from 1.4 m image

# Build our complete model
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),  # Input layer
    
    data_augmentation,                  #  Augment data
    
    layers.Lambda(preprocess_input),    # math function for translating it to the model 
    
    base_model,                         #   MobileNetV2 reading that translated image
    

    # HENA THE ENTIRE CLASSIFIER 
    layers.GlobalAveragePooling2D(),    # Average all features, makes it ready for the final decision layer.
    
    layers.Dropout(0.2),                # During training, randomly switches off 20% of neurons each batch, 
    #prevents the 100% accuracy and the 20% of the images model that it never trained on 
    
    layers.Dense(1, activation="sigmoid")  # Output: 0=Cat, 1=Dog
])

# Compile the model (prepare it for training)
model.compile(
    optimizer="adam",          # a7san algorithm for optimization of the model weights, adjusts the weights to get better results 
    loss="binary_crossentropy", # Loss function for binary classification
    metrics=["accuracy"]       # Track accuracy
)

model.summary()  # Show the model architecture, da 


# ============================================================
# STEP 6: PLOTTING THE GRAPH
# ============================================================

print("\n🏋️  -  Starting training...")

# Train for 3 epochs (passes through the data)
history = model.fit( #this is used for plotting the graph
    train_ds,         # Training data
    validation_data=val_ds,  # Validation data
    epochs=3          # Number of passes
)


# ============================================================
# STEP 7: EVALUATE THE MODEL
# ============================================================

print("\n Evaluating model...")

# Test the model on validation data
loss, accuracy = model.evaluate(val_ds)

print(f"\n - Validation Accuracy: {accuracy * 100:.2f}%")
print(f" -  Validation Loss: {loss:.4f}")


# ============================================================
# STEP 8: SAVE THE MODEL
# ============================================================

model.save("cat_dog_model.keras") #save it again in the file so everytime it trains from it 
print("\n💾 - Model saved as 'cat_dog_model.keras'")


# ============================================================
# STEP 9: TEST ON 5 RANDOM IMAGES
# ============================================================

print("\nTesting on 5 sample images...")

# Get one batch of images from validation set
test_images = []
test_labels = []

for images, labels in val_ds.take(1):  # Take just 1 batch
    # Take up to 5 images from this batch
    for i in range(min(5, len(images))):
        test_images.append(images[i])
        test_labels.append(int(labels[i].numpy().item()))

# Make predictions
predictions = model.predict(tf.stack(test_images), verbose=0)

# Check each prediction
correct_predictions = 0

print("\nResults:")
print("-" * 50)

for i in range(len(test_images)):
    # Get the prediction (value between 0 and 1)
    probability = float(predictions[i][0])
    
    # If probability > 0.5, it's a Dog (1), else Cat (0) - 3ashan e7na 2asemna dog as 1 
    predicted_label = 1 if probability >= 0.5 else 0
    actual_label = test_labels[i]
    
    # Get the class names
    predicted_class = class_names[predicted_label]
    actual_class = class_names[actual_label]
    
    # Check if correct
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

# Show summary
print(f"\n📊 Test Results: {correct_predictions}/{len(test_images)} correct")
print(f"🎯 Test Accuracy: {(correct_predictions/len(test_images)) * 100:.2f}%")


# ============================================================
# STEP 10: PLOT TRAINING PROGRESS
# ============================================================

print("\n📈 Creating training graph...")

# Create the graph
plt.figure(figsize=(8, 5))

# Plot training accuracy
plt.plot(
    history.history["accuracy"],
    label="Training Accuracy",
    marker='o'
)

# Plot validation accuracy
plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy",
    marker='s'
)

# Add labels and title
plt.title("🐱 vs 🐶 Training Progress", fontsize=16)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Save and show
plt.savefig("training_accuracy.png", dpi=150)
#plt.show() #REMOVE THE COMMENT LAW 3AYEZ TSHOOF EL GRAPH FEL VS CODE TERMINAL 

print(" -  Graph saved as 'training_accuracy.png'")
print("\n🎉 ALL DONE! Your cat vs dog classifier is ready!")

# ============================================================
# STEP 11: GRADIO WEB INTERFACE FOR THE WEBSITE LAUNCH
# ============================================================

import gradio as gr

def predict_image(img):
    img = img.resize((128, 128))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) #change the batch dimension to the model from 16 to 1 at position 0
    prediction = model.predict(img_array, verbose=0) #get the number betweeen 0 and 1 
    probability = float(prediction[0][0])
    if probability >= 0.5:
        label = "Dog"
        confidence = probability
    else:
        label = "Cat"
        confidence = 1 - probability
    return f"{label} ({confidence:.2%} confidence)" #make the format to percent by getting 2 decimal places


#the interface in the web , python to HTML conversion 3alatool
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Text(label="Prediction"),
    title="Cat vs Dog Classifier",
    description="Upload an image to classify it as a Cat or Dog"
)

interface.launch()
