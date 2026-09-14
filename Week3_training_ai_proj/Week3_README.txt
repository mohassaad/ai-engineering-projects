Week 3 - Cat vs Dog Classifier
================================
Mohamed Assaad

WHAT THIS PROJECT DOES
-----------------------
Trains a deep learning model using MobileNetV2 to classify images as either
a Cat or a Dog. After training, a Gradio web interface launches in the browser
where you can upload any image and get a prediction with confidence percentage.

FOLDER STRUCTURE
-----------------
Week3_training_ai_proj/
├── main.py               → the main code
├── requirements.txt      → all required libraries
├── PetImages/            → dataset (Cat/ and Dog/ folders)
│   ├── Cat/              → ~12,000 cat images
│   └── Dog/              → ~12,000 dog images
│
│ These are generated automatically after running:
├── cat_dog_model.keras   → the saved trained model
└── training_accuracy.png → accuracy graph

NOTE: PetImages dataset (~800MB) must be downloaded separately from:
https://www.microsoft.com/en-us/download/details.aspx?id=54765
Extract it and place the PetImages folder inside this project folder.

HOW TO RUN
-----------
Run these commands one by one in the terminal:

    cd ..
    cd Week3_training_ai_proj
    dir
    python -m venv venv
    venv\Scripts\activate
    pip install tensorflow gradio pillow matplotlib
    python main.py

Then open your browser and go to:
    http://127.0.0.1:7860

WHAT HAPPENS WHEN YOU RUN IT
------------------------------
1. Scans and removes any broken images from PetImages
2. Loads and splits images (80% training, 20% validation)
3. Trains the MobileNetV2 model for 3 epochs
4. Saves the model as cat_dog_model.keras
5. Tests on 5 random images and prints results
6. Saves training accuracy graph as training_accuracy.png
7. Launches Gradio web interface at http://127.0.0.1:7860

NOTE: Training takes 5-15 minutes depending on your machine.
Do not close the terminal while it is running.

LIBRARIES USED
---------------
TensorFlow, Keras, MobileNetV2, Gradio, Pillow, Matplotlib