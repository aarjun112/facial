import cv2
import os
import numpy as np

# Face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []

# Person ID
label_id = 0

# Dataset folder
dataset_path = "dataset"

# Go through each person's folder
for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    print(f"Training: {person_name}")

    # Read every image
    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        # Read image as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        # Add face image
        faces.append(image)

        # Add corresponding person's ID
        labels.append(label_id)

    label_id += 1

# Train recognizer
print("\nTraining model...")

recognizer.train(
    faces,
    np.array(labels)
)

# Create models folder
os.makedirs("models", exist_ok=True)

# Save trained model
recognizer.save("models/face_model.yml")

# Save names
with open("models/names.txt", "w") as file:

    for person_name in os.listdir(dataset_path):

        person_folder = os.path.join(dataset_path, person_name)

        if os.path.isdir(person_folder):
            file.write(person_name + "\n")

print("\nTraining complete!")
print("Model saved to: models/face_model.yml")