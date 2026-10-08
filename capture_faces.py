import cv2
import os
import time

# Create dataset folder
os.makedirs("dataset", exist_ok=True)

# Ask for person's name
name = input("Enter your name: ")

# Create person's folder
person_folder = os.path.join("dataset", name)
os.makedirs(person_folder, exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open webcam
cap = cv2.VideoCapture(0)

count = 0

# Time between pictures
capture_delay = 0.5

last_capture_time = 0

print("\nStarting face capture...")
print("Move your head slowly.")
print("Look straight, left, right, up and down.")
print("Press Q to stop.\n")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not access webcam")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        current_time = time.time()

        # Capture only every 0.5 seconds
        if current_time - last_capture_time >= capture_delay:

            face = gray[y:y + h, x:x + w]

            filename = os.path.join(
                person_folder,
                f"{count}.jpg"
            )

            cv2.imwrite(filename, face)

            count += 1

            last_capture_time = current_time

        # Display progress
        cv2.putText(
            frame,
            f"Images: {count}/50",
            (x, y - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Registration", frame)

    # Stop after 50 images
    if count >= 50:
        break

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nDone! {count} face images saved.")
print(f"Location: {person_folder}")