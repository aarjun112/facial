import cv2

# Load trained face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/face_model.yml")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load names
names = []

with open("models/names.txt", "r") as file:
    for line in file:
        names.append(line.strip())

# Open webcam
cap = cv2.VideoCapture(0)

print("Face recognition started...")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not access webcam")
        break

    # Convert webcam image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        # Recognize face
        face = gray[y:y + h, x:x + w]

        label, confidence = recognizer.predict(face)

        # Convert LBPH confidence to approximate percentage
        confidence_percent = max(
            0,
            min(100, 100 - confidence)
        )

        # Decide if face is known
        if confidence < 70 and label < len(names):

            name = names[label]

        else:

            name = "Unknown"

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display name
        cv2.putText(
            frame,
            name,
            (x, y - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        # Display confidence
        cv2.putText(
            frame,
            f"Confidence: {confidence_percent:.1f}%",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Display webcam
    cv2.imshow("Face Recognition", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()