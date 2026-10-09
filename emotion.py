from sympy import python


python
import cv2
import time
from collections import deque
from deepface import DeepFace

# ---------------- SETTINGS ----------------
CAMERA_INDEX = 0
EMOTION_INTERVAL = 2.0
SMOOTHING_WINDOW = 5
CONFIDENCE_THRESHOLD = 65

# ---------------- LOAD FACE MODEL ----------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/face_model.yml")

with open("models/names.txt", "r", encoding="utf-8") as file:
    names = [line.strip() for line in file if line.strip()]

# ---------------- LOAD FACE DETECTOR ----------------
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Could not load Haar cascade face detector.")

# ---------------- START CAMERA ----------------
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam.")

emotion_history = deque(maxlen=SMOOTHING_WINDOW)
emotion_scores = {}
last_emotion_analysis = 0
current_emotion = "Analyzing..."
current_name = "Unknown"
current_distance = None

print("Face recognition + expression detection started.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read webcam frame.")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    # Reset identity when no face is visible
    if len(faces) == 0:
        current_name = "Unknown"
        current_distance = None
        current_emotion = "No face"
        emotion_scores = {}
        emotion_history.clear()

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]

        # -------- RECOGNIZE NAME --------
        label, distance = recognizer.predict(face_gray)
        current_distance = distance

        # LBPH distance is LOWER for a better match.
        # This threshold must be tuned using your own dataset.
        if 0 <= label < len(names) and distance < CONFIDENCE_THRESHOLD:
            current_name = names[label]
        else:
            current_name = "Unknown"

        # -------- DRAW FACE BOX --------
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # -------- ANALYZE EXPRESSION PERIODICALLY --------
        now = time.time()

        if now - last_emotion_analysis >= EMOTION_INTERVAL:
            last_emotion_analysis = now

            try:
                result = DeepFace.analyze(
                    img_path=frame,
                    actions=["emotion"],
                    detector_backend="opencv",
                    enforce_detection=False,
                    silent=True
                )

                if result:
                    scores = result[0].get("emotion", {})

                    if scores:
                        emotion_scores = scores
                        emotion_history.append(
                            max(scores, key=scores.get)
                        )

            except Exception as error:
                print("Expression analysis error:", error)

        if emotion_history:
            current_emotion = max(
                set(emotion_history),
                key=emotion_history.count
            )

        # -------- DISPLAY NAME AND EXPRESSION --------
        cv2.putText(
            frame,
            f"Name: {current_name}",
            (x, max(y - 35, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Expression: {current_emotion}",
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        if current_distance is not None:
            cv2.putText(
                frame,
                f"LBPH distance: {current_distance:.1f}",
                (x, y + h + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1
            )

    cv2.imshow("Face Recognition + Expression Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()