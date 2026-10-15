
import cv2
import time
from ultralytics import YOLO

# Load a small pretrained object-detection model.
# The model downloads automatically on first run.
model = YOLO("yolo26n.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

frame_count = 0
last_inference = 0
cached_objects = []
fps = 0
previous_time = time.time()

print("Object and shape detection started. Press Q to quit.")

def detect_shapes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    height, width = frame.shape[:2]

    for contour in contours:
        area = cv2.contourArea(contour)

        # Ignore tiny edges and very large outlines.
        if area < 1500 or area > (height * width * 0.30):
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        approx = cv2.approxPolyDP(
            contour, 0.03 * perimeter, True
        )
        vertices = len(approx)

        if vertices == 3:
            shape = "Triangle"
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / float(h)

            if 0.90 <= ratio <= 1.10:
                shape = "Square"
            else:
                shape = "Rectangle"
        elif vertices == 5:
            shape = "Pentagon"
        elif vertices == 6:
            shape = "Hexagon"
        elif vertices >= 7:
            shape = "Circle / Round"
        else:
            continue

        x, y, w, h = cv2.boundingRect(approx)

        cv2.drawContours(frame, [approx], -1, (255, 180, 0), 2)
        cv2.putText(
            frame, shape, (x, max(y - 7, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 180, 0), 2
        )


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read webcam.")
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1
        now = time.time()

        # Run YOLO every 0.25 seconds to reduce workload.
        if now - last_inference >= 0.25:
            last_inference = now

            results = model.predict(
                source=frame,
                imgsz=416,
                conf=0.35,
                verbose=False
            )

            cached_objects = []

            for box in results[0].boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                cached_objects.append(
                    (x1, y1, x2, y2, label, confidence)
                )

        # Draw the latest object predictions.
        for x1, y1, x2, y2, label, confidence in cached_objects:
            cv2.rectangle(
                frame, (x1, y1), (x2, y2), (0, 255, 0), 2
            )
            cv2.putText(
                frame,
                f"{label}: {confidence:.0%}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        # Detect geometric outlines.
        detect_shapes(frame)

        # Approximate FPS display.
        elapsed = now - previous_time
        if elapsed > 0:
            fps = 0.9 * fps + 0.1 / elapsed if fps else 1 / elapsed
        previous_time = now

        cv2.putText(
            frame, f"FPS: {fps:.1f}", (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )
        cv2.putText(
            frame, "Press Q to quit", (15, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        cv2.imshow("Object and Shape Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()