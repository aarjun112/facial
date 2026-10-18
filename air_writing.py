
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
canvas = None
previous_point = None

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    print("Air writing started.")
    print("Use your index finger to draw.")
    print("Press C to clear, Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = np.zeros_like(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

            # Index fingertip is landmark 8.
            tip = hand.landmark[8]
            point = (
                int(tip.x * frame.shape[1]),
                int(tip.y * frame.shape[0])
            )

            # Draw only when the index finger is extended
            # and the middle finger is folded.
            index_up = (
                hand.landmark[8].y < hand.landmark[6].y
            )
            middle_folded = (
                hand.landmark[12].y > hand.landmark[10].y
            )

            if index_up and middle_folded:
                if previous_point is not None:
                    cv2.line(
                        canvas, previous_point, point,
                        (255, 255, 255), 8
                    )
                previous_point = point
            else:
                previous_point = None
        else:
            previous_point = None

        combined = cv2.addWeighted(frame, 1, canvas, 1, 0)

        cv2.putText(
            combined, "Draw with index finger | C: Clear | Q: Quit",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 0), 2
        )

        cv2.imshow("JARVIS Air Writing - Hand Tracking", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas = np.zeros_like(frame)
            previous_point = None

cap.release()
cv2.destroyAllWindows()
