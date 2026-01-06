import cv2
import mediapipe as mp
import numpy as np
import math
import time

# ---------- Hand setup ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ---------- Filters ----------
filters = ["Normal", "Warm", "Cool", "Matte", "Vintage", "Vignette"]
index = 0

snap_ready = False
last_snap = 0

def dist(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

# ---------- Filter functions (EASY) ----------
def warm(frame):
    frame[:,:,2] = cv2.add(frame[:,:,2], 30)
    return frame

def cool(frame):
    frame[:,:,0] = cv2.add(frame[:,:,0], 30)
    return frame

def matte(frame):
    return cv2.convertScaleAbs(frame, alpha=0.9, beta=25)

def vintage(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def vignette(frame):
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w//2, h//2), w//2, 255, -1)
    return cv2.bitwise_and(frame, frame, mask=mask)

# ---------- Main loop ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        for hand in res.multi_hand_landmarks:
            draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            thumb = hand.landmark[4]
            middle = hand.landmark[12]

            t = (int(thumb.x*w), int(thumb.y*h))
            m = (int(middle.x*w), int(middle.y*h))

            d = dist(t, m)

            # SNAP LOGIC
            if d < 30:
                snap_ready = True

            if d > 80 and snap_ready:
                if time.time() - last_snap > 0.8:
                    index = (index + 1) % len(filters)
                    last_snap = time.time()
                    snap_ready = False

    # ---------- Apply filter ----------
    if filters[index] == "Warm":
        frame = warm(frame)

    elif filters[index] == "Cool":
        frame = cool(frame)

    elif filters[index] == "Matte":
        frame = matte(frame)

    elif filters[index] == "Vintage":
        frame = vintage(frame)

    elif filters[index] == "Vignette":
        frame = vignette(frame)

    cv2.putText(frame, f"Filter : {filters[index]}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.imshow("Snap Cinematic Filters", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
