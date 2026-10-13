import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import math
import time
import os
import urllib.request

# Automatically download the required hand landmark model if not present
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("Download complete.")

# OS & PyAutoGUI Setup
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
CAM_WIDTH, CAM_HEIGHT = 640, 480

# Cursor Stabilization Parameters
SMOOTHING = 5.0
DEADZONE_PIXELS = 4.0
curr_x, curr_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
prev_x, prev_y = curr_x, curr_y

# State Trackers
is_mouse_down = False
right_click_active = False
last_action_time = 0
prev_dual_dist = None

# --- MediaPipe Tasks Setup ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.75,
    min_hand_presence_confidence=0.75,
    min_tracking_confidence=0.75
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)


def get_fingers_up(landmarks):
    """Accurately checks extended fingers relative to wrist (0) and MCP joints."""
    fingers = []
    wrist = landmarks[0]

    # Thumb: Distance from tip(4) to pinky MCP(17) vs IP(3) to pinky MCP(17)
    d_thumb_tip = math.hypot(landmarks[4].x - landmarks[17].x, landmarks[4].y - landmarks[17].y)
    d_thumb_ip = math.hypot(landmarks[3].x - landmarks[17].x, landmarks[3].y - landmarks[17].y)
    fingers.append(1 if d_thumb_tip > d_thumb_ip else 0)

    # 4 Fingers: Check if tip distance from wrist is greater than PIP distance from wrist
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        d_tip = math.hypot(landmarks[tip].x - wrist.x, landmarks[tip].y - wrist.y)
        d_pip = math.hypot(landmarks[pip].x - wrist.x, landmarks[pip].y - wrist.y)
        fingers.append(1 if d_tip > d_pip else 0)

    return fingers


print("[Ready] Dual-mode Iron Man system initialized. Press 'q' to quit.")

start_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert frame to MediaPipe Image format required by Tasks API
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.time() - start_time) * 1000)

    results = detector.detect_for_video(mp_image, timestamp_ms)

    current_time = time.time()
    status_text = "TRACKING"

    if results.hand_landmarks:
        num_hands = len(results.hand_landmarks)

        # ----------------- DUAL-HAND PINCH TO ZOOM -----------------
        if num_hands == 2:
            h1 = results.hand_landmarks[0]
            h2 = results.hand_landmarks[1]

            # Index tips for both hands
            p1 = (int(h1[8].x * w), int(h1[8].y * h))
            p2 = (int(h2[8].x * w), int(h2[8].y * h))

            cv2.line(frame, p1, p2, (0, 255, 255), 3)
            current_dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])

            if prev_dual_dist is not None:
                delta = current_dist - prev_dual_dist
                if abs(delta) > 25:
                    if delta > 0:
                        pyautogui.hotkey("ctrl", "+")
                        status_text = "ZOOM IN"
                    else:
                        pyautogui.hotkey("ctrl", "-")
                        status_text = "ZOOM OUT"
                    prev_dual_dist = current_dist
            else:
                prev_dual_dist = current_dist

            if is_mouse_down:
                pyautogui.mouseUp(button='left')
                is_mouse_down = False

        # ----------------- SINGLE HAND PRECISE CONTROL -----------------
        elif num_hands == 1:
            prev_dual_dist = None
            lm = results.hand_landmarks[0]

            fingers = get_fingers_up(lm)

            # Pixel landmarks
            ix, iy = int(lm[8].x * w), int(lm[8].y * h)  # Index Tip
            tx, ty = int(lm[4].x * w), int(lm[4].y * h)  # Thumb Tip
            mx, my = int(lm[12].x * w), int(lm[12].y * h)  # Middle Tip

            # Hand scale normalization (distance between wrist 0 and middle base 9)
            hand_scale = math.hypot(int(lm[0].x * w) - int(lm[9].x * w), int(lm[0].y * h) - int(lm[9].y * h))
            pinch_threshold = max(20, hand_scale * 0.28)

            # 1. SPECIAL POSE: 4/5 Fingers Open (Close Tab)
            if fingers == [0, 1, 1, 1, 1] or fingers == [1, 1, 1, 1, 1]:
                if current_time - last_action_time > 1.2:
                    pyautogui.hotkey("ctrl", "w")
                    status_text = "CLOSED TAB"
                    last_action_time = current_time

            # 2. SPECIAL POSE: Fist (0 Fingers) -> Switch App
            elif sum(fingers) == 0:
                if current_time - last_action_time > 1.0:
                    pyautogui.hotkey("alt", "tab")
                    status_text = "SWITCH APP"
                    last_action_time = current_time

            # 3. SPECIAL POSE: Peace Sign (Index + Middle Up) -> Scroll Mode
            elif fingers == [0, 1, 1, 0, 0]:
                scroll_amount = int((iy - (h // 2)) / 10)
                pyautogui.scroll(-scroll_amount)
                status_text = "SCROLLING"
                cv2.circle(frame, (ix, iy), 10, (255, 255, 0), cv2.FILLED)

            # 4. CURSOR NAVIGATION & CLICKS (Index Up or Pointing)
            else:
                target_x = int(lm[8].x * SCREEN_WIDTH)
                target_y = int((lm[8].y) * SCREEN_HEIGHT)

                target_x = max(0, min(SCREEN_WIDTH, target_x))
                target_y = max(0, min(SCREEN_HEIGHT, target_y))

                dx = target_x - prev_x
                dy = target_y - prev_y

                if math.hypot(dx, dy) > DEADZONE_PIXELS:
                    curr_x = prev_x + dx / SMOOTHING
                    curr_y = prev_y + dy / SMOOTHING
                    pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y

                # Left Click & Drag: Index + Thumb Pinch
                d_index_thumb = math.hypot(ix - tx, iy - ty)
                if d_index_thumb < pinch_threshold:
                    cv2.circle(frame, (ix, iy), 12, (0, 255, 0), cv2.FILLED)
                    if not is_mouse_down:
                        pyautogui.mouseDown(button='left')
                        is_mouse_down = True
                    status_text = "LEFT CLICK / DRAGGING"
                else:
                    if is_mouse_down:
                        pyautogui.mouseUp(button='left')
                        is_mouse_down = False

                # Right Click: Middle + Thumb Pinch
                d_middle_thumb = math.hypot(mx - tx, my - ty)
                if d_middle_thumb < pinch_threshold:
                    cv2.circle(frame, (mx, my), 12, (0, 0, 255), cv2.FILLED)
                    if not right_click_active:
                        pyautogui.rightClick()
                        right_click_active = True
                        status_text = "RIGHT CLICK"
                else:
                    right_click_active = False

    else:
        prev_dual_dist = None
        if is_mouse_down:
            pyautogui.mouseUp(button='left')
            is_mouse_down = False

    # HUD Status Display
    cv2.putText(frame, f"ACTION: {status_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Iron Man Control Engine", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()