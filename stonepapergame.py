import cv2
import mediapipe as mp
import random
import time
from collections import deque, Counter

# ── Mediapipe setup ────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)
mp_draw = mp.solutions.drawing_utils

# ── Game state ─────────────────────────────────────────────────────────────────
choices = ["Rock", "Paper", "Scissors"]
user_score, hemant_score, round_num = 0, 0, 1
game_start = False
t0 = 0.0
user_move = "None"
bot_move = "None"
result_text = ""
gesture_buffer = deque(maxlen=30)  # ~1 second of votes at ~30 FPS

# ── Gesture classifier (handedness-aware thumb) ────────────────────────────────
def classify_gesture(landmarks, handed_label):
    """
    landmarks: normalized landmarks (0..1) from mediapipe
    handed_label: 'Left' or 'Right' from mediapipe
    returns: 'Rock' | 'Paper' | 'Scissors' | None
    """
    # Finger open test: tip.y < pip.y (image origin is top-left)
    def finger_open(tip, pip):
        return landmarks[tip].y < landmarks[pip].y

    # Thumb open test depends on handedness: compare x positions
    # In the displayed (flipped) image:
    #  - Right hand thumb points to the LEFT when open (tip.x < ip.x)
    #  - Left hand  thumb points to the RIGHT when open (tip.x > ip.x)
    if handed_label == "Right":
        thumb_open = landmarks[4].x < landmarks[3].x
    else:  # "Left"
        thumb_open = landmarks[4].x > landmarks[3].x

    index_open  = finger_open(8, 6)
    middle_open = finger_open(12, 10)
    ring_open   = finger_open(16, 14)
    pinky_open  = finger_open(20, 18)

    # Map to gestures (explicit patterns)
    if (not thumb_open and not index_open and not middle_open and
        not ring_open and not pinky_open):
        return "Rock"

    if (thumb_open and index_open and middle_open and ring_open and pinky_open):
        return "Paper"

    if (not thumb_open and index_open and middle_open and
        not ring_open and not pinky_open):
        return "Scissors"

    return None

# ── Winner logic ───────────────────────────────────────────────────────────────
def decide_winner(user, bot):
    if user == bot:
        return "Tie"
    if (user == "Rock" and bot == "Scissors") or \
       (user == "Scissors" and bot == "Paper") or \
       (user == "Paper" and bot == "Rock"):
        return "You Win!"
    return "Hemant Wins!"

# ── Camera ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # ── Hand tracking ─────────────────────────────────────────────────────────
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        # zip with handedness to keep alignment
        for hand_lms, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            label = handed.classification[0].label  # 'Left' or 'Right'

            # During the "capture window" (3s..4s), collect votes
            if game_start:
                elapsed = time.time() - t0
                if 3.0 <= elapsed < 4.0:
                    g = classify_gesture(hand_lms.landmark, label)
                    if g:
                        gesture_buffer.append(g)

    # ── Round flow: press 's' → countdown 3s → 1s capture → score ─────────────
    if not game_start:
        cv2.putText(frame, "Press 's' to Start Round", (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    else:
        elapsed = time.time() - t0
        if elapsed < 3.0:
            # Countdown
            cv2.putText(frame, f"Get Ready: {int(3 - elapsed)}", (40, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        elif elapsed < 4.0:
            # Capture window indicator
            cv2.putText(frame, "Show your move!", (40, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        else:
            # Decide from votes
            if len(gesture_buffer) > 0:
                user_move = Counter(gesture_buffer).most_common(1)[0][0]
            else:
                user_move = "None"

            bot_move = random.choice(choices)
            if user_move in choices:
                result_text = decide_winner(user_move, bot_move)
                if result_text == "You Win!":
                    user_score += 1
                elif result_text == "Hemant Wins!":
                    hemant_score += 1
            else:
                result_text = "No valid move"

            round_num += 1
            game_start = False
            gesture_buffer.clear()

    # ── Cool Dashboard (card on the right) ────────────────────────────────────
    dash_x, dash_y, dash_w, dash_h = w - 380, 40, 340, 620
    cv2.rectangle(frame, (dash_x, dash_y), (dash_x + dash_w, dash_y + dash_h), (45, 45, 45), -1)
    cv2.rectangle(frame, (dash_x, dash_y), (dash_x + dash_w, dash_y + dash_h), (255, 255, 255), 2)

    cv2.putText(frame, "GAME DASHBOARD", (dash_x + 24, dash_y + 48),
                cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 255), 2)

    cv2.putText(frame, f"Round: {round_num}", (dash_x + 24, dash_y + 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (220, 220, 220), 2)

    # Scores
    cv2.putText(frame, f"You: {user_score}", (dash_x + 24, dash_y + 170),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 0), 3)
    cv2.putText(frame, f"Hemant: {hemant_score}", (dash_x + 24, dash_y + 230),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 220), 3)

    # Status line
    if user_score > hemant_score:
        status, color = "You're Leading!", (0, 220, 0)
    elif hemant_score > user_score:
        status, color = "Hemant is Winning!", (0, 0, 220)
    else:
        status, color = "It's a Tie!", (0, 220, 220)
    cv2.putText(frame, status, (dash_x + 24, dash_y + 290),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

    # Moves & last result
    cv2.putText(frame, f"Your Move: {user_move}", (dash_x + 24, dash_y + 360),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Hemant's Move: {bot_move}", (dash_x + 24, dash_y + 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Result: {result_text}", (dash_x + 24, dash_y + 450),
                cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)

    # Controls hint
    cv2.putText(frame, "Press 's' to start, 'q' to quit", (dash_x + 24, dash_y + dash_h - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # ── Show window ───────────────────────────────────────────────────────────
    cv2.imshow("Rock Paper Scissors", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and not game_start:
        game_start = True
        t0 = time.time()
        user_move, bot_move, result_text = "None", "None", ""
        gesture_buffer.clear()
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
