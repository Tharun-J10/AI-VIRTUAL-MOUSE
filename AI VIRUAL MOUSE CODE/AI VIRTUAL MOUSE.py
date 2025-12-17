import time
from collections import deque

import cv2
import numpy as np
import pyautogui

try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("mediapipe is required. Install with: pip install mediapipe")

# ------------------ Config ------------------
SMOOTHING_ALPHA = 0.25   # EMA smoothing for cursor (0..1)
PINCH_THRESH = 0.035     # Thumb-to-finger distance (normalized)
SCROLL_PAIR_THRESH = 0.04  # Index-Middle distance for scroll mode
DRAG_HOLD_SECONDS = 0.5
CLICK_TAP_MAX_SECONDS = 0.35
ACTIVE_MARGIN = 0.12     # Ignore margin at edges (normalized)
CAM_INDEX = 0            # Camera index
SHOW_HUD = True

# --------------------------------------------
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# --------- Utility Functions ---------
def norm_dist(a, b):
    return np.linalg.norm(np.array([a.x - b.x, a.y - b.y]))


class EMA:
    """Exponential Moving Average smoothing."""
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.value = None

    def update(self, new):
        if self.value is None:
            self.value = np.array(new, dtype=float)
        else:
            self.value = self.alpha * np.array(new) + (1 - self.alpha) * self.value
        return self.value


class GestureState:
    """Keeps track of gesture states."""
    def __init__(self):
        self.pinching_index = False
        self.pinching_middle = False
        self.pinch_start_time = None
        self.dragging = False
        self.scroll_mode = False
        self.scroll_buffer = deque(maxlen=5)

    def start_pinch(self):
        self.pinch_start_time = time.time()

    def pinch_duration(self):
        if self.pinch_start_time is None:
            return 0.0
        return time.time() - self.pinch_start_time


def map_to_screen(nx, ny):
    """Map normalized [0..1] coords to screen with margins and horizontal flip."""
    m = ACTIVE_MARGIN
    nx = np.clip((nx - m) / max(1e-6, (1 - 2 * m)), 0.0, 1.0)
    ny = np.clip((ny - m) / max(1e-6, (1 - 2 * m)), 0.0, 1.0)
    # Flip horizontally for natural mirroring
    nx = 1.0 - nx
    return int(nx * screen_w), int(ny * screen_h)


# --------- Main Program ---------
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Try a different CAM_INDEX.")

    # Improve capture quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    ema = EMA(SMOOTHING_ALPHA)
    state = GestureState()

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror view
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            h, w, _ = frame.shape

            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                lm = hand.landmark

                # ---- Key landmarks ----
                thumb_tip = lm[4]
                index_tip = lm[8]
                middle_tip = lm[12]

                # ---- Distances ----
                d_thumb_index = norm_dist(thumb_tip, index_tip)
                d_thumb_middle = norm_dist(thumb_tip, middle_tip)
                d_index_middle = norm_dist(index_tip, middle_tip)

                # ---- Cursor Position ----
                nx, ny = index_tip.x, index_tip.y
                sx, sy = map_to_screen(nx, ny)
                sx, sy = ema.update((sx, sy))
                pyautogui.moveTo(int(sx), int(sy), duration=0)

                # ---- Gesture States ----
                pinch_index_now = d_thumb_index < PINCH_THRESH
                pinch_middle_now = d_thumb_middle < PINCH_THRESH
                scroll_mode_now = (d_index_middle < SCROLL_PAIR_THRESH) and not pinch_index_now and not pinch_middle_now

                # Left click / Drag
                if pinch_index_now and not state.pinching_index:
                    state.pinching_index = True
                    state.start_pinch()
                elif not pinch_index_now and state.pinching_index:
                    if state.dragging:
                        pyautogui.mouseUp()
                        state.dragging = False
                    else:
                        if state.pinch_duration() <= CLICK_TAP_MAX_SECONDS:
                            pyautogui.click()
                    state.pinching_index = False
                    state.pinch_start_time = None

                # Drag hold
                if state.pinching_index and not state.dragging and state.pinch_duration() >= DRAG_HOLD_SECONDS:
                    pyautogui.mouseDown()
                    state.dragging = True

                # Right click
                if pinch_middle_now and not state.pinching_middle:
                    state.pinching_middle = True
                    state.start_pinch()
                elif not pinch_middle_now and state.pinching_middle:
                    if state.pinch_duration() <= CLICK_TAP_MAX_SECONDS:
                        pyautogui.click(button='right')
                    state.pinching_middle = False
                    state.pinch_start_time = None

                # Scroll
                if scroll_mode_now:
                    state.scroll_mode = True
                    state.scroll_buffer.append(ny)
                    if len(state.scroll_buffer) >= 2:
                        dy = state.scroll_buffer[-2] - state.scroll_buffer[-1]
                        steps = int(dy * 100)
                        if steps != 0:
                            pyautogui.scroll(steps)
                else:
                    state.scroll_mode = False
                    state.scroll_buffer.clear()

                # ---- HUD ----
                if SHOW_HUD:
                    mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                    hud = [
                        f"thumb-index: {d_thumb_index:.3f}",
                        f"thumb-middle: {d_thumb_middle:.3f}",
                        f"idx-mid(for scroll): {d_index_middle:.3f}",
                        f"dragging: {state.dragging}",
                        f"scroll: {state.scroll_mode}",
                    ]
                    y0 = 24
                    for i, t in enumerate(hud):
                        cv2.putText(frame, t, (10, y0 + i*22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 50), 2)

                    m = ACTIVE_MARGIN
                    x1, y1 = int(m*w), int(m*h)
                    x2, y2 = int((1-m)*w), int((1-m)*h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 50), 1)

            else:
                if SHOW_HUD:
                    cv2.putText(frame, "Show one hand to control cursor", (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)

            cv2.imshow('AI Virtual Mouse', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
