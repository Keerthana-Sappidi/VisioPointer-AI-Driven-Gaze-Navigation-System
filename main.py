import threading
import time
import cv2
import mediapipe as mp
import pyautogui
import vosk
import json
import pyaudio
import os
from fuzzywuzzy import fuzz


print("🚀 Starting VisioPointer with Vosk voice control...")

# === Webcam Initialization ===
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("❌ Error: Could not access webcam. Close other applications.")
    exit()

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
print("✅ Webcam initialized. Press 'Q' to exit.")

# === Globals ===
last_click_time = 0
last_scroll_time = 0
dwell_start_time = None

# === Vosk Initialization ===
vosk_path = os.path.join("models", "vosk-model-small-en-us-0.15")
if not os.path.exists(vosk_path):
    print(f"❌ Vosk model not found in '{vosk_path}'. Please check the folder name.")
    exit()

vosk_model = vosk.Model(vosk_path)
recognizer = vosk.KaldiRecognizer(vosk_model, 16000)

audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
stream.start_stream()

# === Blink Click Function ===
def click_mouse():
    global last_click_time
    current_time = time.time()
    if current_time - last_click_time > 1:
        print("🖱️ Blink detected! Clicking mouse...")
        pyautogui.click()
        last_click_time = current_time

# === Eye Aspect Ratio Function ===
def eye_aspect_ratio(eye):
    return abs(eye[1].y - eye[0].y)

# === Scroll Function ===
def scroll_page(direction):
    global last_scroll_time
    current_time = time.time()
    if current_time - last_scroll_time > 1:
        if direction == "up":
            print("⬆️ Scrolling Up...")
            pyautogui.scroll(100)
        elif direction == "down":
            print("⬇️ Scrolling Down...")
            pyautogui.scroll(-100)
        last_scroll_time = current_time

# === Voice Command Listener ===
def voice_control():
    while True:
        data = stream.read(4096, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            command = result.get("text", "").strip().lower()

            if not command:
                continue

            print(f"🎤 Voice Command: '{command}'")

            command_list = ["click", "scroll up", "scroll down", "exit"]
            threshold = 80  # Match strength required

            matched = None
            for cmd in command_list:
                if fuzz.ratio(cmd, command) > threshold:
                    matched = cmd
                    break

            if matched:
                print(f"✅ Matched Command: '{matched}'")

                if matched == "click":
                    pyautogui.click()
                    print("🖱️ Voice Click Activated.")
                elif matched == "scroll up":
                    pyautogui.scroll(300)
                    print("⬆️ Voice Scroll Up.")
                elif matched == "scroll down":
                    pyautogui.scroll(-300)
                    print("⬇️ Voice Scroll Down.")
                elif matched == "exit":
                    print("🚪 Voice Exit Triggered.")
                    cam.release()
                    cv2.destroyAllWindows()
                    break
            else:
                print(f"🤷 Unrecognized voice command: '{command}'")


# Start Voice Thread
threading.Thread(target=voice_control, daemon=True).start()

# === Main Loop ===
while True:
    ret, frame = cam.read()
    if not ret:
        print("❌ Frame capture error.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # === Gaze Tracking ===
        eye_landmark = landmarks[474]
        x = int(eye_landmark.x * frame_w)
        y = int(eye_landmark.y * frame_h)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        screen_x = screen_w * eye_landmark.x
        screen_y = screen_h * eye_landmark.y
        pyautogui.moveTo(screen_x, screen_y, duration=0.1)

        # === Blink Detection ===
        left_eye = [landmarks[145], landmarks[159]]
        for landmark in left_eye:
            lx = int(landmark.x * frame_w)
            ly = int(landmark.y * frame_h)
            cv2.circle(frame, (lx, ly), 3, (0, 255, 255), -1)

        ear = eye_aspect_ratio(left_eye)
        if ear < 0.015:
            threading.Thread(target=click_mouse, daemon=True).start()

            # === Scroll Detection ===
            forehead_y = landmarks[10].y
            eye_center_y = (landmarks[159].y + landmarks[145].y) / 2
            threshold = 0.015
            if eye_center_y < forehead_y - threshold:
                threading.Thread(target=scroll_page, args=("up",), daemon=True).start()
            elif eye_center_y > forehead_y + threshold:
                threading.Thread(target=scroll_page, args=("down",), daemon=True).start()

            # === Dwell Click (look at one place for 2s) ===
            if dwell_start_time is None:
                dwell_start_time = time.time()
            elif time.time() - dwell_start_time > 2:
                pyautogui.click()
                print("🕒 Dwell click triggered!")
                dwell_start_time = None
        else:
            dwell_start_time = None

    cv2.imshow('VisioPointer - Eye & Voice Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exit triggered by keyboard.")
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()
print("✅ VisioPointer closed.")
