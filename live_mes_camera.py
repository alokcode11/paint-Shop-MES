import cv2
import time
import numpy as np
import sys
import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Load Trained Model
MODEL_PATH = os.path.join(BASE_DIR, "ai-model", "weights", "best.pt")
print(f"Loading YOLO Model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print(f"Model classes: {model.names}")

# 2. Video Source Configuration
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")

CONVEYOR_19MIN = os.path.join(VIDEOS_DIR, "master_conveyor_20inch_merged.mp4")
PHONE_5MIN = os.path.join(VIDEOS_DIR, "video_20261011_151649.mp4")
MASTER_SHOWCASE = os.path.join(VIDEOS_DIR, "master_complete_6parts_showcase.mp4")

# Set RTSP transport to TCP (prevents packet drop and disconnection)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# PRAMA Channel 102 (Sub-stream - super smooth & fast) or 101 (Main-stream)
PRAMA_RTSP = "rtsp://admin:Alok2611@192.168.1.64:554/Streaming/Channels/102"
PRAMA_RTSP_MAIN = "rtsp://admin:Alok2611@192.168.1.64:554/Streaming/Channels/101"

if len(sys.argv) > 1 and sys.argv[1].isdigit():
    CAMERA_SOURCE = int(sys.argv[1])
    print(f"Loading Local Camera Index: {CAMERA_SOURCE}")
elif len(sys.argv) > 1 and (sys.argv[1].startswith("http") or sys.argv[1].startswith("rtsp")):
    CAMERA_SOURCE = sys.argv[1]
    print(f"Loading Live Stream: {CAMERA_SOURCE}")
elif len(sys.argv) > 1 and "prama_main" in sys.argv[1].lower():
    CAMERA_SOURCE = PRAMA_RTSP_MAIN
    print(f"Loading PRAMA Main Stream (4MP): {CAMERA_SOURCE}")
elif len(sys.argv) > 1 and "prama" in sys.argv[1].lower():
    CAMERA_SOURCE = PRAMA_RTSP
    print(f"Loading PRAMA Sub Stream (Smooth Live): {CAMERA_SOURCE}")
elif len(sys.argv) > 1 and "phone" in sys.argv[1].lower():
    CAMERA_SOURCE = PHONE_5MIN
    print(f"Loading Phone Video (5 mins): {PHONE_5MIN}")
elif len(sys.argv) > 1 and "showcase" in sys.argv[1].lower():
    CAMERA_SOURCE = MASTER_SHOWCASE
    print(f"Loading Showcase Video: {MASTER_SHOWCASE}")
elif os.path.exists(CONVEYOR_19MIN):
    CAMERA_SOURCE = CONVEYOR_19MIN
    print(f"Loading 19-Minute Original Conveyor Video: {CONVEYOR_19MIN}")
else:
    CAMERA_SOURCE = 0
    print("No offline conveyor video found. Defaulting to local webcam (Camera 0)...")

cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_FFMPEG)
if not cap.isOpened() and isinstance(CAMERA_SOURCE, str) and "102" in CAMERA_SOURCE:
    print("Channel 102 failed, trying Channel 101 (Main Stream)...")
    CAMERA_SOURCE = PRAMA_RTSP_MAIN
    cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_FFMPEG)

if not cap.isOpened() and isinstance(CAMERA_SOURCE, int):
    print("Camera 0 failed, trying Camera 1...")
    cap = cv2.VideoCapture(1)

if not cap.isOpened() and not (isinstance(CAMERA_SOURCE, str) and (CAMERA_SOURCE.startswith("http") or CAMERA_SOURCE.startswith("rtsp"))):
    print(f"Opening video file: {MASTER_SHOWCASE}")
    cap = cv2.VideoCapture(MASTER_SHOWCASE)

if not cap.isOpened():
    print("ERROR: Could not open any video or camera source!")
    exit()

# 6 Target Components
TARGET_COMPONENTS = [
    "front_panel",
    "top_cover",
    "base_panel",
    "rh_stand",
    "lh_stand",
    "valve_plate"
]

# Display Canvas Setup (1280x720 Standard HD)
CANVAS_W = 1280
CANVAS_H = 720
PANEL_W = 340
VIDEO_W = CANVAS_W - PANEL_W  # 940 width for video

WINDOW_NAME = "Paint Shop MES - Live AI Inspection"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, CANVAS_W, CANVAS_H)

# State Tracking
detected_in_current_hanger = set()
prev_time = time.time()
paused = False

is_live = isinstance(CAMERA_SOURCE, str) and (CAMERA_SOURCE.startswith("http") or CAMERA_SOURCE.startswith("rtsp"))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if (isinstance(CAMERA_SOURCE, str) and not is_live) else 0
fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_sec = int(total_frames / fps_video) if (fps_video > 0 and not is_live) else 0
seek_seconds = 5

def on_trackbar(val):
    if not is_live and total_frames > 0:
        target_frame = min(int(val * fps_video), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

if not is_live and total_sec > 0:
    cv2.createTrackbar("Timeline (Sec)", WINDOW_NAME, 0, total_sec, on_trackbar)

print("\n--- PAINT SHOP MES REAL-TIME INSPECTION STARTED ---")
if is_live:
    print("Mode: LIVE PHONE CAMERA STREAM")
    print("Controls: Space: Pause | R: Reset Checklist | Q: Exit\n")
else:
    print("Controls: Drag Slider | A/D or Arrow keys: Seek +-5s | Space: Pause | R: Reset | Q: Exit\n")

while True:
    if not paused:
        ret, raw_frame = cap.read()
        if not ret:
            # Loop video
            if isinstance(CAMERA_SOURCE, str) and not is_live:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                detected_in_current_hanger.clear()
                continue
            else:
                # Live stream initial buffer wait
                time.sleep(0.05)
                continue
    else:
        # Keep same frame when paused
        pass

    # FPS Calculation
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    current_sec = int(current_frame / fps_video) if fps_video > 0 else 0
    time_str = f"{current_sec//60:02d}:{current_sec%60:02d} / {total_sec//60:02d}:{total_sec%60:02d}"

    # Run YOLO Detection on raw frame
    results = model.predict(source=raw_frame, conf=0.35, verbose=False)
    detections = results[0].boxes

    for box in detections:
        cls_id = int(box.cls[0].item())
        cls_name = model.names.get(cls_id, f"class_{cls_id}")
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].cpu().numpy().astype(int)

        detected_in_current_hanger.add(cls_name)

        # Draw box on raw frame
        color = (0, 255, 0)
        cv2.rectangle(raw_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 3)
        label = f"{cls_name} {conf*100:.0f}%"
        cv2.putText(raw_frame, label, (xyxy[0], max(xyxy[1] - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # CREATE MASTER DISPLAY CANVAS (1280x720)
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    # Fit video inside left VIDEO_W x CANVAS_H with aspect ratio preserved (Letterboxing)
    fh, fw = raw_frame.shape[:2]
    scale = min(VIDEO_W / fw, CANVAS_H / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    scaled_v = cv2.resize(raw_frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    # Center video in video section
    xo = (VIDEO_W - nw) // 2
    yo = (CANVAS_H - nh) // 2
    canvas[yo:yo+nh, xo:xo+nw] = scaled_v

    # DRAW DEDICATED MES DASHBOARD ON THE RIGHT (PANEL_W)
    px = VIDEO_W
    cv2.rectangle(canvas, (px, 0), (CANVAS_W, CANVAS_H), (24, 24, 28), -1)
    cv2.line(canvas, (px, 0), (px, CANVAS_H), (60, 60, 70), 2)

    # Header
    cv2.putText(canvas, "PAINT SHOP MES", (px + 20, 42),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(canvas, "Plant: NGM PG Supa | ODU 20\"", (px + 20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.line(canvas, (px + 15, 85), (CANVAS_W - 15, 85), (80, 80, 90), 1)

    # Checklist Header
    cv2.putText(canvas, "COMPONENT VERIFICATION", (px + 20, 115),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (220, 220, 220), 1)

    y_pos = 155
    completed_count = 0
    for comp in TARGET_COMPONENTS:
        is_detected = comp in detected_in_current_hanger
        display_name = comp.replace('_', ' ').upper()
        if is_detected:
            completed_count += 1
            icon = "[OK]"
            color = (0, 255, 0)
        else:
            icon = "[..]"
            color = (120, 120, 125)

        cv2.putText(canvas, f"{icon}  {display_name}", (px + 20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2 if is_detected else 1)
        y_pos += 42

    cv2.line(canvas, (px + 15, y_pos + 10), (CANVAS_W - 15, y_pos + 10), (80, 80, 90), 1)
    y_pos += 50

    # Big Status Banner
    if completed_count == len(TARGET_COMPONENTS):
        status_text = "SET COMPLETE 6/6"
        banner_bg = (0, 140, 0)
        text_color = (255, 255, 255)
    else:
        status_text = f"INSPECTING: {completed_count}/6"
        banner_bg = (0, 80, 200)
        text_color = (255, 255, 255)

    cv2.rectangle(canvas, (px + 15, y_pos - 25), (CANVAS_W - 15, y_pos + 15), banner_bg, -1)
    cv2.putText(canvas, status_text, (px + 25, y_pos),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, text_color, 2)

    # Bottom Status Bar
    cv2.putText(canvas, f"FPS: {fps:.1f}", (px + 20, CANVAS_H - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
    cv2.putText(canvas, f"TIME: {time_str}", (px + 20, CANVAS_H - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1)
    
    state_str = "PAUSED" if paused else "PLAYING"
    cv2.putText(canvas, f"STATE: {state_str}", (px + 20, CANVAS_H - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120) if not paused else (0, 160, 255), 1)
    cv2.putText(canvas, "Drag Slider | Arr/AD:Seek | Spc:Pause", (px + 20, CANVAS_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1)

    # Sync trackbar with video time
    if isinstance(CAMERA_SOURCE, str) and not is_live and not paused and total_sec > 0:
        cv2.setTrackbarPos("Timeline (Sec)", WINDOW_NAME, current_sec)

    cv2.imshow(WINDOW_NAME, canvas)

    key = cv2.waitKeyEx(1)
    if key in [ord('q'), ord('Q'), 27]: # Q or ESC to exit
        break
    elif key in [ord('r'), ord('R')]: # R to reset
        detected_in_current_hanger.clear()
        print("Checklist reset!")
    elif key == ord(' '): # Space to pause
        paused = not paused
        print("PAUSED" if paused else "RESUMED")
    elif key in [2555904, 65363, ord('d'), ord('D')]: # Seek forward 5s
        new_f = min(current_frame + int(seek_seconds * fps_video), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_f)
        print(f"Jumped forward to {new_f//int(fps_video)}s")
    elif key in [2424832, 65361, ord('a'), ord('A')]: # Seek back 5s
        new_f = max(current_frame - int(seek_seconds * fps_video), 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_f)
        print(f"Jumped backward to {new_f//int(fps_video)}s")

cap.release()
cv2.destroyAllWindows()
print("Inspection finished cleanly.")
