import cv2
import os
import glob

video_dir = r"D:\Download_\Video Processing"
output_file = r"D:\Developer\Projects\paint-Shop-MES\master_conveyor_20inch_merged.mp4"

# Collect unique video files
seen = set()
unique_videos = []
for f in sorted(glob.glob(os.path.join(video_dir, "*.mp4"))):
    fname = os.path.basename(f)
    if fname not in seen:
        seen.add(fname)
        unique_videos.append(f)

print(f"Total unique videos to merge: {len(unique_videos)}")

TARGET_W = 1280
TARGET_H = 720
TARGET_FPS = 30.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, TARGET_FPS, (TARGET_W, TARGET_H))

total_written = 0

for idx, v_path in enumerate(unique_videos):
    cap = cv2.VideoCapture(v_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[{idx+1}/{len(unique_videos)}] Merging: {os.path.basename(v_path)} ({total_frames} frames)...")
    
    written_from_this = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize/Letterbox to TARGET_W x TARGET_H preserving aspect ratio
        h, w = frame.shape[:2]
        scale = min(TARGET_W / w, TARGET_H / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Center on black canvas
        canvas = cv2.copyMakeBorder(
            resized,
            top=(TARGET_H - new_h) // 2,
            bottom=TARGET_H - new_h - (TARGET_H - new_h) // 2,
            left=(TARGET_W - new_w) // 2,
            right=TARGET_W - new_w - (TARGET_W - new_w) // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        
        out.write(canvas)
        written_from_this += 1
        total_written += 1

    cap.release()
    print(f"   -> Added {written_from_this} frames.")

out.release()
print(f"\nSUCCESS! Master merged video created at: {output_file}")
print(f"Total frames merged: {total_written}")

