import cv2
import os
import glob

# Source videos directory
video_files = [
    r"D:\Download_\PXL_20260928_051600702.TS.mp4",
    r"D:\Download_\PXL_20260928_051652979.TS.mp4",
    r"D:\Download_\PXL_20260928_051716286.TS.mp4",
    r"D:\Download_\PXL_20260928_051941433.TS.mp4",
    r"D:\Download_\conveyor 20 inch.mp4"
]

output_dir = r"D:\Developer\Projects\paint-Shop-MES\extracted_frames"
os.makedirs(output_dir, exist_ok=True)

total_extracted = 0

for v_idx, v_path in enumerate(video_files):
    if not os.path.exists(v_path):
        print(f"Skipping (not found): {v_path}")
        continue
    
    cap = cv2.VideoCapture(v_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nProcessing Video {v_idx+1}: {os.path.basename(v_path)} (Total frames: {total_frames}, FPS: {fps:.1f})")
    
    # Extract 1 frame every 1.0 to 1.5 seconds to avoid duplicates
    frame_interval = max(int(fps * 1.2), 1)
    current_frame = 0
    extracted_from_this = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame % frame_interval == 0:
            frame_filename = f"vid{v_idx+1}_frame_{current_frame:05d}.jpg"
            save_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(save_path, frame)
            extracted_from_this += 1
            total_extracted += 1
            
        current_frame += 1

    cap.release()
    print(f"-> Extracted {extracted_from_this} clean frames from this video.")

print(f"\nSUCCESS! Total {total_extracted} frames extracted into: {output_dir}")
