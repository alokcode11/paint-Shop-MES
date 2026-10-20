import cv2
import os
import zipfile

video_path = r"D:\paint-Shop-MES\master_conveyor_20inch_merged.mp4"
output_dir = r"D:\paint-Shop-MES\extracted_frames_3000"
zip_output = r"D:\paint-Shop-MES\roboflow_upload_3000_frames.zip"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

TARGET_COUNT = 3000
step = max(int(total_frames / TARGET_COUNT), 1)

print(f"Master Video Loaded: {total_frames} total frames ({total_frames/fps:.1f}s)")
print(f"Sampling Step: Har {step} frames par 1 frame capture hoga (Target: {TARGET_COUNT} frames)...\n")

current_frame = 0
extracted_count = 0

while cap.isOpened() and extracted_count < TARGET_COUNT:
    ret, frame = cap.read()
    if not ret:
        break
    
    if current_frame % step == 0:
        filename = f"part20_frame_{extracted_count+1:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
        # High quality JPEG save (quality=92)
        cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        extracted_count += 1
        
        if extracted_count % 300 == 0:
            print(f"Progress: {extracted_count}/{TARGET_COUNT} frames successfully extracted ({extracted_count/TARGET_COUNT*100:.0f}%)...")

    current_frame += 1

cap.release()
print(f"\nExtraction Finished! Total {extracted_count} high-quality frames saved to: {output_dir}")

# Create zip file for Roboflow
print(f"Creating 1-click Zip file: {zip_output} ...")
with zipfile.ZipFile(zip_output, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(output_dir):
        for f in files:
            zipf.write(os.path.join(root, f), f)

print(f"\nSUCCESS! 3000 Frames Zip Ready: {zip_output}")

