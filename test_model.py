"""
Test script for verifying YOLOv8s pothole model (best.pt) on test media.
"""

import os
import cv2
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

def test_inference():
    print(f"[1] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"    Classes: {model.names}")
    
    samples = [
        os.path.join(BASE_DIR, "sample_images", "sample_city_pothole.jpg"),
        os.path.join(BASE_DIR, "sample_images", "sample_highway_pothole.jpg"),
    ]
    
    out_dir = os.path.join(BASE_DIR, "runs", "test_eval")
    os.makedirs(out_dir, exist_ok=True)

    print("\n[2] Testing Image Inference:")
    for img_path in samples:
        if not os.path.exists(img_path):
            continue
        res = model.predict(source=img_path, conf=0.4, save=True, project=out_dir, name="images", exist_ok=True)[0]
        boxes = res.boxes
        print(f"  -> {os.path.basename(img_path)}: {len(boxes)} pothole(s) detected")
        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            xyxy = [round(x, 1) for x in box.xyxy[0].tolist()]
            print(f"     Box {i+1}: conf={conf:.3f}, bbox={xyxy}")

    video_path = os.path.join(BASE_DIR, "sample_images", "sample_dashcam.mp4")
    if os.path.exists(video_path):
        print(f"\n[3] Testing Video Inference: {os.path.basename(video_path)}")
        res_video = model.predict(
            source=video_path,
            conf=0.4,
            save=True,
            project=out_dir,
            name="dashcam_test",
            exist_ok=True,
            stream=True
        )
        total_frames = 0
        pothole_frames = 0
        total_potholes = 0
        for r in res_video:
            total_frames += 1
            n = len(r.boxes)
            if n > 0:
                pothole_frames += 1
                total_potholes += n
            if total_frames % 25 == 0:
                print(f"     Processed {total_frames} frames... ({pothole_frames} with potholes)")
        print(f"  -> Video complete: {total_frames} frames, {pothole_frames} frames with potholes, {total_potholes} total detections.")
        print(f"  -> Saved annotated video to: {os.path.join(out_dir, 'dashcam_test')}")

if __name__ == "__main__":
    test_inference()
