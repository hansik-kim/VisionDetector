# 파일 예: ipc_yolo_onion.py
from ultralytics import YOLO
import cv2
from pathlib import Path
import os

MODEL_PATH = r"C:\Project\VisionDetector\Dataset\onion.v2i.yolov8\runs\detect\train\weights\best.pt"
SHARE_DIR = Path(r"C:\yolo_share")
ORIGINAL_IMAGE_PATH = SHARE_DIR / "original.jpg"
RESULT_IMAGE_PATH   = SHARE_DIR / "result.jpg"

def main():
    os.makedirs(SHARE_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=1280)
        r = results[0]
        annotated = r.plot()

        cv2.imwrite(str(ORIGINAL_IMAGE_PATH), frame)
        cv2.imwrite(str(RESULT_IMAGE_PATH), annotated)

        # 디버그용:
        # cv2.imshow("IPC Debug", annotated)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
