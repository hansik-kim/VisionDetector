import os
from pathlib import Path

import cv2
from pypylon import pylon
from ultralytics import YOLO

# ===========================
# 설정 영역
# ===========================

# 1) YOLO 모델 경로 (본인 학습 모델로 수정)
MODEL_PATH = r"C:\Project\VisionDetector\Dataset\onion.v2i.yolov8\runs\detect\train\weights\best.pt"

# 2) Basler 카메라 IP (지금 알려준 값)
CAMERA_IP = "168.0.36.36"

# 3) IPC에서 이미지를 저장할 공유 폴더
#    이 폴더를 나중에 UIPC에서 네트워크 드라이브로 매핑(Z: 등)
SHARE_DIR = Path(r"C:\yolo_share")
ORIGINAL_IMAGE_PATH = SHARE_DIR / "original.jpg"
RESULT_IMAGE_PATH   = SHARE_DIR / "result.jpg"

# 4) 내부 처리용 YOLO 이미지 사이즈
YOLO_IMGSZ = 1280  # 작은 객체 많으면 1280 추천


def select_camera_by_ip(ip: str):
    """주어진 IP와 일치하는 Basler 디바이스를 찾는다. 없으면 None."""
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if not devices:
        print("[ERROR] Basler 디바이스를 찾을 수 없습니다.")
        return None

    print("[INFO] 연결된 Basler 디바이스 목록:")
    selected = None
    for d in devices:
        try:
            dev_ip = d.GetIpAddress()
        except Exception:
            dev_ip = "<Unknown>"
        print(f"  - {d.GetFriendlyName()} | IP: {dev_ip}")
        if dev_ip == ip:
            selected = d

    if selected is None:
        print(f"[WARN] IP가 {ip} 인 카메라를 찾지 못했습니다. 첫 번째 디바이스를 사용합니다.")
        selected = devices[0]

    return selected


def main():
    # 공유 폴더 준비
    os.makedirs(SHARE_DIR, exist_ok=True)
    print(f"[INFO] 공유 폴더: {SHARE_DIR}")

    # YOLO 모델 로드
    print(f"[INFO] YOLO 모델 로드: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Basler 카메라 선택
    tl_factory = pylon.TlFactory.GetInstance()
    dev_info = select_camera_by_ip(CAMERA_IP)
    if dev_info is None:
        return

    # Basler 카메라 오픈
    camera = pylon.InstantCamera(tl_factory.CreateDevice(dev_info))
    print(f"[INFO] 사용 카메라: {camera.GetDeviceInfo().GetModelName()} | "
          f"{camera.GetDeviceInfo().GetIpAddress()}")

    camera.Open()

    # 이미지 포맷 변환기 (Basler → BGR OpenCV)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    # 그랩 시작
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    print("[INFO] 카메라 그랩 시작. 'q' 키를 누르면 종료합니다. (디버그 창 기준)")

    try:
        while camera.IsGrabbing():
            grab_result = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)

            if grab_result.GrabSucceeded():
                # Basler 이미지 → numpy 배열(BGR)
                image = converter.Convert(grab_result)
                frame = image.GetArray()  # 이게 OpenCV가 처리할 수 있는 BGR 이미지

                # ============== YOLO 추론 ==============
                results = model(frame, imgsz=YOLO_IMGSZ)
                r = results[0]
                annotated = r.plot()  # bounding box 그려진 결과 이미지

                # ============== 공유 폴더에 저장 ==============
                cv2.imwrite(str(ORIGINAL_IMAGE_PATH), frame)
                cv2.imwrite(str(RESULT_IMAGE_PATH), annotated)

                # 디버그용: IPC에서도 한 번 보여보고 싶을 때
                cv2.imshow("Basler + YOLO (IPC Debug)", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] 'q' 입력으로 종료합니다.")
                    break
            else:
                print("[WARN] 프레임 그랩 실패")

            grab_result.Release()
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt 로 종료합니다.")
    finally:
        camera.StopGrabbing()
        camera.Close()
        cv2.destroyAllWindows()
        print("[INFO] 카메라 및 리소스 정리 완료.")


if __name__ == "__main__":
    main()
