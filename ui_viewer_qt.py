import sys
import os
import time
from pathlib import Path
import cv2
import numpy as np

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QVBoxLayout, QHBoxLayout, QPushButton, QFrame,
                             QStatusBar)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap, QFont

# ==========================================
# 설정 영역
# ==========================================

# 패널 PC에서 접근할 공유 폴더 경로
SHARE_DIR = Path(r"C:\yolo_share")

ORIGINAL_PATH = SHARE_DIR / "original.jpg"
RESULT_PATH = SHARE_DIR / "result.jpg"

# 갱신 주기 (ms)
REFRESH_RATE = 100

class VisionViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Vision Inspector (Panel PC) - PyQt6")
        self.resize(1600, 900)
        
        # 상태 변수
        self.is_running = True  # Start/Stop 제어용
        self.last_mtime_orig = 0
        self.last_mtime_result = 0
        
        # 메인 위젯 및 레이아웃
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # 1. 제목 영역
        title_layout = QHBoxLayout()
        self.lbl_title_orig = QLabel("Original Image")
        self.lbl_title_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_title_orig.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        
        self.lbl_title_result = QLabel("Result Image (AI)")
        self.lbl_title_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_title_result.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        
        title_layout.addWidget(self.lbl_title_orig)
        title_layout.addWidget(self.lbl_title_result)
        main_layout.addLayout(title_layout)
        
        # 2. 이미지 표시 영역
        image_layout = QHBoxLayout()
        
        # 왼쪽 (원본)
        self.view_orig = QLabel()
        self.view_orig.setStyleSheet("background-color: black; border: 2px solid #555;")
        self.view_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view_orig.setMinimumSize(320, 240)
        
        # 오른쪽 (결과)
        self.view_result = QLabel()
        self.view_result.setStyleSheet("background-color: black; border: 2px solid #555;")
        self.view_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view_result.setMinimumSize(320, 240)
        
        image_layout.addWidget(self.view_orig, 1) # stretch 1
        image_layout.addWidget(self.view_result, 1) # stretch 1
        main_layout.addLayout(image_layout, 1) # 메인 레이아웃에서도 늘어나도록
        
        # 3. 하단 버튼 영역
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 20, 0, 10)
        
        btn_style = """
            QPushButton {
                font-size: 16px;
                padding: 10px;
                border-radius: 5px;
                background-color: #eee;
                border: 1px solid #ccc;
            }
            QPushButton:hover {
                background-color: #ddd;
            }
        """
        
        self.btn_start = QPushButton("Start")
        self.btn_start.setStyleSheet(btn_style)
        self.btn_start.clicked.connect(self.on_start)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet(btn_style)
        self.btn_stop.clicked.connect(self.on_stop)
        
        self.btn_exit = QPushButton("Exit")
        self.btn_exit.setStyleSheet("QPushButton { font-size: 16px; padding: 10px; border-radius: 5px; background-color: #ffcccc; border: 1px solid #cc9999; } QPushButton:hover { background-color: #ffaaaa; }")
        self.btn_exit.clicked.connect(self.close)
        
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.btn_exit)
        
        main_layout.addLayout(btn_layout)
        
        # 4. 상태바
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # 타이머 설정 (이미지 갱신용)
        self.timer = QTimer()
        self.timer.setInterval(REFRESH_RATE)
        self.timer.timeout.connect(self.update_view)
        self.timer.start()

    def on_start(self):
        self.is_running = True
        self.status_bar.showMessage("Monitor Started")
        
    def on_stop(self):
        self.is_running = False
        self.status_bar.showMessage("Monitor Stopped (Paused)")

    def read_image_qt(self, path):
        """OpenCV로 읽어서 QPixmap으로 변환"""
        if not os.path.exists(path):
            return None
            
        try:
            # 한글 경로 지원을 위해 numpy로 읽기
            stream = open(str(path), "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
            stream.close()
            
            if img is None:
                return None
            
            # BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # QImage -> QPixmap
            return QPixmap.fromImage(qimg)
            
        except Exception as e:
            # print(f"Error: {e}")
            return None

    def update_label_image(self, label, pixmap):
        """라벨 크기에 맞춰 이미지 리사이즈 후 표시"""
        if pixmap is None:
            return
            
        # 라벨 크기에 맞게 스케일링 (비율 유지)
        scaled_pixmap = pixmap.scaled(
            label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)

    def update_view(self):
        if not self.is_running:
            return

        # 디버깅용: 경로 확인 (100번 호출마다 한 번씩만 출력하여 로그 폭주 방지)
        if not hasattr(self, 'debug_cnt'): self.debug_cnt = 0
        self.debug_cnt += 1
        if self.debug_cnt % 50 == 0:
             print(f"[DEBUG] Checking path: {ORIGINAL_PATH} / Exists: {os.path.exists(ORIGINAL_PATH)}")

        # 1. 원본 이미지
        try:
            if os.path.exists(ORIGINAL_PATH):
                mtime = os.path.getmtime(ORIGINAL_PATH)
                if mtime > self.last_mtime_orig:
                    # print(f"[INFO] New original image detected. Reading...")
                    pixmap = self.read_image_qt(ORIGINAL_PATH)
                    if pixmap:
                        self.update_label_image(self.view_orig, pixmap)
                        self.last_mtime_orig = mtime
                    else:
                        print(f"[WARN] Failed to load original image.")
        except Exception as e:
            print(f"[ERROR] Original Update Error: {e}")


        # 2. 결과 이미지
        try:
            if os.path.exists(RESULT_PATH):
                mtime = os.path.getmtime(RESULT_PATH)
                if mtime > self.last_mtime_result:
                    pixmap = self.read_image_qt(RESULT_PATH)
                    if pixmap:
                        self.update_label_image(self.view_result, pixmap)
                        self.last_mtime_result = mtime
                        
                        # 상태바 업데이트
                        cur_time = time.strftime("%Y-%m-%d %H:%M:%S")
                        self.status_bar.showMessage(f"Last Update: {cur_time}")
        except Exception:
            pass
            
    def resizeEvent(self, event):
        # 창 크기가 바뀔 때 현재 이미지가 있다면 다시 그려서 크기 맞춤
        # (이미지가 고정되어 보이는 현상 방지)
        # 단, 실제로는 update_view가 계속 돌기 때문에 자동으로 맞춰지긴 함.
        super().resizeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisionViewer()
    window.show()
    sys.exit(app.exec())

