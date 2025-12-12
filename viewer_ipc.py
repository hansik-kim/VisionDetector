import sys
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QMainWindow,
)

# ==============================
# 1) IPC 로컬 이미지 경로 (C:\yolo_share)
# ==============================
ORIGINAL_IMAGE_PATH = Path(r"C:\yolo_share\original.jpg")  # Basler 원본
RESULT_IMAGE_PATH   = Path(r"C:\yolo_share\result.jpg")    # YOLO 결과


class PanelViewer(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("IPC Viewer - Original / Inference")
        self.resize(1600, 900)

        central = QWidget(self)
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        # 상단: 두 이미지 나란히
        image_layout = QHBoxLayout()
        main_layout.addLayout(image_layout, stretch=1)

        # 왼쪽: 원본
        self.label_original = QLabel("Original Image", self)
        self.label_original.setAlignment(Qt.AlignCenter)
        self.label_original.setStyleSheet(
            "background-color: #202020; color: #ffffff; font-size: 16px;"
        )
        image_layout.addWidget(self.label_original, stretch=1)

        # 오른쪽: 추론 결과
        self.label_result = QLabel("Inference Result", self)
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setStyleSheet(
            "background-color: #202020; color: #ffffff; font-size: 16px;"
        )
        image_layout.addWidget(self.label_result, stretch=1)

        # 하단: 상태 메시지
        self.status_label = QLabel("", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #cccccc; font-size: 12px;")
        main_layout.addWidget(self.status_label)

        # 타이머: 0.5초마다 이미지 갱신
        self.timer = QTimer(self)
        self.timer.setInterval(500)  # ms
        self.timer.timeout.connect(self.update_images)
        self.timer.start()

        self.update_images()

    def update_images(self):
        status_msgs = []

        # ----- 왼쪽: 원본 -----
        if ORIGINAL_IMAGE_PATH.is_file():
            pix = QPixmap(str(ORIGINAL_IMAGE_PATH))
            if not pix.isNull():
                pix = pix.scaled(
                    self.label_original.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self.label_original.setPixmap(pix)
            else:
                self.label_original.setText("원본 이미지 로드 실패")
                status_msgs.append("original: load failed")
        else:
            self.label_original.setText("원본 이미지 없음")
            status_msgs.append(f"original: not found ({ORIGINAL_IMAGE_PATH})")

        # ----- 오른쪽: 결과 -----
        if RESULT_IMAGE_PATH.is_file():
            pix = QPixmap(str(RESULT_IMAGE_PATH))
            if not pix.isNull():
                pix = pix.scaled(
                    self.label_result.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self.label_result.setPixmap(pix)
            else:
                self.label_result.setText("추론 결과 이미지 로드 실패")
                status_msgs.append("result: load failed")
        else:
            self.label_result.setText("추론 결과 이미지 없음")
            status_msgs.append(f"result: not found ({RESULT_IMAGE_PATH})")

        # 상태 메시지
        if status_msgs:
            self.status_label.setText(" | ".join(status_msgs))
        else:
            self.status_label.setText("이미지 업데이트 완료")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_images()


def main():
    app = QApplication(sys.argv)
    win = PanelViewer()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
