import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter, \
    QPushButton, QFileDialog
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import Qt, QUrl, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
from pose import pose_monitor
from pose.pose_monitor import PoseMonitor
import time


def save_standard():
    monitor = pose_monitor.PoseMonitor(streaming=False)
    monitor.save_standard_pose('baduanjin', '1',
                               'videos/baduanjin/1/1.mp4',
                               'videos/baduanjin/1/1_start.jpg',
                               'videos/baduanjin/1/1_end.jpg')
    monitor.save_standard_pose('baduanjin', '2.1',
                               'videos/baduanjin/2/2.1.mp4',
                               'videos/baduanjin/2/2_start.jpg',
                               'videos/baduanjin/2/2.1_end.jpg')
    monitor.save_standard_pose('baduanjin', '2.2',
                               'videos/baduanjin/2/2.2.mp4',
                               'videos/baduanjin/2/2_start.jpg',
                               'videos/baduanjin/2/2.2_end.jpg')
    monitor.save_standard_pose('baduanjin', '3.1',
                               'videos/baduanjin/3/3.1.mp4',
                               'videos/baduanjin/3/3_start.jpg',
                               'videos/baduanjin/3/3.1_end.jpg')
    monitor.save_standard_pose('baduanjin', '3.2',
                               'videos/baduanjin/3/3.2.mp4',
                               'videos/baduanjin/3/3_start.jpg',
                               'videos/baduanjin/3/3.2_end.jpg')
    monitor.save_standard_pose('baduanjin', '4',
                               'videos/baduanjin/4/4.mp4',
                               None,
                               None)
    monitor.save_standard_pose('baduanjin', '5.1',
                               'videos/baduanjin/5/5.1.mp4',
                               'videos/baduanjin/5/5_start.jpg',
                               'videos/baduanjin/5/5_end.jpg')
    monitor.save_standard_pose('baduanjin', '5.2',
                               'videos/baduanjin/5/5.2.mp4',
                               'videos/baduanjin/5/5_start.jpg',
                               'videos/baduanjin/5/5_end.jpg')
    monitor.save_standard_pose('baduanjin', '6',
                               'videos/baduanjin/6/6.mp4',
                               'videos/baduanjin/6/6_start.jpg',
                               'videos/baduanjin/6/6_end.jpg')
    monitor.save_standard_pose('baduanjin', '7.1',
                               'videos/baduanjin/7/7.1.mp4',
                               'videos/baduanjin/7/7_start.jpg',
                               'videos/baduanjin/7/7.1_end.jpg')
    monitor.save_standard_pose('baduanjin', '7.2',
                               'videos/baduanjin/7/7.2.mp4',
                               'videos/baduanjin/7/7_start.jpg',
                               'videos/baduanjin/7/7.2_end.jpg')
    monitor.save_standard_pose('baduanjin', '8',
                               'videos/baduanjin/8/8.mp4',
                               None,
                               None)

    print('处理完成')





class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.monitor = pose_monitor.PoseMonitor(streaming=True)
        self.standard_pose, self.srt_pose, self.end_pose = PoseMonitor.load_standard_pose("baduanjin", "1")
        self.count = 0
        self.FPS = 0

    def run(self):
        cap = cv2.VideoCapture(0)
        # # 调整视图范围
        # width = 640
        # height = 960
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        while True:
            self.start_time = time.time()
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                end_time2 = time.time()
                frame = self.monitor.user_streaming_process(frame, self.standard_pose, self.srt_pose, self.end_pose, self.count, self.start_time)

                self.change_pixmap_signal.emit(frame)



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('八段锦运动评估系统')
        self.setGeometry(100, 100, 1000, 600)

        # 创建主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主窗口的布局管理器，使用水平布局 QHBoxLayout
        main_layout = QHBoxLayout(central_widget)

        # 创建左侧区域的部件
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_label = QLabel('') # 左侧视频
        left_label.setAlignment(Qt.AlignCenter)  # 居中对齐
        left_layout.addWidget(left_label)

        # 创建视频播放器部件
        self.video_widget = QVideoWidget()
        left_layout.addWidget(self.video_widget)

        # 创建控制按钮部件
        control_layout = QHBoxLayout()
        self.play_button = QPushButton('播放')
        self.pause_button = QPushButton('暂停')
        self.stop_button = QPushButton('停止')
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.stop_button)
        left_layout.addLayout(control_layout)

        # 创建右侧区域的部件
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_label = QLabel('') # 动作采集
        right_label.setAlignment(Qt.AlignCenter)  # 居中对齐
        right_layout.addWidget(right_label)

        # 创建显示视频的 Label
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumHeight(800)  # 设置最小高度
        self.camera_label.setMaximumHeight(1600)  # 设置最大高度
        right_layout.addWidget(self.camera_label)

        # 创建视频采集线程
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_camera_image)
        self.video_thread.start()

        # 创建 QSplitter 控件，并设置左右部件
        splitter = QSplitter()
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        # 设置分割线样式和大小
        splitter.setStyleSheet("QSplitter::handle { background-color: #ccc; }")
        splitter.setSizes([self.width() // 2, self.width() // 2])
        main_layout.addWidget(splitter)

        # 初始化视频播放器
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)

        # 连接按钮信号和槽函数
        self.play_button.clicked.connect(self.play_video)
        self.pause_button.clicked.connect(self.pause_video)
        self.stop_button.clicked.connect(self.stop_video)

    def play_video(self):
        # 打开视频文件
        file_name, _ = QFileDialog.getOpenFileName(self, "打开视频", "", "视频文件 (*.mp4 *.avi *.mov)")
        if file_name != '':
            video_url = QUrl.fromLocalFile(file_name)
            media_content = QMediaContent(video_url)
            self.media_player.setMedia(media_content)
            self.media_player.play()

    def pause_video(self):
        self.media_player.pause()

    def stop_video(self):
        self.media_player.stop()

    def update_camera_image(self, frame):

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.camera_label.width(), self.camera_label.height(), Qt.KeepAspectRatio)
        self.camera_label.setPixmap(QPixmap.fromImage(p))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
