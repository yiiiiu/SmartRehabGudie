import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter, \
    QPushButton, QFileDialog, QTextEdit
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import Qt, QUrl, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
from pose import pose_monitor
from pose.pose_monitor import PoseMonitor
import time
import io


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


class TextStream(io.StringIO):
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit

    def write(self, s):
        cursor = self.text_edit.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(s)
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    play_standard_video_signal = pyqtSignal(str)  # 用于播放标准动作视频的信号
    video_playback_finished_signal = pyqtSignal()  # 视频播放完成信号

    def __init__(self, num_array):
        super().__init__()
        self.monitor = pose_monitor.PoseMonitor(streaming=True)
        self.num = num_array
        self.len = len(num_array)
        self.a_count = 0
        self.video_stop = False
        # self.standard_pose, self.srt_pose, self.end_pose = PoseMonitor.load_standard_pose("baduanjin", "1")
        self.FPS = 0
        
        self.running = False  # 添加一个标志位控制线程的运行

    def run(self):
        self.video_stop = False
        standard_pose, srt_pose, end_pose = PoseMonitor.load_standard_pose("baduanjin", self.num[0])
        cap = cv2.VideoCapture(0)
        # # 调整视图范围
        # width = 640
        # height = 960
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # 检查摄像头是否打开
        if not cap.isOpened():
            print("Error: Could not open video capture.")
            return
        
        count = 0
        srt_count = 0
        w_count = 0
        self.a_count += 1
        olo_video_stop = self.video_stop
        while self.running:
            start_time = time.time()
            ret, frame = cap.read()
            if ret:
                w_count += 1  # 有效桢数计数
                frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                # print(f'当前是否可以开始动作练习:{self.video_stop}')
                if self.video_stop:
                    frame, count, srt_count = self.monitor.user_streaming_process(frame, standard_pose, srt_pose, end_pose, count, srt_count, start_time)
                else:
                    frame, count, srt_count = self.monitor.user_streaming_process(frame, standard_pose, srt_pose, end_pose, start_time=start_time, count=0, srt_count=0)
                self.change_pixmap_signal.emit(frame)

                # 计算FPS
                end_time = time.time()
                self.FPS = 1 / (end_time - start_time)

        
            if w_count == 10 and self.a_count == 1:
                self.monitor.speak_async('1')

            if w_count == 200:
                self.monitor.speak_async('2')
                # 根据动作编号获取标准动作视频路径
                standard_video_path = fr'E:\Users\13194\Desktop\Training\git_repository\SmartRehabGudie-2\pose\videos\baduanjin\{self.num[0][0]}\{self.num[0]}.mp4'
                # # 发射信号播放标准动作视频
                self.play_standard_video_signal.emit(standard_video_path)
  
            if self.video_stop and not olo_video_stop:
                self.monitor.speak_async('3')
            olo_video_stop = self.video_stop  

            if count == 4:
                self.monitor.speak_async('5-1')
            elif count == 9:
                self.monitor.speak_async('5-2')
                time.sleep(4)
                self.num = self.num[1:]
                if not self.num:
                    self.monitor.speak_async('6')
                    cap.release()
                    return
                self.monitor.speak_async('4')
                cap.release()
                self.cycle()  # 切换下一个动作
        
        # if count >= 3:
        #     if self.a_count == self.len:
        #         self.monitor.speak_async('6')
        #     else:
        #         self.monitor.speak_async('4')
        #         cap.release()
        #         self.cycle()  # 切换下一个动作

        cap.release()
    
    def cycle(self):
        self.run()

    def start_video(self):
        if not self.running:
            self.running = True
            self.start()

    def stop_video(self):
        self.running = False
        self.wait()  # 等待线程完全退出




class MainWindow(QMainWindow):
    def __init__(self, num_array):
        super().__init__()      
        self.num_array = num_array
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
        # self.camera_label.setMinimumHeight(400)  # 设置最小高度
        # self.camera_label.setMaximumHeight(800)  # 设置最大高度
        right_layout.addWidget(self.camera_label)

        # 创建日志输出的 QTextEdit
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        # self.log_text_edit.setMinimumHeight(0)
        self.log_text_edit.setMaximumHeight(200)
        right_layout.addWidget(self.log_text_edit)

        # 创建视频采集线程
        self.video_thread = VideoThread(self.num_array)
        self.video_thread.change_pixmap_signal.connect(self.update_camera_image)
        self.video_thread.play_standard_video_signal.connect(self.play_standard_video)  # 连接播放标准动作视频的信号
        # self.video_thread.start()

        # 定义 delay_timer 为类的成员变量
        self.delay_timer = QTimer()
        self.delay_timer.setSingleShot(True)  # 设置为单次触发

        # 创建新的控制按钮部件，用于视频流
        video_control_layout = QHBoxLayout()
        self.start_stream_button = QPushButton('打开视频流')
        self.stop_stream_button = QPushButton('关闭视频流')
        video_control_layout.addWidget(self.start_stream_button)
        video_control_layout.addWidget(self.stop_stream_button)
        right_layout.addLayout(video_control_layout)       

        # 创建 QSplitter 控件，并设置左右部件
        splitter = QSplitter()
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        # 设置分割线样式和大小
        splitter.setStyleSheet("QSplitter::handle { background-color: #ccc; }")
        splitter.setSizes([self.width() // 2, self.width() // 2])
        # splitter.setStretchFactor(0, 1)  # 左侧区域伸展因子
        # splitter.setStretchFactor(1, 3)  # 右侧区域伸展因子
        main_layout.addWidget(splitter)

        # 初始化视频播放器
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)

        # 连接按钮信号和槽函数
        self.play_button.clicked.connect(self.play_video)
        self.pause_button.clicked.connect(self.pause_video)
        self.stop_button.clicked.connect(self.stop_video)
        self.start_stream_button.clicked.connect(self.start_video_stream)
        self.stop_stream_button.clicked.connect(self.stop_video_stream)

        # 连接 mediaStatusChanged 信号到槽函数
        self.media_player.mediaStatusChanged.connect(self.on_media_status_changed)

        # 重定向标准输出到 QTextEdit
        sys.stdout = TextStream(self.log_text_edit)

    def play_video(self):
        # 打开视频文件
        file_name, _ = QFileDialog.getOpenFileName(self, "打开视频", "", "视频文件 (*.mp4 *.avi *.mov)")
        if file_name:
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

    def start_video_stream(self):
        self.video_thread.start_video()

    def stop_video_stream(self):
        self.video_thread.stop_video()
        self.clear_camera_image()

    def clear_camera_image(self):
        self.camera_label.clear()

    def play_standard_video(self, video_path):
        # 设置延时时间
        delay_seconds = 4

        # 使用类的成员变量 delay_timer
        self.delay_timer.timeout.connect(lambda: self.start_video_after_delay(video_path))
        self.delay_timer.start(delay_seconds * 1000)  # 将秒转换为毫秒
    
    def start_video_after_delay(self, video_path):
        # 在延时结束后播放视频
        video_url = QUrl.fromLocalFile(video_path)
        media_content = QMediaContent(video_url)
        self.media_player.setMedia(media_content)
        self.media_player.play()

    def on_media_status_changed(self, status):
        # 检测视频是否播放完成
        if status == QMediaPlayer.EndOfMedia:
            self.media_player.stop()
            self.video_thread.video_stop = True
    
    # 覆盖默认的窗口关闭事件，在关闭窗口时确保线程被停止。
    def closeEvent(self, event):
        self.stop_video_stream()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    num_array = ['1', '2.1', '2.2', '3.1', '3.2', '4', '5.1', '5.2', '6', '7.1', '7.2', '8']
    window = MainWindow(num_array)
    window.show()
    sys.exit(app.exec_())
