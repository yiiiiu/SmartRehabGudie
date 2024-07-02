import os
import sys
import cv2
import numpy as np
import time
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch
import pickle
import threading
import pygame
from PIL import Image, ImageDraw, ImageFont

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在的目录添加到系统目录中
if current_dir not in sys.path:
    sys.path.append(current_dir)

from utils import tool

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录
current_directory = os.path.dirname(current_file_path)
# 更改工作目录为当前文件所在的目录
os.chdir(current_directory)

# 骨架连接

skeleton_map = [
    {'srt_kpt_id': 15, 'dst_kpt_id': 13, 'sim': '右侧脚踝-右侧膝盖'},
    {'srt_kpt_id': 13, 'dst_kpt_id': 11, 'sim': '右侧膝盖-右侧胯'},
    {'srt_kpt_id': 16, 'dst_kpt_id': 14, 'sim': '左侧脚踝-左侧膝盖'},
    {'srt_kpt_id': 14, 'dst_kpt_id': 12, 'sim': '左侧膝盖-左侧胯'},
    {'srt_kpt_id': 11, 'dst_kpt_id': 12, 'sim': '右侧胯-左侧胯'},
    {'srt_kpt_id': 5, 'dst_kpt_id': 11, 'sim': '右边肩膀-右侧胯'},
    {'srt_kpt_id': 6, 'dst_kpt_id': 12, 'sim': '左边肩膀-左侧胯'},
    {'srt_kpt_id': 5, 'dst_kpt_id': 6, 'sim': '右边肩膀-左边肩膀'},
    {'srt_kpt_id': 5, 'dst_kpt_id': 7, 'sim': '右边肩膀-右侧胳膊肘'},
    {'srt_kpt_id': 6, 'dst_kpt_id': 8, 'sim': '左边肩膀-左侧胳膊肘'},
    {'srt_kpt_id': 7, 'dst_kpt_id': 9, 'sim': '右侧胳膊肘-右侧手腕'},
    {'srt_kpt_id': 8, 'dst_kpt_id': 10, 'sim': '左侧胳膊肘-左侧手腕'},
    # {'srt_kpt_id':0, 'dst_kpt_id':5, 'sim': '鼻尖-右肩'},
    # {'srt_kpt_id':0, 'dst_kpt_id':6, 'sim': '鼻尖-左肩'}
]


class PoseMonitor:
    def __init__(self, streaming=False):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        if streaming:
            self.model = YOLO('models/yolov8n-pose.pt')
        else:
            self.model = YOLO('models/yolov8x-pose-p6.pt')
        self.model.to(self.device)
        pygame.mixer.init()  # 确保在类初始化时就调用pygame.mixer.init()

    def save_standard_pose(self, sport_name, pose_num, standard_video, standard_srt_pose, standard_end_pose):
        '''
        sport_name: 运动名称
        pose_num: 动作编号
        standard_video: 标准视频
        standard_srt_pose: 标准准备动作
        standard_end_pose: 标准结束动作
        '''

        # 创建标准动作关键点字典
        keypoint_dict = {}

        cap = cv2.VideoCapture(standard_video)

        frame_count = 0
        keypoint_list = []
        while (cap.isOpened()):
            success, frame = cap.read()
            frame_count += 1
            if frame_count % 6:
                continue
            if not success:
                break
            frame = self.frame2keypoint(frame)
            keypoint_list.append(frame)

        cv2.destroyAllWindows()
        cap.release()
        keypoint_dict['standard_pose'] = keypoint_list
        if standard_srt_pose:
            keypoint_dict['standard_srt_pose'] = self.frame2keypoint(standard_srt_pose)
        else:
            keypoint_dict['standard_srt_pose'] = None
        if standard_end_pose:
            keypoint_dict['standard_end_pose'] = self.frame2keypoint(standard_end_pose)
        else:
            keypoint_dict['standard_end_pose'] = None

        folder_path = f'standard_pose/{sport_name}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(f'{folder_path}/{pose_num}.pkl', 'wb') as file:
            # 将字典保存到文件中
            pickle.dump(keypoint_dict, file)
        # self,
    def load_standard_pose(sport_name, pose_num):

        with open(f'standard_pose/{sport_name}/{pose_num}.pkl', 'rb') as file:
            loaded_keypoint_dict = pickle.load(file)

        standard_pose = loaded_keypoint_dict['standard_pose']
        standard_srt_pose = loaded_keypoint_dict['standard_srt_pose']
        standard_end_pose = loaded_keypoint_dict['standard_end_pose']
        return standard_pose, standard_srt_pose, standard_end_pose

    def frame2keypoint(self, img_bgr):
        results = self.model(img_bgr, verbose=False)  # verbose设置为False，不单独打印每一帧预测结果
        bboxes_keypoints = results[0].keypoints.data.cpu().numpy().astype('uint32')[0]
        return bboxes_keypoints

    def user_video_process(self, user_video, pose_name, pose_num):

        filehead = user_video.split('/')[-1]
        output_path = "out-" + filehead

        print('视频开始处理', user_video)

        cap = cv2.VideoCapture(user_video)
        frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)

        out = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))

        # 读取标准姿态关键点
        standard_pose, srt_pose, end_pose = self.load_standard_pose(pose_name, pose_num)
        # frame_count = 0
        count = 0
        while (cap.isOpened()):
            start_time = time.time()

            success, frame = cap.read()
            # frame_count += 1
            # if frame_count % 3:
            #     continue
            if not success:
                break
            # 提取用户姿态关键点
            keypoint = self.frame2keypoint(frame)
            if len(keypoint) == 0:
                print('未检测到用户')
                # continue
                return frame
            # 画出用户姿态
            frame, missing = self.process_draw_keypoint(keypoint, frame)

            if missing:
                print('摄像头未能捕捉用户全身')
                pass

            end_time1 = time.time()
            print(f'预测用户姿态时间: {end_time1 - start_time}')

            sim_list = []
            sim_dic_list = []
            # 遍历每一帧的标准姿态
            for pose in standard_pose:
                # 计算每一帧的标准姿态和用户姿态的相似度
                sim_dic, sim = tool.keypoint2similarity2(keypoint, pose)
                sim_list.append(sim)
                sim_dic_list.append(sim_dic)
            # 找到与用户姿态相似度最大的一帧标准姿态
            sim_ = max(sim_list)
            # 如果最大相似度小于0.96, 则纠正用户姿态
            if sim_ < 0.96 and missing == 0:
                sim_max_index = sim_list.index(sim_)  # 拿到最大相似度的索引
                max_sim_dic = sim_dic_list[sim_max_index]  # 通过索引拿到各个关节的相似度字典
                max_sim_pose = standard_pose[sim_max_index]  # 通过索引拿到相似度最大一帧的标准姿态关键点
                # 将标准姿态关键点与用户姿态关键点以左肩重合，并标准化大小
                max_sim_pose = tool.map_keypoints(keypoint, max_sim_pose)
                # 遍历各个关节的相似度字典，找到相似度小于0.96的关节名称
                result = [key for key, value in max_sim_dic.items() if value < 0.96]

                order = ''
                for sim_key in result:
                    # 用关节名称找到关节id
                    srt_kpt_id1, end_kpt_id1, srt_kpt_id2, end_kpt_id2 = tool.find_ids2(sim_key)
                    # 画出需要纠正的关节姿态
                    frame = self.process_draw_keypoint3(max_sim_pose, frame, srt_kpt_id1, end_kpt_id1, srt_kpt_id2,
                                                        end_kpt_id2)

                    order += f'{self.correct_pose(keypoint, max_sim_pose, end_kpt_id1, sim_key)}, '

                order = order.strip(', ')

                # frame = cv2.putText(frame, order, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                frame = self.chinese_pic(frame, order, (50, 200), (255, 0, 255))

                # 异步播放语音
                speech_thread = threading.Thread(target=self.speak_async, args=(order,))
                speech_thread.start()

                # frame = cv2.putText(frame, order, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                frame = self.chinese_pic(frame, order, (50, 200), (255, 0, 255))

                # engine.say(order)
                # engine.runAndWait()
                # 确保主线程在语音线程结束后再终止（如果需要）
                speech_thread.join()

            sim_ = f'{round(sim_ * 100, 1)}%'  # 相似度

            # 使用标准起始动作和结束动作为用户动作计数
            if srt_pose is not None and end_pose is not None:
                srt_sim_dic, srt_sim = tool.keypoint2similarity2(keypoint, srt_pose)
                if srt_sim > 0.98:
                    srt_count = 1
                end_sim_dic, end_sim = tool.keypoint2similarity2(keypoint, end_pose)
                if end_sim > 0.98 and srt_count == 1:
                    srt_count = 0
                    count += 1

                    # 将标准姿态关键点与用户姿态关键点以左肩重合，并标准化大小
                    end_pose_ = tool.map_keypoints(keypoint, end_pose)
                    # 遍历各个关节的相似度字典，找到相似度小于0.96的关节名称
                    result_ = [key for key, value in end_sim_dic.items() if value < 0.96]
                    if result_:
                        for sim_key in result_:
                            # 用关节名称找到关节id
                            srt_kpt_id1, end_kpt_id1, srt_kpt_id2, end_kpt_id2 = tool.find_ids2(sim_key)
                            # 画出需要纠正的关节姿态
                            frame_ = self.process_draw_keypoint3(end_pose_, frame, srt_kpt_id1, end_kpt_id1,
                                                                 srt_kpt_id2, end_kpt_id2)
                    else:
                        frame_ = frame

                    end_sim_ = f'{round(end_sim * 100, 1)}%'  # 相似度

                    frame_ = self.chinese_pic(frame_, f'相似度: {end_sim_}', (50, 50), (0, 255, 0))

                    # frame_ = cv2.putText(frame_, f'similarity: {end_sim_}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imwrite('frame_image.jpg', frame_)
                frame = self.chinese_pic(frame, f'完成数量: {count}', (50, 100), (0, 255, 0))
                # frame = cv2.putText(frame, f'count: {count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # 记录该帧处理完毕的时间
            end_time2 = time.time()
            # 计算每秒处理图像帧数FPS
            print(f'用户姿态纠错时间: {end_time2 - end_time1}')
            FPS = 1 / (end_time2 - start_time)
            # 在画面上写字：图片，字符串，左上角坐标，字体，字体大小，颜色，字体粗细
            FPS_string = '每秒帧数: ' + str(int(FPS))  # 写在画面上的字符串
            frame = self.chinese_pic(frame, FPS_string, (50, 150), (0, 255, 0))
            frame = self.chinese_pic(frame, f'相似度: {sim_}', (50, 50), (0, 255, 0))

            # frame = cv2.putText(frame, f'similarity: {sim_}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            out.write(frame)

        cv2.destroyAllWindows()
        out.release()
        cap.release()
        print('视频已保存', output_path)

    def play_audio(self, file_path):
        """
        使用pygame库播放音频文件
        :param file_path: 音频文件路径
        """
        pygame.mixer.music.load(file_path)  # 加载音频文件
        pygame.mixer.music.play()  # 播放音频
        while pygame.mixer.music.get_busy():  # 等待音频播放完毕
            pygame.time.Clock().tick(10)

    def speak_async(self, order):
        """
        异步语音播放函数
        """
        audio_file_path = "./videos/monitor.mp3"  # 根据text生成或指定录音文件路径
        audio_thread = threading.Thread(target=self.play_audio, args=(audio_file_path,))
        audio_thread.start()
        # engine = pyttsx3.init()
        # engine.say(order)

    # engine.runAndWait()

    def user_streaming_process(self, frame, standard_pose, srt_pose, end_pose, count, start_time):

        # 设置帧率为30帧
        # cap.set(cv2.CAP_PROP_FPS, 30)

        # 读取标准姿态关键点
        # standard_pose, srt_pose, end_pose = self.load_standard_pose(pose_name, pose_num)
        # frame_count = 0
        # count = 0
        # start_time = time.time()

        # frame_count += 1
        # if frame_count % 3:
        #     continue

        # 提取用户姿态关键点
        srt_count = 0
        keypoint = self.frame2keypoint(frame)
        if len(keypoint) == 0:
            print('未检测到用户')
            return frame
            # continue
        # 画出用户姿态
        frame, missing = self.process_draw_keypoint(keypoint, frame)

        if missing:
            print('摄像头未能捕捉用户全身')
            pass

        end_time1 = time.time()
        print(f'预测用户姿态时间: {end_time1 - start_time}')

        sim_list = []
        sim_dic_list = []
        # 遍历每一帧的标准姿态
        for pose in standard_pose:
            # 计算每一帧的标准姿态和用户姿态的相似度
            sim_dic, sim = tool.keypoint2similarity2(keypoint, pose)
            sim_list.append(sim)
            sim_dic_list.append(sim_dic)
        # 找到与用户姿态相似度最大的一帧标准姿态
        sim_ = max(sim_list)
        # 如果最大相似度小于0.96, 则纠正用户姿态
        if sim_ < 0.96 and missing == 0:
            sim_max_index = sim_list.index(sim_)  # 拿到最大相似度的索引
            max_sim_dic = sim_dic_list[sim_max_index]  # 通过索引拿到各个关节的相似度字典
            max_sim_pose = standard_pose[sim_max_index]  # 通过索引拿到相似度最大一帧的标准姿态关键点
            # 将标准姿态关键点与用户姿态关键点以左肩重合，并标准化大小
            max_sim_pose = tool.map_keypoints(keypoint, max_sim_pose)
            # 遍历各个关节的相似度字典，找到相似度小于0.96的关节名称
            result = [key for key, value in max_sim_dic.items() if value < 0.96]

            order = ''
            for sim_key in result:
                # 用关节名称找到关节id
                srt_kpt_id1, end_kpt_id1, srt_kpt_id2, end_kpt_id2 = tool.find_ids2(sim_key)
                # 画出需要纠正的关节姿态
                frame = self.process_draw_keypoint3(max_sim_pose, frame, srt_kpt_id1, end_kpt_id1, srt_kpt_id2,
                                                    end_kpt_id2)

                order += f'{self.correct_pose(keypoint, max_sim_pose, end_kpt_id1, sim_key)}, '
            order = order.strip(', ')

            speech_thread = threading.Thread(target=self.speak_async, args=(order,))
            speech_thread.start()

            # frame = cv2.putText(frame, order, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            frame = self.chinese_pic(frame, order, (50, 200), (255, 0, 255))

            speech_thread.join()

        sim_ = f'{round(sim_ * 100, 1)}%'  # 相似度

        # 使用标准起始动作和结束动作为用户动作计数
        if srt_pose is not None and end_pose is not None:
            srt_sim_dic, srt_sim = tool.keypoint2similarity2(keypoint, srt_pose)
            if srt_sim > 0.98:
                srt_count = 1
            end_sim_dic, end_sim = tool.keypoint2similarity2(keypoint, end_pose)
            if end_sim > 0.98 and srt_count == 1:
                srt_count = 0
                count += 1

                # 将标准姿态关键点与用户姿态关键点以左肩重合，并标准化大小
                end_pose_ = tool.map_keypoints(keypoint, end_pose)
                # 遍历各个关节的相似度字典，找到相似度小于0.96的关节名称
                result_ = [key for key, value in end_sim_dic.items() if value < 0.96]
                if result_:
                    for sim_key in result_:
                        # 用关节名称找到关节id
                        srt_kpt_id1, end_kpt_id1, srt_kpt_id2, end_kpt_id2 = tool.find_ids2(sim_key)
                        # 画出需要纠正的关节姿态
                        frame_ = self.process_draw_keypoint3(end_pose_, frame, srt_kpt_id1, end_kpt_id1,
                                                             srt_kpt_id2, end_kpt_id2)
                else:
                    frame_ = frame

                end_sim_ = f'{round(end_sim * 100, 1)}%'  # 相似度
                # frame_ = cv2.putText(frame_, f'similarity: {end_sim_}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                frame = self.chinese_pic(frame, f'相似度: {end_sim_}', (50, 50), (0, 255, 0))

                cv2.imwrite('frame_image.jpg', frame_)
            frame = self.chinese_pic(frame, f'完成数量: {count}', (50, 100), (0, 255, 0))

            # frame = cv2.putText(frame, f'count: {count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # 记录该帧处理完毕的时间
        end_time2 = time.time()
        # 计算每秒处理图像帧数FPS
        print(f'用户姿态纠错时间: {end_time2 - end_time1}')
        FPS = 1 / (end_time2 - start_time)
        # 在画面上写字：图片，字符串，左上角坐标，字体，字体大小，颜色，字体粗细
        FPS_string = '每秒帧数: ' + str(int(FPS))  # 写在画面上的字符串
        # frame = cv2.putText(frame, FPS_string, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        frame = self.chinese_pic(frame, FPS_string, (50, 150), (0, 255, 0))
        # frame = cv2.putText(frame, f'similarity: {sim_}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        frame = self.chinese_pic(frame, f'相似度: {sim_}', (50, 50), (0, 255, 0))
        return frame





    def process_draw_keypoint(self, point, img_bgr):

        missing = 0
        # 画该框的骨架连接
        for skeleton in skeleton_map:

            # 获取起始点坐标
            srt_kpt_id = skeleton['srt_kpt_id']
            srt_kpt_x = point[srt_kpt_id][0]
            srt_kpt_y = point[srt_kpt_id][1]
            if srt_kpt_x == 0 and srt_kpt_y == 0:
                missing += 1
                continue
            # 获取终止点坐标
            dst_kpt_id = skeleton['dst_kpt_id']
            dst_kpt_x = point[dst_kpt_id][0]
            dst_kpt_y = point[dst_kpt_id][1]
            if dst_kpt_x == 0 and dst_kpt_y == 0:
                missing += 1
                continue

            # 骨架连接颜色
            skeleton_color = [0, 255, 0]  # 绿色

            # 骨架连接线宽
            skeleton_thickness = 4

            # 画骨架连接
            img_bgr = cv2.line(img_bgr, (srt_kpt_x, srt_kpt_y), (dst_kpt_x, dst_kpt_y), color=skeleton_color,
                               thickness=skeleton_thickness)

        return img_bgr, missing

    def process_draw_keypoint2(self, point, img_bgr, srt_id, dst_id):

        # 获取起始点坐标
        srt_kpt_x = point[srt_id][0]
        srt_kpt_y = point[srt_id][1]
        if srt_kpt_x == 0 and srt_kpt_y == 0:
            return img_bgr
        # 获取终止点坐标
        dst_kpt_x = point[dst_id][0]
        dst_kpt_y = point[dst_id][1]
        if dst_kpt_x == 0 and dst_kpt_y == 0:
            return img_bgr

        # 骨架连接颜色
        skeleton_color = [0, 0, 255]  # 红色

        # 骨架连接线宽
        skeleton_thickness = 4

        # 画骨架连接
        img_bgr = cv2.line(img_bgr, (srt_kpt_x, srt_kpt_y), (dst_kpt_x, dst_kpt_y), color=skeleton_color,
                           thickness=skeleton_thickness)

        return img_bgr

    def process_draw_keypoint3(self, point, img_bgr, srt_id1, dst_id1, srt_id2, dst_id2):

        # 获取起始点坐标
        srt_kpt_x1 = point[srt_id1][0]
        srt_kpt_y1 = point[srt_id1][1]
        if srt_kpt_x1 == 0 and srt_kpt_y1 == 0:
            return img_bgr
        # 获取终止点坐标
        dst_kpt_x1 = point[dst_id1][0]
        dst_kpt_y1 = point[dst_id1][1]
        if dst_kpt_x1 == 0 and dst_kpt_y1 == 0:
            return img_bgr

        # 获取起始点坐标
        srt_kpt_x2 = point[srt_id2][0]
        srt_kpt_y2 = point[srt_id2][1]
        if srt_kpt_x2 == 0 and srt_kpt_y2 == 0:
            return img_bgr
        # 获取终止点坐标
        dst_kpt_x2 = point[dst_id2][0]
        dst_kpt_y2 = point[dst_id2][1]
        if dst_kpt_x2 == 0 and dst_kpt_y2 == 0:
            return img_bgr

        # 骨架连接颜色
        skeleton_color = [0, 0, 255]  # 红色

        # 骨架连接线宽
        skeleton_thickness = 4

        # 画骨架连接
        img_bgr = cv2.line(img_bgr, (srt_kpt_x1, srt_kpt_y1), (dst_kpt_x1, dst_kpt_y1), color=skeleton_color,
                           thickness=skeleton_thickness)
        img_bgr = cv2.line(img_bgr, (srt_kpt_x2, srt_kpt_y2), (dst_kpt_x2, dst_kpt_y2), color=skeleton_color,
                           thickness=skeleton_thickness)

        return img_bgr

    def correct_pose(self, user_keypoint, standard_keypoint, jointid, joint):

        difference = user_keypoint[jointid] - standard_keypoint[jointid]
        """
                    5: 'leaf shoulder',
                    6: 'right shoulder',
                    7: 'left elbow',
                    8: 'right elbow',
                    9: 'left hand',
                    10: 'right hand',
                    11: 'left hip',
                    12: 'right hip',
                    13: 'left knee',
                    14: 'right knee',
                    15: 'left foot',
                    16: 'right foot',
                    """
        joint_names = {

            # ... 添加其他关节的ID和中文名称
            5: '左肩膀',
            6: '右肩膀',
            7: '左肘',
            8: '右肘',
            9: '左手',
            10: '右手',
            11: '左臀',
            12: '右臀',
            13: '左膝',
            14: '右膝',
            15: '左脚',
            16: '右脚',
        }

        if jointid < 10:
            if jointid == 6:
                return ''
            if difference[1] > 0:
                return f'请抬高 {joint_names[jointid]}'
            else:
                return f'请降低 {joint_names[jointid]}'

        else:
            if difference[1] > 0:
                return f'请下蹲'
            else:
                return f'请站起'

    def run(self, camera):
        pass

    def chinese_pic(self, img, text: str, org, color):
        """
        :param img: 图像
        :param text: 文字
        :param org: 位置
        :param color: 颜色
        :return:
        """
        # 图像从OpenCV格式转换成PIL格式
        img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 字体包
        font = ImageFont.truetype(r'E:\Users\13194\Desktop\Training\project-2024.7.1\project\SimHei.ttf', 40)
        # 需要先把输出的中文字符转换成Unicode编码形式
        if not isinstance(text, str):
            text = str.decode('utf8')

        draw = ImageDraw.Draw(img_PIL)
        draw.text(org, text, font=font, fill=color)

        # 转换回OpenCV格式
        img_OpenCV = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
        return img_OpenCV

