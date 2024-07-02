
    def user_video_process(self, user_video, pose_name, pose_num):
        filehead = user_video.split('/')[-1]
        output_path = "out-" + filehead
        
        print('视频开始处理',user_video)
        
        cap = cv2.VideoCapture(user_video)
        frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)

        out = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))

        # 读取标准姿态关键点
        standard_pose, srt_pose, end_pose = self.load_standard_pose(pose_name, pose_num)
        # frame_count = 0
        count = 0
        while(cap.isOpened()):
            success, frame = cap.read()
            # frame_count += 1
            # if frame_count % 3:
            #     continue
            if not success:
                break
            # 提取用户姿态关键点
            keypoint = self.frame2keypoint(frame)
            # 画出用户姿态
            frame, missing = self.process_draw_keypoint(keypoint, frame)

            sim_list = []
            sim_dic_list = []
            # 遍历每一帧的标准姿态
            for pose in standard_pose:
                # 计算每一帧的标准姿态和用户姿态的相似度
                sim_dic, sim = tool.keypoint2similarity(keypoint, pose)
                sim_list.append(sim)
                sim_dic_list.append(sim_dic)
            # 找到与用户姿态相似度最大的一帧标准姿态
            sim_ = max(sim_list)
            # 如果最大相似度小于0.99, 则纠正用户姿态
            if sim_ < 0.99:
                sim_max_index = sim_list.index(sim_)  # 拿到最大相似度的索引
                max_sim_dic = sim_dic_list[sim_max_index]  # 通过索引拿到各个关节的相似度字典
                max_sim_pose = standard_pose[sim_max_index]  # 通过索引拿到相似度最大一帧的标准姿态关键点
                # 将标准姿态关键点与用户姿态关键点以左肩重合，并标准化大小
                max_sim_pose = tool.map_keypoints(keypoint, max_sim_pose)
                # 遍历各个关节的相似度字典，找到相似度小于0.99的关节名称
                result = [key for key, value in max_sim_dic.items() if value < 0.99]

                for sim_key in result:
                    # 用关节名称找到关节id
                    srt_kpt_id, end_kpt_id = tool.find_ids(sim_key)
                    # 画出需要纠正的关节姿态
                    frame = self.process_draw_keypoint2(max_sim_pose, frame, srt_kpt_id, end_kpt_id)

            sim_ = f'{round(sim_ * 100, 1)}%'  # 相似度
            
            # 使用标准起始动作和结束动作为用户动作计数
            srt_sim_dic, srt_sim = tool.keypoint2similarity(keypoint, srt_pose)
            if srt_sim > 0.99:
                srt_count = 1
            end_sim_dic, end_sim = tool.keypoint2similarity(keypoint, end_pose)
            if end_sim > 0.99 and srt_count == 1:
                srt_count = 0
                count += 1

            frame = cv2.putText(frame, f'similarity: {sim_}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            frame = cv2.putText(frame, f'count: {count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            out.write(frame)
        

        cv2.destroyAllWindows()
        out.release()
        cap.release()
        print('视频已保存', output_path)

