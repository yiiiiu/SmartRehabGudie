import os
import sys


from pose import pose_monitor

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



if __name__ == '__main__':

    # save_standard()

    monitor = pose_monitor.PoseMonitor(streaming=True)
    #monitor.user_video_process('videos/user/user2.mp4', 'baduanjin', '1')
    monitor.user_streaming_process(0, 'baduanjin', '1')
    # print(monitor.load_standard_pose('baduanjin', '3.1')[0])
    pass