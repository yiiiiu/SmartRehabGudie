import numpy as np


skeleton_map_sim = [
    {'srt_kpt_id':15, 'dst_kpt_id':13, 'sim': '左侧脚踝-左侧膝盖'},
    {'srt_kpt_id':13, 'dst_kpt_id':11, 'sim': '左侧膝盖-左侧胯'},
    {'srt_kpt_id':16, 'dst_kpt_id':14, 'sim': '右侧脚踝-右侧膝盖'},
    {'srt_kpt_id':14, 'dst_kpt_id':12, 'sim': '右侧膝盖-右侧胯'},
    {'srt_kpt_id':11, 'dst_kpt_id':12, 'sim': '右侧胯-左侧胯'},
    {'srt_kpt_id':5, 'dst_kpt_id':11, 'sim': '左边肩膀-左侧胯'},
    {'srt_kpt_id':6, 'dst_kpt_id':12, 'sim': '右边肩膀-右侧胯'},
    {'srt_kpt_id':5, 'dst_kpt_id':6, 'sim': '右边肩膀-左边肩膀'},
    {'srt_kpt_id':5, 'dst_kpt_id':7, 'sim': '左边肩膀-左侧胳膊肘'},
    {'srt_kpt_id':6, 'dst_kpt_id':8, 'sim': '右边肩膀-右侧胳膊肘'},        
    {'srt_kpt_id':7, 'dst_kpt_id':9, 'sim': '左侧胳膊肘-左侧手腕'},
    {'srt_kpt_id':8, 'dst_kpt_id':10, 'sim': '右侧胳膊肘-右侧手腕'},   
    # {'srt_kpt_id':0, 'dst_kpt_id':5, 'sim': '鼻尖-右肩'},
    # {'srt_kpt_id':0, 'dst_kpt_id':6, 'sim': '鼻尖-左肩'}
    ]

skeleton_map_sim2 = [
    {'srt_kpt_id1':15, 'dst_kpt_id1':13, 'srt_kpt_id2':13, 'dst_kpt_id2':11, 'label': 'right knee'},  # 右侧膝盖
    {'srt_kpt_id1':16, 'dst_kpt_id1':14, 'srt_kpt_id2':14, 'dst_kpt_id2':12, 'label': 'left knee'},  # 左侧膝盖
    {'srt_kpt_id1':12, 'dst_kpt_id1':11, 'srt_kpt_id2':11, 'dst_kpt_id2':13, 'label': 'right side span'},  # 右侧胯
    {'srt_kpt_id1':11, 'dst_kpt_id1':12, 'srt_kpt_id2':12, 'dst_kpt_id2':14, 'label': 'left side span'},    # 左侧胯
    {'srt_kpt_id1':8, 'dst_kpt_id1':6, 'srt_kpt_id2':6, 'dst_kpt_id2':5, 'label': 'right shoulder'},    # 右侧肩膀
    {'srt_kpt_id1':7, 'dst_kpt_id1':5,  'srt_kpt_id2':5, 'dst_kpt_id2':6, 'label': 'left shoulder'},    # 左侧肩膀
    {'srt_kpt_id1':6, 'dst_kpt_id1':8, 'srt_kpt_id2':8, 'dst_kpt_id2':10, 'label': 'right elbow'},    # 右侧胳膊肘
    {'srt_kpt_id1':5, 'dst_kpt_id1':7, 'srt_kpt_id2':7, 'dst_kpt_id2':9, 'label': 'left elbow'}    # 左侧胳膊肘
]


# 将标准图像6号关键点与用户6号关键点重合
def map_keypoints(user_keypoint, standard_keypoint):
    user_keypoint = user_keypoint.astype('int32')
    standard_keypoint = standard_keypoint.astype('int32')

    
    standard_keypoint = standard_keypoint - standard_keypoint[6]
    standard_keypoint = map_keypoints2(user_keypoint, standard_keypoint) + user_keypoint[6]


    return standard_keypoint.astype('int32')


# 骨骼大小标准化
def map_keypoints2(user_keypoint, standard_keypoint):
    user_keypoint = user_keypoint
    standard_keypoint = standard_keypoint

    distance_user = distance(user_keypoint[5], user_keypoint[6]) + distance(user_keypoint[11], user_keypoint[12])
    distance_standard = distance(standard_keypoint[5], standard_keypoint[6]) + distance(standard_keypoint[11], standard_keypoint[12])
    k = distance_user / distance_standard
    
    standard_keypoint = (standard_keypoint * k)

    return standard_keypoint



# 计算余弦相似度
def cosine_similarity(vector1_start, vector1_end, vector2_start, vector2_end):
    vector1 = vector1_end.astype('int32') - vector1_start.astype('int32')
    vector2 = vector2_end.astype('int32') - vector2_start.astype('int32')
    # 计算两个向量的点积
    dot_product = np.dot(vector1, vector2)
    # 计算向量1的模长
    norm_vector1 = np.linalg.norm(vector1)
    # 计算向量2的模长
    norm_vector2 = np.linalg.norm(vector2)
    # 计算余弦相似度
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity


def distance(s1, s2):
    return np.sqrt(np.sum((s1 - s2) ** 2))


def keypoint2similarity(user_keypoint, standard_keypoint):
        
    sim_dict = {}

    for skeleton in skeleton_map_sim:
        srt_kpt_id = skeleton['srt_kpt_id']
        dst_kpt_id = skeleton['dst_kpt_id']
        sim_id = skeleton['sim']
        
        # 获取标准起始点坐标
        srt_kpt_1 = standard_keypoint[srt_kpt_id]

        # 获取标准终止点坐标
        dst_kpt_1 = standard_keypoint[dst_kpt_id]

        # 获取用户起始点坐标
        srt_kpt_2 = user_keypoint[srt_kpt_id]
        if srt_kpt_2[0] == 0 and srt_kpt_2[1] == 0:
            similarity = 0
            sim_dict[sim_id] = similarity
            continue

        # 获取用户终止点坐标
        dst_kpt_2 = user_keypoint[dst_kpt_id]
        if dst_kpt_2[0] == 0 and dst_kpt_2[1] == 0:
            similarity = 0
            sim_dict[sim_id] = similarity
            continue

        similarity = cosine_similarity(srt_kpt_1, dst_kpt_1, srt_kpt_2, dst_kpt_2)
        sim_dict[sim_id] = similarity
    
    ave_sim = sum(sim_dict.values()) / 12

    return sim_dict, ave_sim



def keypoint2similarity2(user_keypoint, standard_keypoint):
        
    sim_dict = {}

    for skeleton in skeleton_map_sim2:
        srt_kpt_id1 = skeleton['srt_kpt_id1']
        dst_kpt_id1 = skeleton['dst_kpt_id1']
        srt_kpt_id2 = skeleton['srt_kpt_id2']
        dst_kpt_id2 = skeleton['dst_kpt_id2']
        sim_id = skeleton['label']
        
        # 获取标准起始点坐标
        srt_kpt_11 = standard_keypoint[srt_kpt_id1]
        # 获取标准终止点坐标
        dst_kpt_11 = standard_keypoint[dst_kpt_id1]
        # 获取标准起始点坐标
        srt_kpt_12 = standard_keypoint[srt_kpt_id2]
        # 获取标准终止点坐标
        dst_kpt_12 = standard_keypoint[dst_kpt_id2]

        # 获取用户起始点坐标
        srt_kpt_21 = user_keypoint[srt_kpt_id1]
        if srt_kpt_21[0] == 0 and srt_kpt_21[1] == 0:
            similarity = 0
            sim_dict[sim_id] = similarity
            continue

        # 获取用户终止点坐标
        dst_kpt_21 = user_keypoint[dst_kpt_id1]
        if dst_kpt_21[0] == 0 and dst_kpt_21[1] == 0:
            similarity = 0
            sim_dict[sim_id] = similarity
            continue

        # 获取用户起始点坐标
        srt_kpt_22 = user_keypoint[srt_kpt_id2]
        if srt_kpt_22[0] == 0 and srt_kpt_22[1] == 0:
            similarity = 0
            sim_dict[sim_id] = similarity
            continue

        # 获取用户终止点坐标
        dst_kpt_22 = user_keypoint[dst_kpt_id2]
        if dst_kpt_22[0] == 0 and dst_kpt_22[1] == 0:
            similarity = 0
            sim_dict[sim_id] = similarity
            continue



        similarity1 = cosine_similarity(srt_kpt_11, dst_kpt_11, srt_kpt_21, dst_kpt_21)
        similarity2 = cosine_similarity(srt_kpt_12, dst_kpt_12, srt_kpt_22, dst_kpt_22)
        similarity = (similarity1 + similarity2) / 2

        sim_dict[sim_id] = similarity
    
    ave_sim = sum(sim_dict.values()) / 8

    return sim_dict, ave_sim


def find_ids(sim):
    for item in skeleton_map_sim:
        if item['sim'] == sim:
            return item['srt_kpt_id'], item['dst_kpt_id']
    return None, None


def find_ids2(sim):
    for item in skeleton_map_sim2:
        if item['label'] == sim:
            return item['srt_kpt_id1'], item['dst_kpt_id1'], item['srt_kpt_id2'], item['dst_kpt_id2']
    return None, None, None, None

