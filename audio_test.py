from aip import AipSpeech
from playsound import playsound
import os
import time


def baidu_voice(voice, audio):
    """ 你的 APPID AK SK """
    APP_ID = '90031657'
    API_KEY = 'WaYvsIe1BXwN0W9JENaB7SfT'
    SECRET_KEY = 'Jkr7RELTW67CWDOq3nRaZD6RhdcQcojM'
    directory = './audio'
    path = os.path.join(directory, audio)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
    result  = client.synthesis(voice, 'zh', 1, {
        'vol': 8,'per':0
    })
    # 识别正确返回语音二进制 错误则返回dict 参照下面错误码
    if not isinstance(result, dict):
        with open(path, 'wb') as f:
            f.write(result)
    else:
        return print(result)
    abs_path = os.path.abspath(path)
    # current_path = abs_path.replace('\\', '\\\\')
    # playsound(current_path)
    playsound(abs_path)
    #time.sleep(4)
    # os.system('auido.mp3')

    #time.sleep(t)

if __name__ == '__main__':
    baidu_voice('今天的康复训练结束，请慢慢深呼吸，放松身体。', 'Flow6.mp3')
