from aip import AipSpeech
from playsound import playsound
import os
import time


def baidu_voice(voice):
    """ 你的 APPID AK SK """
    APP_ID = '90031657'
    API_KEY = 'WaYvsIe1BXwN0W9JENaB7SfT'
    SECRET_KEY = 'Jkr7RELTW67CWDOq3nRaZD6RhdcQcojM'

    client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
    result  = client.synthesis(voice, 'zh', 1, {
        'vol': 8,'per':0
    })
    # 识别正确返回语音二进制 错误则返回dict 参照下面错误码
    if not isinstance(result, dict):
        with open('auido.mp3', 'wb') as f:
            f.write(result)
    else:
        return print(result)
    playsound(r'E:\Users\13194\Desktop\Training\git_repository\SmartRehabGudie-2\auido.mp3')
    #time.sleep(4)
    # os.system('auido.mp3')

    #time.sleep(t)

if __name__ == '__main__':
    baidu_voice('请开始八段锦动作练习')
