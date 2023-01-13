import cv2
import numpy as np
import pandas as pd
from utils.simulator import get_color
import redis
from time import sleep
import base64

OUT_CH = "weather_CH"

if __name__ == "__main__":
    R = redis.Redis(decode_responses=True)
    try:
        R.info()
    except Exception as e:
        print(e)
        exit(-1)
    
    while True:
        sleep(20)
        color_alert = get_color()   
        R.publish(OUT_CH, color_alert)
        sleep(20)
