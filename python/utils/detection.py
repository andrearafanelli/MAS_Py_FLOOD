import cv2
import numpy as np
import pandas as pd
import random
from utils.mask_detection import MaskInspection, savePL
import os
import redis
from time import sleep
import base64

OUT_CH = "mask_CH"

if __name__ == "__main__":
    R = redis.Redis(decode_responses=True)
    try:
        R.info()
    except Exception as e:
        print(e)
        exit(-1)
    IMG_PATH = f"{os.getcwd()}/predictions/"
    
    while True:
        image_files = [f for f in os.listdir(IMG_PATH)]
        choice = random.choice(image_files)
        mask = cv2.imread(f"{IMG_PATH+choice}")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        arr_bytes = mask.tobytes()
        b64_str = base64.b64encode(arr_bytes).decode()
        
        R.publish(OUT_CH, b64_str)
        sleep(10)
