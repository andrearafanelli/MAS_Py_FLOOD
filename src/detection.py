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
    PL_PATH = f"{os.getcwd()}/prolog/facts"
    flooded = pd.read_pickle('flooded_test.pkl')
    while True:
        choice = random.choice(flooded)
        idx = choice.replace('FloodNet','dataNew')
        idy = idx.replace('img','label').replace('.jpg','_lab.png')
        mask = cv2.imread(idy)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, (384,384))
        arr_bytes = mask.tobytes()
        b64_str = base64.b64encode(arr_bytes).decode()
        
        R.publish(OUT_CH, b64_str)
        sleep(10)
