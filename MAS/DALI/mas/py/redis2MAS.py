import LindaProxy.lindaproxy as lp
import redis
import base64
import numpy as np
from mask_detection import MaskInspection, savePL


def mask_2_pl(mask):
    inspection = MaskInspection(mask,tr=5)
    information = inspection.coordinates_calculation()
    regions, adjacent = inspection.get_dictionaries(information)

    pl_files = savePL(regions, adjacent, path ='..')
    pl_files.adjacent_pl()
    pl_files.region_pl()


IN_CH = "mask_CH"
IN_CH2 = "weather_CH"
OUT_CH = "toMAS"
AGENT1 = "perceptor"
AGENT2 = "alert"

if __name__ == "__main__":
    L = lp.LindaProxy(host='127.0.0.1')
    L.connect()
    R = redis.Redis(decode_responses=True)
    pubsub = R.pubsub()

    # Subscribe to multiple channels
    pubsub.subscribe([IN_CH, IN_CH2])

    # Start a loop to listen for messages on the subscribed channels
    while True:
        message = pubsub.get_message()

        # Start a loop to listen for messages on the subscribed channels
        for item in pubsub.listen():
            if item['type'] == 'message':
                # Check the channel of the received message
                if item['channel'] == IN_CH:
                    msg = item['data']
                    arr_bytes = base64.b64decode(msg)
                    mask = np.frombuffer(arr_bytes, dtype='uint8')
                    mask = mask.reshape((384,384))
                    mask_2_pl(mask)

                    print('.')
                    print('evento redis al MAS:', mask.shape)
                    L.send_message(AGENT1, "redis(mask)")

                elif item['channel'] == IN_CH2:
                    msg = item['data']
                    print('evento redis al MAS:', msg)
                    L.send_message(AGENT2, f"redis({msg})")
