# import packages
import cv2
import numpy as np
from urllib.request import urlopen
import os
import datetime
import time
import sys
from time import sleep
from threading import Thread, Lock

# URL's
url_cam1 = 'http://10.0.0.250:81/stream'
url_cam2 = 'http://10.0.0.251:81/stream'
urls = [url_cam1]

# image frames
frame_camn = [None]*len(urls)

# buffer size and stream
CAMERA_BUFFRER_SIZE=4096

# mutex
mutex = Lock()

def thread_grab_cam(url):
    stream = urlopen(url)
    bts = b''
    global frame_camn
    while True:
        # get stream
        bts += stream.read(CAMERA_BUFFRER_SIZE)
        jpghead = bts.find(b'\xff\xd8')
        jpgend = bts.find(b'\xff\xd9')
        # identify end of stream
        if jpghead>-1 and jpgend>-1:
            # extract image frame
            jpg = bts[jpghead:jpgend+2]
            bts = bts[jpgend+2:]
            frame_ = cv2.imdecode(np.frombuffer(jpg,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
            idx_ = url_cam1.index(url)
            mutex.acquire()
            frame_camn[idx_] = frame_
            mutex.release()
            sleep(0.1)

if __name__ == '__main__':
    threads = list()
    for i in range(len(urls)):
        print('Starting capture thread for camera {}'.format(i))
        x = Thread(target=thread_grab_cam, args=(urls[i],))
        threads.append(x)
        x.start()

    # loop forever
    while True:
        try:
            for i in range(len(urls)):
                if frame_camn[i] is not None:
                    mutex.acquire()
                    cv2.imshow("Camera {} Stream".format(i), frame_camn[i])
                    frame_camn[i] = None
                    mutex.release()

            # wait for key to be pressed
            key = cv2.waitKey(1)

        except KeyboardInterrupt:
            # close all threads
            for i, thread in enumerate(threads):
                print('Closing capture thread for camera {}'.format(i))
                thread.join()
            # break out of loop
            break
        except:
            pass

    # destroy window
    cv2.destroyAllWindows()