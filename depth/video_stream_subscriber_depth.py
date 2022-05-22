"""pub_sub_receive.py -- receive OpenCV stream using PUB SUB."""

import sys

import socket
import traceback
import cv2
from imutils.video import VideoStream
import imagezmq
import threading
import numpy as np
from time import sleep
from multiprocessing import Queue
import simplejpeg

from main_stream_depth import Detector


# Helper class implementing an IO deamon thread
class VideoStreamSubscriber:

    def __init__(self, hostname):
        self._data = '', None
        self.queue = Queue(maxsize=256)
        self.detector = Detector()
        self.hostname = hostname
        self._stop = False
        self._data_ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(self.queue,))
        self._thread.daemon = True
        self._thread.start()

    def receive(self):
        """
        Receive data from queue and output it
        Returns:
        -------
            data: Tuple[str, nd.array]
        """
        if not self.queue.empty():
            # print('get from queue')
            data_received = self.queue.get()
            img = self.detector.detect(data_received[1])
            data = data_received[0], img
        else:
            # print(' ! queue is empty')
            data = '', None
        return data

    def _run(self, queue):
        receiver = imagezmq.ImageHub("tcp://{}".format(self.hostname), REQ_REP=False)
        while not self._stop:
            if not queue.full():
                sent_from, jpeg_buffer = receiver.recv_image()
                # img = simplejpeg.decode_jpeg(jpeg_buffer, colorspace='BGRA')
                img = cv2.imdecode(np.frombuffer(jpeg_buffer, dtype='uint8'), cv2.IMREAD_UNCHANGED)
                # print('put to queue')
                queue.put((sent_from, img))
            # else:
            #     print('!!! queue is full')
        receiver.close()

    def close(self):
        self._stop = True


if __name__ == "__main__":
    # Receive from broadcast
    # There are 2 hostname styles; comment out the one you don't need
    host = "127.0.0.1:5555"  # Use to receive from localhost
    video_subscriber = VideoStreamSubscriber(host)

    try:
        while True:
            msg, frame = video_subscriber.receive()
            image = cv2.imdecode(np.frombuffer(frame, dtype='uint8'), -1)
            cv2.imshow("Pub Sub Receive", image)
            cv2.waitKey(1)
    except (KeyboardInterrupt, SystemExit):
        print('Exit due to keyboard interrupt')
    except Exception as ex:
        print('Python error with no Exception handler:')
        print('Traceback error:', ex)
        traceback.print_exc()
    finally:
        video_subscriber.close()
        sys.exit()
