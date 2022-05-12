# to start together with video_hub.py
from multiprocessing import Process, Queue
import cv2
import simplejpeg

import izmq


def video_emitter(window_name: str, connect_to: str, queue: Queue):
    """
    Emit video frames from queue
    Parameters
    ----------
    window_name :
    connect_to :
    queue :

    Returns
    -------

    """
    sender = izmq.VideoSender(connect_to, REQ_REP=False)
    while True:  # send images until Ctrl-C
        if not queue.empty():
            image = queue.get(False)
            if image is None:
                break
            jpg_buffer = simplejpeg.encode_jpeg(image, quality=95, colorspace='BGR')
            sender.send_jpg(window_name, jpg_buffer)


def video_collector(cam, queue: Queue) -> None:
    """
    Send frame to queue from video source.
    :return: None
    """
    cap = cv2.VideoCapture(cam)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cap.isOpened():
        print('cap is opened')
    # else:
    #     print('* cap is NOT opened -> name={}, cam={}'.format(window_name, cam))
    while True:
        if cap.isOpened():
            print('cap is opened')
        else:
            print('cap is closed')
        _, frame = cap.read()
        if _:
            if not queue.full():
                # print('queue put')
                queue.put(frame, False)
        else:
            print('***  Did not read')
            cap.release()
            cap = cv2.VideoCapture(cam)


if __name__ == '__main__':
    wnd_name = 'clientX'
    src = 'rtsp://admin:123456@192.168.1.10:554/H264?ch=1&subtype=0'
    host_hub = '127.0.0.1:5555'
    q = Queue(maxsize=256)
    Process(target=video_collector, args=(src, q,)).start()
    Process(target=video_emitter, args=(wnd_name, host_hub, q,)).start()
