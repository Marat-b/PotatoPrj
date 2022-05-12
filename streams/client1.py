import socket
import time
from imutils.video import FileVideoStream, VideoStream, WebcamVideoStream

# Accept connections on all tcp addresses, port 5555
import izmq
from wvs import WVS

video_file_path = r'C:\softz\work\potato\in\video\20220506_111610_10s.mp4'
video_cam = 'rtsp://admin:123456@192.168.1.10:554/H264?ch=1&subtype=0'
sender = izmq.VideoSender(connect_to='tcp://127.0.0.1:5555', REQ_REP=False)

rpi_name = 'client1'  # socket.gethostname()  # send RPi hostname with each image
# video_stream = FileVideoStream(path=video_file_path, queue_size=512).start()
video_stream = WVS(src=video_cam, name=rpi_name).start()
# time.sleep(2.0)  # allow camera sensor to warm up
while True:  # send images until Ctrl-C
    image = video_stream.read()
    if image is None:
        break
    sender.send_image(rpi_name, image)
    # The execution loop will continue even if no subscriber is connected
