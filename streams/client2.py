import socket
import time
from imutils.video import FileVideoStream, VideoStream


# Accept connections on all tcp addresses, port 5555
import izmq

video_file_path = r'C:\softz\work\potato\in\video\20220506_113122_5s.mp4'
sender = izmq.VideoSender(connect_to='tcp://127.0.0.1:5556', REQ_REP=False)

rpi_name = 'client2'  # socket.gethostname()  # send RPi hostname with each image
picam = FileVideoStream(path=video_file_path, queue_size=512).start()
# time.sleep(2.0)  # allow camera sensor to warm up
while True:  # send images until Ctrl-C
    image = picam.read()
    if image is None:
        break
    sender.send_image(rpi_name, image)
    # The execution loop will continue even if no subscriber is connected