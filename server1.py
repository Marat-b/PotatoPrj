import cv2
import imagezmq

# Instantiate and provide the first sender / publisher address
from main_stream import Detector

image_hub = imagezmq.ImageHub(open_port='tcp://192.168.1.129:5555', REQ_REP=False)
image_hub.connect('tcp://192.168.1.129:5556')
# image_hub.connect('tcp://192.168.0.102:5555')  # must specify address for every sender
# image_hub.connect('tcp://192.168.0.103:5555')  # repeat as needed
# cv2.namedWindow('client1', cv2.WINDOW_NORMAL)
# cv2.namedWindow('client2', cv2.WINDOW_NORMAL)
detector = Detector()
while True:  # show received images
    window_name, image = image_hub.recv_image()
    print(f'window_name={window_name}')
    if image is None:
        break
    image = cv2.resize(image, (600, 600))
    detector.detect(window_name, image)
    # cv2.imshow(window_name, image)  # 1 window for each unique RPi name
    # cv2.waitKey(10)
