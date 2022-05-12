from imutils.video import WebcamVideoStream
import cv2


class WVS(WebcamVideoStream):
    def __init__(self,  src=0, name="WebcamVideoStream"):
        super().__init__(src, name)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
