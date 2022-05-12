from imagezmq import ImageSender
import numpy as np


class VideoSender(ImageSender):
    def __init__(self, connect_to='tcp://127.0.0.1:5555', REQ_REP=True):
        super().__init__(connect_to, REQ_REP)

    def send_image_pubsub(self, msg, image):
        """Sends OpenCV image and msg hub computer in PUB/SUB mode. If
        there is no hub computer subscribed to this socket, then image and msg
        are discarded.

        Arguments:
          msg: text message or image name.
          image: OpenCV image to send to hub.

        Returns:
          Nothing; there is no reply from hub computer in PUB/SUB mode
        """

        if image is not None:
            if image.flags['C_CONTIGUOUS']:
                # if image is already contiguous in memory just send it
                self.zmq_socket.send_array(image, msg, copy=False)
            else:
                # else make it contiguous before sending
                image = np.ascontiguousarray(image)
                self.zmq_socket.send_array(image, msg, copy=False)
