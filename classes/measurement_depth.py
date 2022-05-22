import numpy as np
import cv2
import math


class Measurement:
    def __init__(self, object_width, fov):
        """
           :param object_width: matrix length in pixel
           :param fov: focal of view in degree

           """
        self._object_width = object_width
        self._fov = fov
        # self._distance_to_object = distance_to_object

    def _get_focal_pixel(self):
        """
        Get focal length in pixel
        :return focal_pixel: focal length in pixel
        """
        focal_pixel = (self._object_width * 0.5) / math.tan((self._fov * 0.5 * math.pi) / 180)
        return focal_pixel

    def _get_width_pixel(self, image_mask):
        """
        Get biggest side of object (width) in pixels
        Parameters:
        ----------
            image_mask:  np.array -  image mask (black-white)
        Returns:
        -------
            width: int - width in pixel
        """
        # blurred = cv2.GaussianBlur(image_mask, (5, 5), 0)
        # thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        # print(f'contours={np.array(contours).shape}')
        # print(f'contours={contours}')
        width = 0
        for i, cnt in enumerate(contours):
            rect = cv2.minAreaRect(cnt)
            sides = rect[1]
            width = sides[0] if sides[0] > sides[1] else sides[1]
            # print(f'show_areas width={width}')
        return int(width)

    def _distance_to_object(self, disparity) -> float:
        distance = 882.5 * 0.075 / disparity
        return distance

    def get_width_meter(self, image_mask, disparity):
        """
         Get biggest side of object (width) in meter
        Parameters:
        __________
            image_mask: np.array - image mask (black-white)
            disparity: int - disparity in pixels
        Returns:
        ________
            width_meter: int - width in meter
        """
        width_pixel = self._get_width_pixel(image_mask)
        focal_pixel = self._get_focal_pixel()
        distance_to_object = self._distance_to_object(disparity)
        width_meter = (width_pixel * distance_to_object) / focal_pixel
        return width_meter

