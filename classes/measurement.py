import numpy as np
import cv2
import math


class Measurement:
    def __init__(self, object_width, fov, distance_to_object):
        """
           :param object_width: matrix length in pixel
           :param fov: focal of view in degree
           :param distance_to_object: distance to object from the observer in meter
           """
        self._object_width = object_width
        self._fov = fov
        self._distance_to_object = distance_to_object

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

    def _get_width_pixel_ts(self, image_mask, box):
        """
        Get biggest side of object (width) in pixels
        Parameters:
        ----------
            image_mask:  np.array -  image mask (black-white)
        Returns:
        -------
            width: int - width in pixel
        """
        def calculate_distance(point_a, point_b):
            dist = np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)
            return dist
        width = box[2] - box[0]
        height = box[3] - box[1]
        # print(f'width={width}, height={height}')
        box_len = width if width > height else height
        mask = image_mask[0]  # .astype('uint8')
        mask[mask > 0.9] = 1.0
        mask = mask.astype('uint8')
        f = np.argwhere(mask > 0)
        if len(f)>0:
            f_max = np.argmax(f, axis=0)
            bottom = f[f_max[0]]
            right = f[f_max[1]]
            f_min = np.argmin(f, axis=0)
            top = np.array([f[f_min[0]][0], bottom[1]])
            left = np.array([right[0], f[f_min[1]][1]])
            width = calculate_distance(left, right)
            height = calculate_distance(top, bottom)
            length = width if width > height else height
            return int((box_len * length) / 28)
        else:
            print(f'mask={mask}')
            print(f'f={f}')
            return 0

    def get_width_meter(self, image_mask, box):
        """
         Get biggest side of object (width) in meter
        Parameters:
        __________
            image_mask: np.array - image mask (black-white)
            box: list - x1, y1, x2, y2
        Returns:
        ________
            width_meter: int - width in meter
        """
        width_pixel = self._get_width_pixel_ts(image_mask, box)
        focal_pixel = self._get_focal_pixel()
        width_meter = (width_pixel * self._distance_to_object) / focal_pixel
        return width_meter

