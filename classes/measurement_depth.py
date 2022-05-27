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
        self.focal_length_in_pixel = 882.5
        self.baseline = 0.075
        # self._distance_to_object = distance_to_object

    def _get_focal_pixel(self):
        """
        Get focal length in pixel
        :return focal_pixel: focal length in pixel
        """
        # focal_pixel = (self._object_width * 0.5) / math.tan((self._fov * 0.5 * math.pi) / 180)
        return self.focal_length_in_pixel

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
        return width

    # def _get_width_pixel_ts(self, image_mask):
    #     """
    #     Get biggest side of object (width) in pixels
    #     Parameters:
    #     ----------
    #         image_mask:  np.array -  image mask (black-white)
    #     Returns:
    #     -------
    #         width: int - width in pixel
    #     """
    #
    #     def calculate_distance(point_a, point_b):
    #         dist = np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)
    #         return dist
    #
    #     # width = box[2] - box[0]
    #     # height = box[3] - box[1]
    #     # print(f'width={width}, height={height}')
    #     # box_len = width if width > height else height
    #     # mask = image_mask[0]  # .astype('uint8')
    #     # mask[mask > 0.9] = 1.0
    #     # mask = mask.astype('uint8')
    #     f = np.argwhere(image_mask > 0)
    #     f_max = np.argmax(f, axis=0)
    #     bottom = f[f_max[0]]
    #     right = f[f_max[1]]
    #     f_min = np.argmin(f, axis=0)
    #     top = np.array([f[f_min[0]][0], bottom[1]])
    #     left = np.array([right[0], f[f_min[1]][1]])
    #     width = calculate_distance(left, right)
    #     height = calculate_distance(top, bottom)
    #     length = width if width > height else height
    #     # return int((box_len * length) / 28)
    #     return int(length)

    def _distance_to_object(self, disparity) -> float:
        """
        Get distance from focal length, baseline and disparity
        Parameters
        ----------
        disparity : float

        Returns
        -------
         distance : float

        """
        distance = self.focal_length_in_pixel * self.baseline / disparity
        print(f'distance={distance}')
        return distance

    def get_width_meter(self, image_mask, disparity, box):
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
        print(f'width_meter={width_meter}')
        return width_meter

