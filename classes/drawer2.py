from typing import Dict, List, Tuple

import cv2
import numpy as np
from numpy import ndarray

from classes.drawer import Drawer


class Drawer2(Drawer):
    def __init__(self):
        super().__init__()

    def outputs2(self, img, outputs) -> Tuple[ndarray, Dict, Dict]:
        """
        Return  ndarray, dictionary, dictionary
        Parameters
        ----------
        img :
        outputs :

        Returns
        -------

        """
        image = img.copy()
        self._outputs = outputs
        bboxes = outputs[:, :4]
        self._bbox(bboxes)
        # print(f'outputs[:, -2]={outputs[:, -2]}')
        self._identity(outputs[:, -3])
        self._entity(outputs[:, -2])
        self._mask(outputs[:, -1])
        t_size = []
        # len_bbox = len(self._bbox)
        # print(f'outputs[:, :4]={outputs[:, :4]}')
        # print(f'outputs[:, -1]={outputs[:, -1]}')
        image = self.visualize_bw(image, outputs[:, :4], outputs[:, -1])
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = [int(j) for j in box]
            # print(f'x1={x1}, x2={x2}, y1={y1}, y2={y2}')
            width = float('{:.2f}'.format(self._measurement.get_width_meter(self._mask[i], box)))
            self._calculator.add(self._identity[i], width)
            self._calculator.add_class(self._identity[i], self._entity[i])
            color = self._color_list[int(self._identity[i] % self._color_index)]
            label = '{}-{:d} w={}m'.format(self._class_names[self._entity[i]], self._identity[i], str(width))
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(image, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(image, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        item_sorted = self._calculator.count()
        class_sorted = self._calculator.count_classes()
        # print(f'item_sorted={item_sorted}')
        # print(f'class_sorted={class_sorted}')
        for i, key in enumerate(item_sorted.keys()):
            cv2.putText(
                image, 'amount of {}={}'.format(key, str(item_sorted[key])),
                (5, 5 + t_size[1] + 4 + i * (t_size[1])), cv2.FONT_HERSHEY_PLAIN, 2, [0, 0, 255], 2
                )
        return image, item_sorted, class_sorted

    def visualize_bw(self, img, pr_boxes, masks) -> np.ndarray:
        """
        Get black-white image with colored ROI
        _________
        :param img:
        :type img: np.ndarray
        :param pr_boxes:
        :type pr_boxes: list
        :param masks:
        :type masks: list
        :return:
        :rtype: np.ndarray
        """
        image = img.copy()
        # image = cv2.resize(image, (1024, 1024))
        # pr_boxes, scores, pr_classes, masks = self.detect(image)
        # print(f'pred_boxes={pred_boxes}')
        # print(f'scores={scores}')
        # print(f'masks={masks}')
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_merged = cv2.merge((image_bw, image_bw, image_bw))
        for bbox, mask in zip(pr_boxes, masks):
            # print(f'image_mask={image_mask}')
            bbox = [int(x) for x in bbox]
            x0, y0, x1, y1 = bbox
            image_mask = self._get_mask(mask, bbox)
            z_mask = np.zeros_like(image)
            # print(f'z_mask.shape={z_mask.shape}, image_mask.shape={image_mask.shape}')
            image_mask_merged = cv2.merge((image_mask, image_mask, image_mask))
            z_mask[y0:y1, x0:x1] = image_mask_merged
            image2 = cv2.bitwise_and(image, z_mask * 255)
            # cv2_imshow(image2, 'image2')
            mask_inverted = cv2.bitwise_not(z_mask * 255)
            # cv2_imshow(mask_inverted, 'mask_inverted')
            image_merged = cv2.bitwise_or(cv2.bitwise_and(image_merged, mask_inverted), image2)
            # cv2_imshow(image_merged, 'new_image')

        # description = ""
        # for i, (bbox, score, cls) in enumerate(zip(pr_boxes, scores, pr_classes)):
        #     if score >= confidence:
        #         description += '{}. {}, вероятность {:.2f}%\n'.format(i + 1, self.class_names_ru[cls], score * 100)
        #         bbox = [int(x) for x in bbox]
        #         x0, y0, x1, y1 = bbox
        #         item_description = '{}. {}, score {:.2f}%'.format(i + 1, self.class_names[cls], score * 100)
        #         t_size = cv2.getTextSize(item_description, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        #         cv2.rectangle(image_merged, (x0, y0), (x1, y1), self.color_mask[cls], 3)
        #         cv2.rectangle(
        #             image_merged, (x0, y0), (x0 + t_size[0] + 3, y0 + t_size[1] + 4), self.color_mask[cls], -1
        #             )
        #         cv2.putText(
        #             image_merged, item_description, (x0, y0 + t_size[1] + 4),
        #             cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2
        #         )
        return image_merged

    def _get_mask(self, image_mask, box):
        """
        Get real mask from 28x28 mask
        Parameters:
        ----------
            image_mask:  np.array -  image mask (black-white)
        Returns:
        -------
            new_mask: np.array
        """

        width_box = int(box[2]) - int(box[0])
        height_box = int(box[3]) - int(box[1])
        # print(f'width_box={width_box}, height_box={height_box}')
        mask = image_mask[0]  # .astype('uint8')
        mask[mask > 0.9] = 1.0
        mask = mask.astype('uint8')
        new_mask = cv2.resize(mask, (width_box, height_box))
        return new_mask