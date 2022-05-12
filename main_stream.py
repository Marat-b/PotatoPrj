from distutils.util import strtobool

import cv2

from calculator import Calculator
from classes.bbox import Bbox
from classes.drawer import Drawer
from classes.entity import Entity
from classes.identity import Identity
from classes.mask import Mask
from config.config import DEEPSORT, DETECTRON2, DISPLAY, DISPLAY_HEIGHT, DISPLAY_WIDTH, USE_CUDA
from deep_sort import DeepSort
from detectron2_detection import Detectron2
from measurement import Measurement


class Detector(object):
    def __init__(self):
        self.class_names = ['strong', 'sick', 'stone']
        use_cuda = USE_CUDA
        self.display = DISPLAY
        self.detectron2 = Detectron2(
            detectron2_checkpoint=DETECTRON2, num_classes=len(self.class_names),
            use_cuda=use_cuda
        )

        self.deepsort = DeepSort(DEEPSORT, use_cuda=use_cuda)
        self.drawer = Drawer().add_bbox(Bbox()).add_identity(Identity()).add_entity(Entity()).add_mask(
            Mask(
            )
        ).add_measurement(
            Measurement(3840, 120, 1.5)
        ) \
            .add_calculator(Calculator([['small', 0.0, 0.035], ['middle', 0.035, 0.08], ['big', 0.08, 1.0]])) \
            .add_class_names(self.class_names)

    def detect(self, image):
        """
        Detection objects from image
        Parameters:
        ----------
            image: np.array - image

        Returns:
        -------
            image: np.array - image with detected objects

        """
        bbox_xcycwh, cls_conf, cls_ids, masks = self.detectron2.detect(image)
        # print(f'len(cls_ids)={len(cls_ids)}, len(cls_conf)={len(cls_conf)}, len(masks)={len(masks)}')

        if len(bbox_xcycwh) > 0:
            # select class person
            mask = cls_ids == 0

            bbox_xcycwh = bbox_xcycwh[mask]
            bbox_xcycwh[:, 3:] *= 1.2

            cls_conf = cls_conf[mask]
            outputs = self.deepsort.update(bbox_xcycwh, cls_conf, image, cls_ids, masks)
            # print(f'outputs={outputs}')
            if len(outputs) > 0:
                image = self.drawer.outputs(image, outputs)
                print(f'len(outputs)={len(outputs)}')

        return image
