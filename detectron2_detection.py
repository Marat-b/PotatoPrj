from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo

setup_logger()

import numpy as np

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


class Detectron2:

    def __init__(self, detectron2_checkpoint=None, num_classes=1, score_thresh_test=0.7, use_cuda=True):
        self.cfg = get_cfg()
        # self.cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh_test  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = detectron2_checkpoint if detectron2_checkpoint else \
            "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
        if not use_cuda:
            self.cfg.MODEL.DEVICE = 'cpu'
        self.predictor = DefaultPredictor(self.cfg)

    def bbox(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return cmin, rmin, cmax, rmax

    def detect(self, im):
        outputs = self.predictor(im)
        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        classes = outputs["instances"].pred_classes.cpu().numpy()
        scores = outputs["instances"].scores.cpu().numpy()
        pr_masks = outputs["instances"].pred_masks.cpu().numpy()
        pr_masks = pr_masks.astype(np.uint8)
        pr_masks[pr_masks > 0] = 255

        bbox_xcycwh, cls_conf, cls_ids, masks = [], [], [], []

        for (box, _class, score, pr_mask) in zip(boxes, classes, scores, pr_masks):

            if _class == 0:
                x0, y0, x1, y1 = box
                bbox_xcycwh.append([(x1 + x0) / 2, (y1 + y0) / 2, (x1 - x0), (y1 - y0)])
                cls_conf.append(score)
                cls_ids.append(_class)
                masks.append(pr_mask)

        return np.array(bbox_xcycwh, dtype=np.float64), np.array(cls_conf), np.array(cls_ids), np.array(masks)
