import argparse
import os
import time
from distutils.util import strtobool

import cv2
from tqdm import tqdm

from calculator import Calculator
from classes.bbox import Bbox
from classes.drawer import Drawer
from classes.entity import Entity
from classes.identity import Identity
from classes.mask import Mask
from deep_sort import DeepSort
from detectron2_detection import Detectron2
from funcz import DrawAreaRect2
from measurement import Measurement
from util import cv2_imshow, draw_bboxes


class Detector(object):
    def __init__(self, args):
        self.args = args
        self.class_names = ['strong', 'sick', 'stone']
        use_cuda = bool(strtobool(self.args.use_cuda))
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.detectron2 = Detectron2(
            detectron2_checkpoint=args.detectron2_checkpoint, num_classes=3,
            use_cuda=use_cuda
            )

        self.deepsort = DeepSort(args.deepsort_checkpoint, use_cuda=use_cuda)

    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.num_frames = int(self.vdo.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.vdo.release()
        self.output.release()
        cv2.destroyAllWindows()
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self):
        drawer = Drawer()
        drawer.add_bbox(Bbox()).add_identity(Identity()).add_entity(Entity()).add_mask(Mask()).add_measurement(
            Measurement(3840, 120, 1.5))\
            .add_calculator(Calculator([['small', 0.0, 0.035], ['middle', 0.035, 0.08], ['big', 0.08, 1.0]]))

        for i in tqdm(range(self.num_frames)):
            if not self.vdo.grab():
                continue
            # start = time.time()
            _, im = self.vdo.retrieve()
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            bbox_xcycwh, cls_conf, cls_ids, masks = self.detectron2.detect(im)
            # print(f'len(cls_ids)={len(cls_ids)}, len(cls_conf)={len(cls_conf)}, len(masks)={len(masks)}')
            # print(f'masks={masks}')
            # for mask in masks:
                # cv2_imshow(mask)
                # show_area(mask)

            if len(bbox_xcycwh) > 0:
                # select class person
                mask = cls_ids == 0

                bbox_xcycwh = bbox_xcycwh[mask]
                bbox_xcycwh[:, 3:] *= 1.2

                cls_conf = cls_conf[mask]
                outputs = self.deepsort.update(bbox_xcycwh, cls_conf, im, cls_ids, masks)
                # print(f'outputs={outputs}')
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    im = drawer.outputs(im, outputs)
                    # print(f'len(bbox={len(bbox)})')
                    identities = outputs[:, -3]
                    cls_id = outputs[:, -2]
                    msk = outputs[:, -1]
                    # cls_id.sort()
                    # print(f'len(cls_id)={len(cls_id)}, cls_id={cls_id}')
                    # print(f'msk={msk}')
                    # im = draw_bboxes(im, bbox_xyxy, identities, cls_id=cls_id, masks=msk, class_names=self.class_names)


            # end = time.time()
            # print("time: {}s, fps: {}".format(end - start, 1 / (end - start)))

            if self.args.display:
                cv2.imshow("test", im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.output.write(im)
            # exit(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--detectron2_checkpoint", type=str, default=None)
    parser.add_argument("--max_dist", type=float, default=0.3)
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="demo.avi")
    parser.add_argument("--use_cuda", type=str, default="True")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with Detector(args) as det:
        det.detect()
    print(' *************  END ***********************')
