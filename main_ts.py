import argparse
import os
from distutils.util import strtobool

import cv2
import torch
from tqdm import tqdm

from classes.calculator2 import Calculator2
from classes.bbox import Bbox
from classes.drawer2 import Drawer2
from classes.entity import Entity
from classes.identity import Identity
from classes.mask import Mask
from classes.ts_detection import TorchscriptDetection
from deep_sort import DeepSort
from classes.measurement import Measurement


class Detector(object):
    def __init__(self, args):
        self.args = args
        # self.class_names = ['strong', 'sick', 'stone']
        self.class_names = ['strong', 'alternariosis', 'anthracnose', 'fomosis', 'fusarium', 'internalrot',
                            'necrosis', 'phytophthorosis', 'pinkrot', 'scab', 'wetrot']
        self.confidence = args.confidence
        self.max_dist = args.max_dist
        self.min_confidence = args.min_conf
        self.display = bool(strtobool(args.display))
        use_cuda = bool(strtobool(self.args.use_cuda))
        if  self.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.detectron2 = TorchscriptDetection(
            args.detectron2_checkpoint,
            use_cuda=use_cuda
        )

        self.deepsort = DeepSort(args.deepsort_checkpoint, max_dist=self.max_dist,
        min_confidence=self.min_confidence,  use_cuda=use_cuda)

    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.num_frames = int(self.vdo.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.vdo.get(cv2.CAP_PROP_FPS))

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, self.fps, (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.vdo.release()
        self.output.release()
        cv2.destroyAllWindows()
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self):
        drawer = Drawer2()
        drawer.add_bbox(Bbox()).add_identity(Identity()).add_entity(Entity()).add_mask(Mask()).add_measurement(
            Measurement(3840, 120, 1.5)
        ) \
            .add_calculator(Calculator2([['small', 0.0, 0.035], ['middle', 0.035, 0.08], ['big', 0.08, 1.0]])) \
            .add_class_names(self.class_names)

        # counter = int(self.fps / 2)
        # count = 0

        for i in tqdm(range(self.num_frames)):
            if not self.vdo.grab():
                continue
            _, im = self.vdo.retrieve()
            # if i % 2 == 0:
            #     count += 1
            #     continue
            # count = 0
            # im = im[:, :, [2, 0, 1]] # BGR to RGB
            bbox_xcycwh, cls_conf, cls_ids, masks = self.detectron2.detect(im, confidence=self.confidence)
            # print(f'len(cls_ids)={len(cls_ids)}, len(cls_conf)={len(cls_conf)}, len(masks)={len(masks)}')

            if len(bbox_xcycwh) > 0:
                outputs = self.deepsort.update(bbox_xcycwh, cls_conf, im, cls_ids, masks)

                if len(outputs) > 0:
                    # print(f'outputs={outputs[:, :6]}')
                    im = drawer.outputs2(im, outputs)[0]

            if self.display:
                cv2.imshow("test", im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.output.write(im)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--deepsort_checkpoint", type=str, default="weights/ckpt.t7")
    parser.add_argument("--detectron2_checkpoint", type=str, default=None)
    parser.add_argument("--max_dist", type=float, default=0.3, help="Max distance, for Deepsort")
    parser.add_argument("--min_conf", type=float, default=0.7, help="Min confidence, for Deepsort")
    parser.add_argument("--display", dest="display", default="False")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="demo.mp4")
    parser.add_argument("--use_cuda", type=str, default="True")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence")
    return parser.parse_args()


if __name__ == "__main__":
    # torch.set_num_threads(1)
    args = parse_args()
    with Detector(args) as det:
        det.detect()
    print(' *************  END ***********************')
