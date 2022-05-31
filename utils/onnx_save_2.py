import os
from multiprocessing import freeze_support

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.export import Caffe2Tracer, TracingAdapter
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.export.caffe2_export import export_onnx_model
from detectron2.modeling import GeneralizedRCNN, build_model
import cv2
import numpy as np
import torch
import onnx


def get_images(path):
    tmpl = '{}/{}'
    l_images = []
    l_inputs = []
    print(path)
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    print(len(files))
    for file in files:
        im = cv2.imread(tmpl.format(path, file))
        im = cv2.resize(im, (1024, 1024))
        im_t = torch.as_tensor(im.astype("uint8").transpose(2, 0, 1))
        l_images.append(im_t)
        l_inputs.append({'image': im_t})
    # print(f'images={l_images[0]}\ninputs={l_inputs[0]}')
    return l_images, l_inputs


def main():
    weights_path = '../weights/potato_model_best_202205311000.pth'
    # weights_path = '../weights/potato_model_final_202204221400.pth'
    # img = cv2.imread('../images/non_ts_1.jpg')

    torch._C._jit_set_bailout_depth(1)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.DATASETS.TEST = ()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = 'cpu'
    cfg.freeze()
    model = build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS)
    model.eval()

    # image = torch.as_tensor(img.astype("uint8").transpose(2, 0, 1))
    # image = torch.rand(3, 1024, 1024)
    # print(f'image.shape={image.shape}, image.dtype={image.dtype}')

    # inputs = [{"image": image}]  # remove other unused keys
    images, inputs = get_images(r"C:\softz\work\potato\dataset\set23")
    # print(f'images={images[0]}\ninputs={inputs[0]}')
    # exit()
    if isinstance(model, GeneralizedRCNN):
        print('inference is Not None')

        def inference(model, inputs):
            # use do_postprocess=False so it returns ROI mask
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

    else:
        print('inference is None')
        inference = None  # assume that we just call the model directly
    traceable_model = TracingAdapter(model, inputs, inference)
    torch.onnx.export(
        traceable_model, tuple(images), '../weights/model_202205311000_ov12.onnx', opset_version=12,
        do_constant_folding=True
    )


if __name__ == '__main__':
    freeze_support()
    main()
