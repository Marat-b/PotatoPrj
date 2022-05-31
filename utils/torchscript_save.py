from multiprocessing import freeze_support

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.export import Caffe2Tracer, TracingAdapter, add_export_config, dump_torchscript_IR
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import GeneralizedRCNN, build_model
import cv2
import numpy as np
import torch



def main():
    weights_path = '../weights/potato_model_best_202205311000.pth'

    register_coco_instances(
        "potato_dataset_test", {},
        r"C:\softz\work\potato\dataset\potato_set6_coco.json",
        r"C:\softz\work\potato\dataset\set6"
    )

    img = cv2.imread('../images/potato.jpg')

    cfg = get_cfg()
    # cfg = add_export_config(cfg)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.DATASETS.TEST = ('potato_dataset_test',)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = 'cpu'
    model = build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS)
    model.eval()

    # height, width = img.shape[:2]
    image = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
    # inputs = {"image": image, "height": height, "width": width}
    inputs = [{"image": image}]  # remove other unused keys
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
    ################ torchscript ###################################################
    ts_model = torch.jit.trace(traceable_model, (image,))
    torch.jit.save(ts_model, '../weights/potato_model_202205311000.ts')
    # dump_torchscript_IR(ts_model, '../weights/ts')


if __name__ == '__main__':
    freeze_support()
    main()
