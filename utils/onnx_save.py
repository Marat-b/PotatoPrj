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


def main():
    # weights_path = '../weights/potato_model_best_202205311000.pth'
    # weights_path = '../weights/potato_model_final_202204221400.pth'
    weights_path = '../weights/potato_model_202206021720.pth'

    # register_coco_instances(
    #     "potato_dataset_test", {},
    #     r"C:\softz\work\potato\dataset\potato_set6_coco.json",
    #     r"C:\softz\work\potato\dataset\set6"
    # )
    # register_coco_instances(
    #     "potato_dataset_test26", {},
    #     r"C:\softz\work\potato\dataset\potato_set26_coco.json",
    #     r"C:\softz\work\potato\dataset\set26"
    # )

    img = cv2.imread('../images/non_ts_1.jpg')
    img = cv2.resize(img, (1024, 1024))

    torch._C._jit_set_bailout_depth(1)
    cfg = get_cfg()
    # cfg = add_export_config(cfg)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.DATASETS.TEST = ()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = 'cpu'
    # cfg.freeze()
    # cfg = add_export_config(cfg)
    model = build_model(cfg).cpu()
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS)
    model.eval()

    # height, width = img.shape[:2]
    image = torch.as_tensor(img.astype("uint8").transpose(2, 0, 1))
    # image = torch.rand(3, 1024, 1024)
    print(f'image.shape={image.shape}, image.dtype={image.dtype}')
    # inputs = {"image": image, "height": height, "width": width}

    # Export to Onnx model
    # onnxModel = export_onnx_model(model, [inputs])
    # onnx.save(onnxModel, "/content/deploy.onnx")
    ##############################################################
    # data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST)
    # first_batch = next(iter(data_loader))
    # image = first_batch[0]["image"]
    # print(f'image.shape={image.shape}, image.dtype={image.dtype}')
    # data_loader_i = iter(data_loader)
    # print(f'data_loader_i={data_loader_i}')
    # images = [img[0]["image"] for img in data_loader_i]
    # exit()
    # image = first_batch[0]["image"]
    #######################################################
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
    torch.onnx.export(
        traceable_model, (image,), '../weights/model_202205311000_ov12.onnx', opset_version=12,
        do_constant_folding=True, verbose=True
    )


if __name__ == '__main__':
    freeze_support()
    main()
