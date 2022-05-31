import cv2
import onnx
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.export import Caffe2Tracer
from detectron2.model_zoo import model_zoo
from detectron2.modeling import build_model

weights_path = '../weights/potato_model_best_202205311000.pth'
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
cfg.freeze()
# cfg = add_export_config(cfg)
model = build_model(cfg)
DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS)
model.eval()

height, width = img.shape[:2]
image = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))

inputs = {"image": image, "height": height, "width": width}
tracer = Caffe2Tracer(cfg, model, [inputs])
onnx_model = tracer.export_onnx()
onnx.save(onnx_model, '../weights/model_202205311000.onnx')
