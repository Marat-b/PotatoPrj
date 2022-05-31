import onnx
from onnxsim import simplify

# load your predefined ONNX model
model = onnx.load('../weights/model_202205311000_ov12.onnx')

# convert model
model_simplified, check = simplify(model)

assert check, "Simplified ONNX model could not be validated"

onnx.save(model_simplified, "../weights/smodel_202205311000_ov12.onnx")
