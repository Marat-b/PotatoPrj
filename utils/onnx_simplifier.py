import onnx
from onnxsim import simplify

# load your predefined ONNX model
model = onnx.load('../weights/potato_20220606_50.onnx')

# convert model
model_simplified, check = simplify(model)

assert check, "Simplified ONNX model could not be validated"

onnx.save(model_simplified, "../weights/spotato_20220606_50.onnx")
