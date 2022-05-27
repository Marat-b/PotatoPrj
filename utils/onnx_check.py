import onnx

# Load the ONNX model
model = onnx.load('../weights/model_202205021400_ov12.onnx')

# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))
