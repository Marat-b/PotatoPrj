import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession("../weights/model_202205021400_ov12.onnx")

outputs = ort_session.run(
    None,
    {"Concat_12758": np.random.randn(10, 3, 1024, 1024).astype(np.float32)},
)
print(outputs[0])
