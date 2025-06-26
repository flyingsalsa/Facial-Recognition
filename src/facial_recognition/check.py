import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image # For loading reference image if needed
import onnxruntime as ort

print(f"ONNX runtime version : {ort.__version__}")
print("Avaliable exeution providers:", ort.get_available_providers())

assert 'CUDAExecutionProvider' in ort.get_available_providers() or 'TensorrtExecutionProvider' in ort.get_available_providers()

print("ONNX runtime is correctly configured to use the GPU")