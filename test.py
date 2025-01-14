import onnxruntime as ort
import sys
import tensorflow as tf
import torch

print("CUDA available:", torch.cuda.is_available())
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("ONNX Runtime Providers:", ort.get_available_providers())
print("Python executable:", sys.executable)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))