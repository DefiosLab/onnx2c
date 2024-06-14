import argparse
import onnxruntime
import onnx
import numpy as np
import cv2
import time
import sys
import os
import ctypes

print("Loading model...", end="")
model_path = "mnist.onnx"
model = onnx.load(model_path)
print("Done")


image = cv2.imread("input.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (28, 28)).astype(np.float32) / 255
img = np.reshape(gray, (1, 1, 28, 28))
img = np.random.rand((1,3,224,224))


# onnxruntime
ort_sess = onnxruntime.InferenceSession(model_path)
input_name = ort_sess.get_inputs()[0].name


print("Inference start")
# onnxruntimeでの推論
st = time.time()
result_ort = ort_sess.run(None, {input_name: img})[0]
ed = time.time()
ort_time = ed - st


lib = ctypes.CDLL("./resnet.so")
c_func = lib.run
c_func.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]

input_c = img.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
out = np.zeros((1, 10)).astype(np.float32)
output_c = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
result_vfv = out

# C言語での推論
st = time.time()
c_func(input_c, output_c)
ed = time.time()
vfv_time = ed - st


print("Predicted label : {}".format(np.argmax(result_vfv)))
print("onnxruntime time:{:.2f}sec".format(ort_time))
print("C Lang time:{:.2f}sec".format(vfv_time))


# 計算誤差
np.testing.assert_almost_equal(result_vfv, result_ort, decimal=2)


print(
    "maximum absolute error : {:.4e}".format(
        float(np.max(np.abs(result_vfv - result_ort)))
    )
)
print(
    "maximum relative error : {:.4e}".format(
        float(np.max(np.abs(result_vfv - result_ort) / result_vfv))
    )
)
