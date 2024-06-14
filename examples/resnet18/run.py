import argparse
import onnxruntime
import onnx
import numpy as np
import cv2
import time
import sys
import os
import ctypes
import urllib
import json


def download_imagenet_labels(filename="imagenet_labels.json"):
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            labels = json.load(f)
    else:
        with urllib.request.urlopen(url) as response:
            labels = json.loads(response.read())
            # ラベルをローカルファイルに保存
            with open(filename, "w") as f:
                json.dump(labels, f)

    return labels


print("Loading model...", end="")
model_path = "resnet18-v1-7.onnx"
model = onnx.load(model_path)
print("Done")
img = cv2.imread("cat.jpg")
img = img.transpose(2, 0, 1)
mean = np.array([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
std = np.array([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)
img = (img - mean) / std
img = np.ascontiguousarray(img.astype(np.float32))

out = np.zeros((1, 1000)).astype(np.float32)

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

output_c = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
result_vfv = out

# C言語での推論
st = time.time()
c_func(input_c, output_c)
ed = time.time()
vfv_time = ed - st

imagenet_labels = download_imagenet_labels()
pred = imagenet_labels[np.argmax(result_vfv)]
print("Predicted label : {}".format(pred))
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
