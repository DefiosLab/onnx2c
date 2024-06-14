import numpy as np


def elem_type2numpy(dtype_onnx):
    if dtype_onnx == 1:
        dtype_numpy = np.float32
    elif dtype_onnx == 2:
        dtype_numpy = np.uint8
    elif dtype_onnx == 3:
        dtype_numpy = np.int8
    elif dtype_onnx == 4:
        dtype_numpy = np.uint16
    elif dtype_onnx == 5:
        dtype_numpy = np.int16
    elif dtype_onnx == 6:
        dtype_numpy = np.int32
    elif dtype_onnx == 7:
        dtype_numpy = np.int64
    elif dtype_onnx == 9:
        dtype_numpy = np.bool_
    elif dtype_onnx == 10:
        dtype_numpy = np.float16
    elif dtype_onnx == 11:
        dtype_numpy = np.double
    else:
        raise ValueError(f"Unknown ONNX data type: {dtype_onnx}")
    return dtype_numpy
