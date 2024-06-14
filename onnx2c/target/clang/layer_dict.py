from .layer import *

layer_dict = {
    "Relu": Relu,
    "Conv": Conv,
    "MaxPool": MaxPool,
    "Reshape": Reshape,
    "Gemm": Gemm,
    "BatchNormalization": BatchNormalization,
    "Add": Add,
    "GlobalAveragePool": GlobalAveragePool,
    "Flatten": Flatten,
}
