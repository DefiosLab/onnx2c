#!/usr/bin/env python3
import argparse
import onnx2c
import onnx
from onnxsim import simplify
from onnx import version_converter
from onnx import shape_inference


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--onnx", type=str, required=True, help="input onnx path")
    parser.add_argument(
        "--input_data",
        type=str,
        help="Include the input data in the header.Please specify an npy file.",
    )

    args = parser.parse_args()
    return args


def version_convert(model):
    target_opset_version = 13
    try:
        model = version_converter.convert_version(model, target_opset_version)
        print("Model conversion successful.")
    except Exception as e:
        print(f"Error converting model: {e}")
    return model


def set_undefined_dims(model):
    graph = model.graph
    for input_tensor in graph.input:
        tensor_shape = input_tensor.type.tensor_type.shape
        for dim in tensor_shape.dim:
            if not (dim.dim_value):
                dim.dim_value = 1
    return model


def main():
    args = arg_parser()
    model = onnx.load(args.onnx)
    model = set_undefined_dims(model)
    model = shape_inference.infer_shapes(model)
    model, check = simplify(model)
    if model.opset_import[0].version != 13:
        print("Convert onnx to opset=13")
        model = version_convert(model)
    # Generate C
    gen = onnx2c.Generator(model)


if __name__ == "__main__":
    main()
