import onnx
import numpy as np
from onnx2c.target import Layer
from onnx2c.helper import same_upper


class MaxPool(Layer):
    def __init__(self, model, node, tensor_data, gen):
        super().__init__(model, node, tensor_data)
        self.gen = gen
        self.model = model

    def run(self):
        input_data = self.tensor_data[self.input_name[0]]

        if "pads" in self.attrs:
            padding = self.attrs["pads"]
        else:
            padding = [0, 0, 0, 0]
        if "strides" in self.attrs:
            stride = self.attrs["strides"]
        else:
            stride = [1, 1]
        if "kernel_shape" in self.attrs:
            kernel_shape = self.attrs["kernel_shape"]
        if "auto_pad" in self.attrs:
            auto_pad = self.attrs["auto_pad"]
            if b"SAME_UPPER" == auto_pad:
                padding = same_upper(input_data, stride, kernel_shape)
            elif b"SAME_LOWER" == auto_pad:
                padding = same_lower(input_data, stride, kernel_shape)
            elif b"NOTSET" == auto_pad:
                pass
            else:
                raise NotImplementedError(f"{auto_pad} is not supported")
        assert input_data.ndim == 4, "Input data must be 4-dimensional"
        batch_size, in_channels, in_height, in_width = input_data.shape

        begin_i, begin_j, end_i, end_j = padding
        stride_height, stride_width = stride

        attr_name = self.node.name + "_attr"
        attrs_code = f"""
pool_attrs {attr_name} = {{  {self.gen.array2text(padding,len(padding))},{self.gen.array2text(kernel_shape,len(kernel_shape))},{self.gen.array2text(stride,len(stride))} }};
"""
        self.gen.write_header(attrs_code)

        out_height = (
            in_height - kernel_shape[0] + begin_i + end_i
        ) // stride_height + 1
        out_width = (in_width - kernel_shape[1] + begin_j + end_j) // stride_width + 1
        if self.gen.check_input(self.input_name[0]):
            input0_mdf = ""
        else:
            input0_mdf = "&"
        if self.output_name not in [output.name for output in self.model.graph.output]:
            shape = f"{{1,{in_channels},{out_height},{out_width}}}"
            output_mdf = "&"
            output_code = f"float {self.output_name}_data[{in_channels * out_height*out_width}] = {{}};\n"
            output_code += f"float_tensor {self.output_name} = {{ 4,{shape},{self.output_name}_data }};\n"
            self.gen.write_header(output_code)
        else:
            output_mdf = ""
        op_func = f"MaxPool_F32({input0_mdf}{self.input_name[0]},{output_mdf}{self.output_name},&{attr_name});\n"
        self.gen.write_source(op_func)
