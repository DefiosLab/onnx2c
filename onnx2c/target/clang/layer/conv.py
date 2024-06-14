import onnx
import numpy as np
from onnx2c.target import Layer
from onnx2c.helper import same_upper


class Conv(Layer):
    def __init__(self, model, node, tensor_data, gen):
        super().__init__(model, node, tensor_data)
        self.gen = gen
        self.model = model

    def run(self):
        input_data = self.tensor_data[self.input_name[0]]
        weights = self.tensor_data[self.input_name[1]]
        if len(self.input_name) >= 3:
            bias_name = "&" + self.input_name[2]
            use_bias = "true"
        else:
            bias_name = "NULL"
            use_bias = "false"

        if "dilations" in self.attrs:
            for i in self.attrs["dilations"]:
                assert i == 1, "Only dilation of 1 is supported."
        if "group" in self.attrs:
            assert self.attrs["group"] == 1, "Only group of 1 is supported."

        if "pads" in self.attrs:
            padding = self.attrs["pads"]
        else:
            padding = [0, 0, 0, 0]
        if "strides" in self.attrs:
            stride = self.attrs["strides"]
        else:
            stride = [1, 1]
        assert input_data.ndim == 4, "Input data must be 4-dimensional"
        assert weights.ndim == 4, "Weights must be 4-dimensional"

        if "auto_pad" in self.attrs:
            auto_pad = self.attrs["auto_pad"]
            if b"SAME_UPPER" == auto_pad:
                padding = same_upper(input_data, stride, weights.shape[-2:])
            elif b"SAME_LOWER" == auto_pad:
                padding = same_lower(input_data, stride, weights.shape[-2:])
            elif b"NOTSET" == auto_pad:
                pass
            else:
                raise NotImplementedError(f"{auto_pad} is not supported")

        batch_size, in_channels, in_height, in_width = input_data.shape
        out_channels, _, kernel_height, kernel_width = weights.shape
        begin_i, begin_j, end_i, end_j = padding
        stride_height, stride_width = stride
        attr_name = self.node.name + "_attr"
        attrs_code = f"""
conv_attrs {attr_name} = {{  {self.gen.array2text(padding,len(padding))},{self.gen.array2text(stride,len(stride))} }};
"""
        self.gen.write_header(attrs_code)

        # output tensorを設定
        out_channels = weights.shape[0]
        out_height = (in_height - kernel_height + begin_i + end_i) // stride_height + 1
        out_width = (in_width - kernel_width + begin_j + end_j) // stride_width + 1
        shape = f"{{1,{out_channels},{out_height},{out_width}}}"

        if self.gen.check_input(self.input_name[0]):
            input0_mdf = ""
        else:
            input0_mdf = "&"
        if self.gen.check_input(self.input_name[1]):
            input1_mdf = ""
        else:
            input1_mdf = "&"
        if self.output_name not in [output.name for output in self.model.graph.output]:
            output_mdf = "&"
            output_code = f"float {self.output_name}_data[{out_channels * out_height*out_width}] = {{}};\n"
            output_code += f"float_tensor {self.output_name} = {{ 4,{shape},{self.output_name}_data }};"
            self.gen.write_header(output_code)
        else:
            output_mdf = ""
        op_func = f"Conv_F32({input0_mdf}{self.input_name[0]},{input1_mdf}{self.input_name[1]},{bias_name},{output_mdf}{self.output_name},&{attr_name},{use_bias});\n"
        self.gen.write_source(op_func)
