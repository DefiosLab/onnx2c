import onnx
import numpy as np
from onnx2c.target import Layer
from onnx2c.helper import same_upper


class BatchNormalization(Layer):
    def __init__(self, model, node, tensor_data, gen):
        super().__init__(model, node, tensor_data)
        self.gen = gen
        self.model = model

    def run(self):
        x = self.tensor_data[self.input_name[0]]
        scale = self.tensor_data[self.input_name[1]]
        B = self.tensor_data[self.input_name[2]]
        mean = self.tensor_data[self.input_name[3]]
        var = self.tensor_data[self.input_name[4]]

        if "epsilon" in self.attrs:
            epsilon = self.attrs["epsilon"]
        else:
            epsilon = 1e-05

        if "momentum" in self.attrs:
            momentum = self.attrs["momentum"]
        else:
            momentum = 0.9

        assert x.ndim == 4, "Input data must be 4-dimensional"

        batch_size, in_channels, in_height, in_width = x.shape
        attr_name = self.node.name + "_attr"
        attrs_code = f"""
bn_attrs {attr_name} = {{  {epsilon},{momentum} }};
"""
        self.gen.write_header(attrs_code)

        input_arg0 = self.gen.generate_input_arg(self.input_name[0])
        input_arg1 = self.gen.generate_input_arg(self.input_name[1])
        input_arg2 = self.gen.generate_input_arg(self.input_name[2])
        input_arg3 = self.gen.generate_input_arg(self.input_name[3])
        input_arg4 = self.gen.generate_input_arg(self.input_name[4])

        # output tensorを設定
        shape = f"{{1,{in_channels},{in_height},{in_width}}}"
        if self.output_name not in [output.name for output in self.model.graph.output]:
            output_arg = "&" + self.output_name
            output_code = f"float {self.output_name}_data[{in_channels * in_height*in_width}] = {{}};\n"
            output_code += f"float_tensor {self.output_name} = {{ 4,{shape},{self.output_name}_data }};"
            self.gen.write_header(output_code)
        else:
            output_arg = self.output_name
        op_func = f"BatchNormalization_F32({input_arg0},{input_arg1},{input_arg2},{input_arg3},{input_arg4},&{attr_name},{output_arg});\n"
        self.gen.write_source(op_func)
