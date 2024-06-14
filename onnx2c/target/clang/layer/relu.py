import onnx
import numpy as np
from onnx2c.target import Layer


class Relu(Layer):
    def __init__(self, model, node, tensor_data, gen):
        super().__init__(model, node, tensor_data)
        self.gen = gen
        self.model = model

    def run(self):
        if self.gen.check_input(self.input_name[0]):
            input0_mdf = ""
        else:
            input0_mdf = "&"
        input_data = self.tensor_data[self.input_name[0]]
        batch, channels, height, width = input_data.shape

        if self.output_name not in [output.name for output in self.model.graph.output]:
            shape = f"{{1,{channels},{height},{width}}}"
            output_mdf = "&"
            output_code = f"float {self.output_name}_data[{channels * height*width}];\n"
            output_code += f"float_tensor {self.output_name} = {{ 4,{shape},{self.output_name}_data }};\n"
            self.gen.write_header(output_code)
        else:
            output_mdf = ""

        op_func = f"Relu_F32({input0_mdf}{self.input_name[0]},{output_mdf}{self.output_name});\n"
        self.gen.write_source(op_func)
