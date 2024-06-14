import onnx
import numpy as np
from onnx2c.target import Layer


class GlobalAveragePool(Layer):
    def __init__(self, model, node, tensor_data, gen):
        super().__init__(model, node, tensor_data)
        self.gen = gen
        self.model = model

    def run(self):
        input_arg0 = self.gen.generate_input_arg(self.input_name[0])

        input_data = self.tensor_data[self.input_name[0]]
        batch, channels, height, width = input_data.shape

        if self.output_name not in [output.name for output in self.model.graph.output]:
            shape = f"{{{batch},{channels},{height},{width}}}"
            output_arg = "&" + self.output_name
            output_code = f"float {self.output_name}_data[{channels * height*width}];\n"
            output_code += f"float_tensor {self.output_name} = {{ 4,{shape},{self.output_name}_data }};\n"
            self.gen.write_header(output_code)
        else:
            output_arg = self.output_name

        op_func = f"GlobalAveragePool_F32({input_arg0},{output_arg});\n"
        self.gen.write_source(op_func)
