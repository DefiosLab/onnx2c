import onnx
import numpy as np
from onnx2c.target import Layer


class Flatten(Layer):
    def __init__(self, model, node, tensor_data, gen):
        super().__init__(model, node, tensor_data)
        self.gen = gen
        self.model = model

    def run(self):
        shape = self.tensor_data[self.input_name[0]].shape
        output_code = ""
        if self.output_name not in [output.name for output in self.model.graph.output]:
            shape = [1, np.prod(np.array(shape[1:]))]
            ndim = len(shape)
            shape = self.gen.array2text(shape, len(shape))
            output_code += f"float_tensor {self.output_name} = {{ {ndim},{shape},{self.input_name[0]}_data }};"
            self.gen.write_header(output_code)
        else:
            size = np.prod(shape)
            shape = [1, np.prod(np.array(shape[1:]))]
            for i in range(len(shape)):
                output_code += f"{self.output_name}->shape[{i}] = {shape[i]};\n"
            output_code += f"memcpy({self.output_name}->data,{self.input_name[0]}.data,{size}*sizeof(float));\n"
            self.gen.write_source(output_code)
