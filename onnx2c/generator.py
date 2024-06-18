from onnx import numpy_helper, shape_inference
from onnx2c.helper import elem_type2numpy
import time
import numpy as np
from onnx2c.target.clang.layer_dict import layer_dict
from onnx2c.target.clang import Codegen


class Generator:
    def __init__(self, model):
        self.model = model
        for opset in model.opset_import:
            if opset.version != 13 and opset.domain == "":
                raise AssertionError(f"The model's opset must be 13.")

        self.tensor_data = {}
        self.layers = []
        self.time_dict = {}

        # Weightを格納
        self.tensor_data.update(
            {
                initializer.name: numpy_helper.to_array(initializer)
                for initializer in self.model.graph.initializer
            }
        )
        self.create_layer()
        self.run()

    def create_layer(self):
        self.layers = []
        self.gen = Codegen(self.model, self.tensor_data)
        self.gen.generate_initializer()  # initializerデータの生成

        for name in [input.name for input in self.model.graph.input]:  # 入力ダミーデータを追加
            if name not in self.tensor_data.keys():
                shape, dtype = self.search_tensorinfo(name)
                self.tensor_data[name] = np.zeros(shape).astype(dtype)

        for node in self.model.graph.node:
            for output_name in node.output:
                shape, dtype = self.search_tensorinfo(output_name)
                self.tensor_data[output_name] = np.zeros(shape).astype(dtype)
            self.layers.append(
                layer_dict[node.op_type](self.model, node, self.tensor_data, self.gen)
            )

    def search_tensorinfo(self, name):
        for value_info in self.model.graph.value_info:
            if value_info.name == name:
                shape_onnx = value_info.type.tensor_type.shape
                shape_tuple = tuple(
                    dim.dim_value if dim.HasField("dim_value") else None
                    for dim in shape_onnx.dim
                )
                dtype = elem_type2numpy(value_info.type.tensor_type.elem_type)
                break
        else:
            for model_output in self.model.graph.output:
                if model_output.name == name:
                    shape_onnx = model_output.type.tensor_type.shape
                    shape_tuple = tuple(
                        dim.dim_value if dim.HasField("dim_value") else None
                        for dim in shape_onnx.dim
                    )
                    dtype = elem_type2numpy(model_output.type.tensor_type.elem_type)
            for model_input in self.model.graph.input:
                if model_input.name == name:
                    shape_onnx = model_input.type.tensor_type.shape
                    shape_tuple = tuple(
                        dim.dim_value if dim.HasField("dim_value") else None
                        for dim in shape_onnx.dim
                    )
                    dtype = elem_type2numpy(model_input.type.tensor_type.elem_type)
        return shape_tuple, dtype

    def run(self):
        for layer in self.layers:
            layer.run()
        self.gen.write_source("}\n")
        self.gen.close()

    def generate_input(self, input_tensor):
        self.gen.write_input(input_tensor)
