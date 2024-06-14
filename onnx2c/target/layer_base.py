from onnx import numpy_helper
import onnx


class Layer:
    def __init__(self, model, node, tensor_data):
        self.tensor_data = tensor_data
        self.node = node
        self.input_name = node.input
        self.attrs = {
            attr.name: onnx.helper.get_attribute_value(attr)
            for attr in self.node.attribute
        }
        self.output_name = self.node.output[0]
        self.init_names = {tensor.name for tensor in model.graph.initializer}
