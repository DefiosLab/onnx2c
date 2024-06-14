import onnx
import numpy as np
from onnx2c.target import Layer
from onnx2c.helper import same_upper


class Gemm(Layer):
    def __init__(self, model, node, tensor_data, gen):
        super().__init__(model, node, tensor_data)
        self.gen = gen
        self.model = model

    def run(self):
        A_data = self.tensor_data[self.input_name[0]]
        B_data = self.tensor_data[self.input_name[1]]
        if len(self.input_name) >= 3:
            use_C = "true"
            c_name = "&" + self.input_name[2]
        else:
            use_C = "false"
            c_name = "NULL"

        if "alpha" in self.attrs:
            alpha = self.attrs["alpha"]
        else:
            alpha = 1.0

        if "beta" in self.attrs:
            beta = self.attrs["beta"]
        else:
            beta = 1.0

        if "transA" in self.attrs:
            transA = "true" if self.attrs["transA"] else "false"
        else:
            transA = "false"
        if "transB" in self.attrs:
            transB = "true" if self.attrs["transB"] else "false"
        else:
            transB = "false"
        assert A_data.ndim == 2, "A data must be 2-dimensional"
        assert B_data.ndim == 2, "B data must be 2-dimensional"

        attr_name = self.node.name + "_attr"
        attrs_code = f"""
gemm_attrs {attr_name} = {{  {alpha},{beta},{transA},{transB} }};
"""
        self.gen.write_header(attrs_code)

        # output tensorを設定
        p, q = A_data.shape
        r = B_data.shape[1]
        shape = f"{{{p},{r}}}"

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
            output_code = f"float {self.output_name}_data[{p*r}] = {{}};\n"
            output_code += f"float_tensor {self.output_name} = {{ 2,{shape},{self.output_name}_data }};"
            self.gen.write_header(output_code)
        else:
            output_mdf = ""
        op_func = f"Gemm_F32({input0_mdf}{self.input_name[0]},{input1_mdf}{self.input_name[1]},{c_name},{output_mdf}{self.output_name},&{attr_name},{use_C});\n"
        self.gen.write_source(op_func)
