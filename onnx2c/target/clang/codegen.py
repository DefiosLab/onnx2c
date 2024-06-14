import numpy as np
from onnx2c.helper import elem_type2numpy
import os


class Codegen:
    def __init__(self, model, tensor_data):
        self.model = model
        self.tensor_data = tensor_data
        self.artifacts_dir = "artifacts"
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.header = open(os.path.join(self.artifacts_dir, "tensor.h"), "w")
        self.source = open(os.path.join(self.artifacts_dir, "inference.c"), "w")
        self.initializer_names = [
            initializer.name for initializer in self.model.graph.initializer
        ]
        self.write_source(
            '#include "inference.h"\n#include "tensor.h"\n\n#include <stdbool.h>\n#include <string.h>\n'
        )
        self.input_names = [
            input.name
            for input in model.graph.input
            if input.name not in self.initializer_names
        ]

        runner_code = "void inference("
        for input in model.graph.input:
            input_name = input.name
            if input_name in self.input_names:
                input_type = self.dtype2ctype(
                    elem_type2numpy(input.type.tensor_type.elem_type)
                )
                runner_code += f"{input_type}_tensor *{input_name},"
        for output in model.graph.output:
            output_name = output.name
            output_type = self.dtype2ctype(
                elem_type2numpy(output.type.tensor_type.elem_type)
            )
            runner_code += f"{output_type}_tensor *{output_name},"
        runner_code = runner_code[:-1] + ")"

        inference_header = open(os.path.join(self.artifacts_dir, "inference.h"), "w")
        inference_header.write(
            f"""#ifndef INFERENCE_H
#define INFERENCE_H
#include "types.h"
#include "operator.h"
{runner_code+';'}
#endif
"""
        )
        inference_header.close()

        runner_code += "{\n"
        self.write_source(runner_code)

        self.write_header("#include<stdint.h>\n")

    def check_input(self, name):
        return name in self.input_names

    def generate_input_arg(self, name):
        if self.check_input(name):
            return name
        else:
            return "&" + name

    def close(self):
        self.source.close()
        self.header.close()

    def dtype2ctype(self, dtype):
        if dtype == np.float32:
            return "float"
        elif dtype == np.int64:
            return "int64_t"
        elif dtype == np.int32:
            return "int32_t"
        else:
            raise ("Unsupports numpy type")

    def array2text(self, array, length):
        text = "{"
        for i in range(length):
            text += f"{array[i]},"
        text = text[:-1]
        text += "}"
        return text

    def generate_initializer(self):
        text = ""
        for key, value in self.tensor_data.items():
            ndim = value.ndim
            ctype = self.dtype2ctype(value.dtype)
            shape = self.array2text(value.shape, ndim)
            value = value.reshape(-1)
            array_len = value.shape[0]
            text += f"static {ctype} {key}_data[{array_len}] = "

            text += self.array2text(value, array_len) + ";\n"
            text += f"""
            {ctype}_tensor {key} = {{ {ndim},{shape},{key}_data }};
            """
        self.write_header(text)

    def write_source(self, text):
        self.source.write(text)

    def write_header(self, text):
        self.header.write(text)
