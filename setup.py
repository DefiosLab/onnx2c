from setuptools import setup, find_packages

setup(
    name="onnx2c",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "onnx2c=onnx2c.onnx2c:main",
        ],
    },
    install_requires=["numpy", "onnxruntime", "scipy", "onnx", "onnx-simplifier"],
)
