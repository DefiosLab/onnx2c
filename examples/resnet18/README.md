# Mnist推論モデルのサンプル

モデルは[こちら](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist)の[Resnet18-V1](https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx)です。


## C言語生成&実行
- 共有ライブラリとしてコンパイルしてpythonから実行します。
```
$ onnx2c --onnx resnet18-v1-7.onnx #artifactsが生成される
$ make 
$ python3 run.py
```

