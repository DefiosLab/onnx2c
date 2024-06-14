# Mnist推論モデルのサンプル

モデルは(こちら)[https://github.com/onnx/models/tree/main/validated/vision/classification/mnist]のmnist-12をop13に変更してonnx-simplifierに通したもの


## C言語生成&実行
- 共有ライブラリとしてコンパイルしてpythonから実行します。
```
$ onnx2c --onnx mnist.onnx
$ make
$ python3 run.py
```

