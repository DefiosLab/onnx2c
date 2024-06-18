# M5Stackで動くMnist推論モデルのサンプル

モデルは[こちら](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist)のmnist-12をop13に変更してonnx-simplifierに通したもの


## 実行
`src/`と`include/`をPlattformIO等のプロジェクトに配置して実行出来ます。

## 自分で生成するには
前処理済み入力データのnpyとonnxを用意して以下コマンドを実行してください
```
$ onnx2c --onnx mnist.onnx --input_data input.npy
```
ここで生成されたartifactsとclang_operators/のコードを使ってください。
artifacts/tensor.hにweightやoutputテンソルの定義があります。M5StackCore2ではメモリが足りなかったため、手動でConv->Reluのoutputを共有化することで動作しました。

前処理や実行のコードはexample/mnist/run.pyが参考になります。

