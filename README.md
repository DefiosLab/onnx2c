# onnx2c

onnxをC99でコンパイル可能なC言語に出力します

## install
```
python3 setup.py install
```

## 使い方
以下のコマンドを実行すると`artifacts`にコードが生成されます。
```
$ onnx2c --onnx <onnx path>
```
`artifacts/`に生成されたコードと`clang_operators/`をリンクしてコンパイルしてください

デバッグ用に入力データもヘッダーに含めたい場合は以下のコマンドが使えます
```
$ onnx2c --onnx <onnx path> --input_data <input tensor npy>
```
npyファイルをartifacts/input.hにダンプします。現状float32形式のみの対応になっています。

## example
`example/mnist`のREADMEを参考

