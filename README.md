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

## example
`example/mnist`のREADMEを参考

