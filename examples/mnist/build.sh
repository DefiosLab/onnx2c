gcc -fPIC -std=c99 -shared -I../../clang_operators/ -I./artifacts/  numpy_run.c artifacts/inference.c ../../clang_operators/operator.c -o mnist.so
