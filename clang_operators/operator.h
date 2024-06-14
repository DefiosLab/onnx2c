#ifndef OPERATOR_H
#define OPERATOR_H
#include "types.h"
#include <stdbool.h>
typedef struct {
  float epsilon;
  float momentum;
} bn_attrs;
typedef struct {
  int32_t pads[10];
  int32_t stride[10];

} conv_attrs;
typedef struct {
  int32_t pads[10];
  int32_t kernel_shape[10];
  int32_t stride[10];
} pool_attrs;
typedef struct {
  float alpha;
  float beta;
  bool transA;
  bool transB;
} gemm_attrs;
void Relu_F32(float_tensor *input, float_tensor *output);
void Conv_F32(float_tensor *input, float_tensor *weight, float_tensor *bias,
              float_tensor *output, conv_attrs *attrs, bool use_bias);
void MaxPool_F32(float_tensor *input, float_tensor *output, pool_attrs *attrs);
void Gemm_F32(float_tensor *A, float_tensor *B, float_tensor *C,
              float_tensor *output, gemm_attrs *attrs, bool use_C);
void BatchNormalization_F32(float_tensor *X, float_tensor *scale,
                            float_tensor *B, float_tensor *mean,
                            float_tensor *var, bn_attrs *attrs,
                            float_tensor *Y);
void Add_F32(float_tensor *A, float_tensor *B, float_tensor *C);
void GlobalAveragePool_F32(float_tensor *input, float_tensor *output);
#endif
