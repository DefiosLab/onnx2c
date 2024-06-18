#ifndef TYPES_H
#define TYPES_H
#include <stdint.h>
typedef struct {
  int32_t ndim;
  int32_t shape[10];
  float *data;
} float_tensor;

typedef struct {
  int32_t ndim;
  int32_t shape[10];
  int32_t *data;
} int32_t_tensor;

typedef struct {
  int32_t ndim;
  int32_t shape[10];
  int64_t *data;
} int64_t_tensor;
#endif
