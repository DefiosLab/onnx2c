#include<stdio.h>
#include "types.h"
#include "inference.h"
void run(float* input, float* output){
  float_tensor inp = {4,{1,1,28,28},input};
  float_tensor out = {2,{1,10},output};
  inference(&inp,&out);
}
