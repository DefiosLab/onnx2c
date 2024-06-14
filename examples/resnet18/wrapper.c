#include<stdio.h>
#include "types.h"
#include "inference.h"
void run(float* input, float* output){
  float_tensor inp = {4,{1,3,224,224},input};
  float_tensor out = {2,{1,64,112,112},output};
  inference(&inp,&out);
}
