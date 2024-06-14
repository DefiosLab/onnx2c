#include "inference.h"
#include "tensor.h"

#include <stdbool.h>
#include <string.h>
void inference(float_tensor *Input3,float_tensor *Plus214_Output_0){
Conv_F32(Input3,&Parameter5,&_v_23,&Convolution28_Output_0,&Convolution28_attr,true);
Relu_F32(&Convolution28_Output_0,&ReLU32_Output_0);
MaxPool_F32(&ReLU32_Output_0,&Pooling66_Output_0,&Pooling66_attr);
Conv_F32(&Pooling66_Output_0,&Parameter87,&_v_24,&Convolution110_Output_0,&Convolution110_attr,true);
Relu_F32(&Convolution110_Output_0,&ReLU114_Output_0);
MaxPool_F32(&ReLU114_Output_0,&Pooling160_Output_0,&Pooling160_attr);
Gemm_F32(&Pooling160_Output_0_reshape0,&Parameter193_reshape1,&Parameter194,Plus214_Output_0,&_attr,true);
}
