#include "inference.h"
#include "tensor.h"

#include <stdbool.h>
#include <string.h>
void inference(float_tensor *_Input3,float_tensor *_Plus214_Output_0){
Conv_F32(_Input3,&_Parameter5,&__v_23,&_Convolution28_Output_0,&_Convolution28_attr,true);
Relu_F32(&_Convolution28_Output_0,&_ReLU32_Output_0);
MaxPool_F32(&_ReLU32_Output_0,&_Pooling66_Output_0,&_Pooling66_attr);
Conv_F32(&_Pooling66_Output_0,&_Parameter87,&__v_24,&_Convolution110_Output_0,&_Convolution110_attr,true);
Relu_F32(&_Convolution110_Output_0,&_ReLU114_Output_0);
MaxPool_F32(&_ReLU114_Output_0,&_Pooling160_Output_0,&_Pooling160_attr);
Gemm_F32(&_Pooling160_Output_0_reshape0,&_Parameter193_reshape1,&_Parameter194,_Plus214_Output_0,&__attr,true);
}
