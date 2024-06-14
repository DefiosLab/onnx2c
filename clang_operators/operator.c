#include <stdio.h>
#include <stdlib.h>
#include "operator.h"
#include "types.h"
#include <stdbool.h>
#include <math.h>

#define ASSERT(expr, message)						\
  if (!(expr)) {							\
    fprintf(stderr, "Assertion failed: %s, message: %s, file: %s, line: %d\n", #expr, message, __FILE__, __LINE__); \
    exit(EXIT_FAILURE);							\
  }

void Relu_F32(float_tensor *input, float_tensor *output){
  ASSERT(input->ndim ==4, "Relu_F32 supports only 4 dimensions");
  int32_t batch = input->shape[0];
  int32_t channels = input->shape[1];
  int32_t height = input->shape[2];
  int32_t width = input->shape[3];
  int32_t img_size = height * width;
  for(int c=0;c<channels;c++){
    for(int i=0;i<height;i++){
      for(int j=0;j<width;j++){
	
	output->data[c*img_size + i*width+j] = input->data[c*img_size + i*width+j] > 0? input->data[c*img_size + i*width+j] : 0;
      }
    }
  }
}
void GlobalAveragePool_F32(float_tensor *input, float_tensor *output){
  ASSERT(input->ndim ==4, "MaxPool_F32 supports only 4 dimensions");

  int32_t in_ch = input->shape[1];
  int32_t in_h = input->shape[2];
  int32_t in_w = input->shape[3];
  
  for(int ich = 0;ich<in_ch;ich++){
    float sum=0;
    for(int i= 0;i<in_h;i++){
      for(int j= 0 ;j<in_w;j++){
	int32_t idx=ich * in_h * in_w + i * in_w + j;
	sum += input->data[idx];
      }
    }
    output->data[ich]=sum / (in_h * in_w);
  }
}


void Add_F32(float_tensor *A, float_tensor *B, float_tensor *C){
  ASSERT(A->ndim ==B->ndim, "The dimensions of A and B must match.");
  ASSERT(A->ndim == 4, "Add supports only 4 dimensions");
  int32_t batch = A->shape[0];
  int32_t channels = A->shape[1];
  int32_t height = A->shape[2];
  int32_t width = A->shape[3];
  int32_t img_size = height * width;
  for(int c=0;c<channels;c++){
    for(int i=0;i<height;i++){
      for(int j=0;j<width;j++){
	C->data[c*img_size + i*width+j] = A->data[c*img_size + i*width+j] + B->data[c*img_size + i*width+j];
      }
    }
  }
}



void MaxPool_F32(float_tensor *input, float_tensor *output, pool_attrs *attrs){
  ASSERT(input->ndim ==4, "MaxPool_F32 supports only 4 dimensions");
  int32_t begin_i = attrs->pads[0];
  int32_t end_i = attrs->pads[1];
  int32_t begin_j = attrs->pads[2];
  int32_t end_j = attrs->pads[3];

  int32_t in_ch = input->shape[1];
  int32_t in_h = input->shape[2];
  int32_t in_w = input->shape[3];
  
  int32_t out_ch = output->shape[1];
  int32_t out_h = output->shape[2];
  int32_t out_w = output->shape[3];
  int32_t ker_h = attrs->kernel_shape[0];
  int32_t ker_w = attrs->kernel_shape[1];  
  for(int ich = 0;ich<in_ch;ich++){
    for(int i= 0;i<out_h;i++){
      for(int j= 0 ;j<out_w;j++){	  
	int32_t out_idx = ich * out_h * out_w + i * out_w + j;
	float mx = -INFINITY;
	for(int ki=0;ki<ker_h;ki++){
	  for(int kj=0;kj<ker_w;kj++){
	    int32_t osi = (i * attrs->stride[0]) + ki - begin_i;
	    int32_t osj = (j * attrs->stride[1]) + kj - begin_j;
	    if(osi >= 0 && osi < in_h && osj >= 0 && osj < in_w){
	      int32_t in_idx = ich * in_h * in_w + osi * in_w + osj;
	      mx = fmax(mx,input->data[in_idx]);
	    }
	  }
	}
	output->data[out_idx]=mx;
      }
    }
  }
}

void BatchNormalization_F32(float_tensor *X, float_tensor *scale, float_tensor *B, float_tensor *mean, float_tensor *var, bn_attrs *attrs, float_tensor *Y){
  ASSERT(X->ndim == 4, "BatchNormalization supports only 4 dimensions");
  float epsilon = attrs->epsilon;
  float momentum = attrs->momentum;
  int32_t in_ch = X->shape[1];
  int32_t in_h = X->shape[2];
  int32_t in_w = X->shape[3];
  
  for(int c = 0;c < in_ch;c++){

    /* //calculate mean */
    /* float sum = 0; */
    /* for(int i = 0;i<in_h;i++){ */
    /*   for(int j=0;j<in_w;j++){ */
    /* 	int32_t idx = c * in_h * in_w + i*in_w+j; */
    /* 	sum+=X->data[idx]; */
    /*   } */
    /* } */
    /* float saved_mean = sum / (in_h*in_w); */

    /* //calculate var */
    /* sum = 0; */
    /* for(int i = 0;i<in_h;i++){ */
    /*   for(int j=0;j<in_w;j++){ */
    /* 	int32_t idx = c * in_h * in_w + i*in_w+j; */
    /* 	sum+=(X->data[idx] - saved_mean) * (X->data[idx] - saved_mean); */
    /*   } */
    /* } */
    /* float saved_var = sum / (in_h * in_w); */
    
    
    /* float output_mean = mean->data[c] * momentum + saved_mean * (1 - momentum); */
    /* float output_var = var->data[c] * momentum + saved_var * (1 - momentum); */
    float output_mean = mean->data[c];
    float output_var = var->data[c];
    for(int i = 0;i<in_h;i++){
      for(int j=0;j<in_w;j++){
	int32_t idx = c * in_h * in_w + i*in_w+j;
	Y->data[idx] = scale->data[c] * (X->data[idx] - output_mean) / sqrt(output_var + epsilon) + B->data[c];
      }
    }
  }
}
  
void Gemm_F32(float_tensor *A, float_tensor *B, float_tensor *C, float_tensor *output,gemm_attrs *attrs, bool use_C){
  int32_t p = A->shape[0];
  int32_t q = A->shape[1];
  int32_t r = output->shape[1];
  float alpha = attrs->alpha;
  float beta = attrs->beta;
  bool transA = attrs->transA;
  bool transB = attrs->transB;
  
  ASSERT(!transA, "Gemm not supports transposeA");
  ASSERT(A->ndim == 2 && B->ndim == 2, "Gemm supports only 2 dimensions");
  for(int i = 0;i<p;i++){
    for(int j = 0;j<r;j++){
      output->data[i * r + j] = 0.0f;
    }
  }
  for(int i=0;i<p;i++){
    for(int j=0;j<r;j++){
      for(int k=0;k<q;k++){
	float b_data;
	if(transB){
	  b_data = B->data[j * q + k];
	}else{
	  b_data = B->data[k * r + j];
	}
	output->data[i * r + j] += A->data[i * q + k] * b_data * alpha;
      }
    }
  }

  if(use_C){
    for(int i=0;i<p;i++){
      for(int j=0;j<r;j++){
	output->data[i * r + j]+=C->data[i * r + j] * beta;
      }
    }
  }
}


void Conv_F32(float_tensor *input, float_tensor *weight, float_tensor *bias, float_tensor *output, conv_attrs *attrs, bool use_bias){
  ASSERT(input->ndim ==4, "Conv_F32 supports only 4 dimensions");
  int32_t begin_i = attrs->pads[0];
  int32_t end_i = attrs->pads[1];
  int32_t begin_j = attrs->pads[2];
  int32_t end_j = attrs->pads[3];

  int32_t in_ch = input->shape[1];
  int32_t in_h = input->shape[2];
  int32_t in_w = input->shape[3];

  int32_t out_ch = output->shape[1];
  int32_t out_h = output->shape[2];
  int32_t out_w = output->shape[3];    

  int32_t ker_h = weight->shape[2];
  int32_t ker_w = weight->shape[3];

  for(int och = 0;och<out_ch;och++){
    for(int i = 0;i< out_h;i++){
      for(int j=0;j<out_w;j++){
	int32_t idx = och * out_h * out_w + i * out_w +j;
	output->data[idx] = 0.0f;
      }
    }
  }
  
  
  for(int och = 0;och<out_ch;och++){
    for(int ich = 0;ich<in_ch;ich++){
      for(int i= 0;i<out_h;i++){
	for(int j= 0 ;j<out_w;j++){	  
	  int32_t out_idx = och * out_h * out_w + i * out_w + j;
	  for(int ki=0;ki<ker_h;ki++){
	    for(int kj=0;kj<ker_w;kj++){
	      int32_t osi = (i * attrs->stride[0]) + ki - begin_i;
	      int32_t osj = (j * attrs->stride[1]) + kj - begin_j;
	      if(osi >= 0 && osi < in_h && osj >= 0 && osj < in_w){
		int32_t in_idx = ich * in_h * in_w + osi * in_w + osj;
		int32_t w_idx = och * (in_ch * ker_h * ker_w) + ich * ker_h * ker_w + ki * ker_w + kj;
		output->data[out_idx] += weight->data[w_idx] * input->data[in_idx];
	      }
	    }
	  }
	}
      }
    }
  }

  if(use_bias){
    for(int och = 0;och<out_ch;och++){
      for(int i = 0;i< out_h;i++){
	for(int j=0;j<out_w;j++){
	  int32_t idx = och * out_h * out_w + i * out_w +j;
	  output->data[idx]+=bias->data[och];
	}
      }
    }
  }
  
}
