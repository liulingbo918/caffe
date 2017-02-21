#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/util/device_alternate.hpp"


template<typename Dtype>
__global__ void permute_kernel(const size_t* dims, size_t ndim, size_t n,
                               Dtype* dst_data, size_t dst_dim, Dtype beta,
                               const Dtype* src_data, size_t src_dim, Dtype alpha){
  CUDA_KERNEL_LOOP(src_idx, n){

    size_t src_dim_idx = 0;
    size_t dst_dim_idx = 0;
    for (int i = ndim - 1, p = src_idx; i >= 0; --i){
      size_t d = dims[i];
      if (i == src_dim) src_dim_idx = p % d;
      if (i == dst_dim) dst_dim_idx = p % d;
      p /= d;
    }

    size_t dst_idx = 0;
    for (int i = 0, p = src_idx, q = n; i < ndim; ++i){

      size_t offset;
      size_t d;
      q /= dims[i];

      if (i == src_dim){
        d = dims[dst_dim];
        offset = dst_dim_idx;
      }else if(i == dst_dim){
        d = dims[src_dim];
        offset = src_dim_idx;
      }else{
        d = dims[i];
        offset = p / q;
      }
      dst_idx = dst_idx * d + offset;
      p %= q;
    }

    dst_data[dst_idx] = src_data[src_idx] * alpha + dst_data[dst_idx] * beta;
  }
}

namespace caffe {

template<typename Dtype>
void permute_dimension_gpu(const size_t *dims, size_t ndim, size_t num,
                           Dtype *dst_data, size_t dst_dim, Dtype beta,
                           const Dtype *src_data, size_t src_dim, Dtype alpha) {

  if (src_dim == dst_dim) return;

      permute_kernel  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS>>> (dims, ndim, num, dst_data, dst_dim, beta,
      src_data, src_dim, alpha);

}

template<typename Dtype>
void PermuteLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
const vector<Blob<Dtype> *> &top) {
  permute_dimension_gpu((const size_t*)dims_.gpu_data(), bottom[0]->num_axes(), bottom[0]->count(),
      top[0]->mutable_gpu_data(), second_dim_, (Dtype)0,
  bottom[0]->gpu_data(), first_dim_, (Dtype)1);
}

template<typename Dtype>
void PermuteLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {

  if (propagate_down[0]) {
    permute_dimension_gpu((const size_t*)permute_dims_.gpu_data(), bottom[0]->num_axes(), bottom[0]->count(),
        bottom[0]->mutable_gpu_diff(), second_dim_, (Dtype) 0,
    top[0]->gpu_diff(), first_dim_, (Dtype) 1);
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(PermuteLayer);

}