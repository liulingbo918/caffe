#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template<typename Dtype>
void PermuteLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  CHECK_EQ(this->layer_param_.permute_param().axis_size(), 2);
  first_dim_ = this->layer_param_.permute_param().axis(0);
  second_dim_ = this->layer_param_.permute_param().axis(1);

}

template<typename Dtype>
void PermuteLayer<Dtype>::Reshape(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  dims_.Resize(bottom[0]->num_axes() * sizeof(size_t));
  permute_dims_.Resize(bottom[0]->num_axes() * sizeof(size_t));
  size_t* dim_data = (size_t*)dims_.mutable_cpu_data();

  for(int i = 0; i < bottom[0]->num_axes(); ++i){
    dim_data[i] = bottom[0]->shape(i);
  }

  vector<int> permute_shape(bottom[0]->shape());
  int tmp = permute_shape[first_dim_];
  permute_shape[first_dim_] = permute_shape[second_dim_];
  permute_shape[second_dim_] = tmp;

  size_t* permute_dim_data = (size_t*)permute_dims_.mutable_cpu_data();
  for (int i = 0; i < bottom[0]->num_axes(); ++i){
    permute_dim_data[i] = permute_shape[i];
  }
  top[0]->Reshape(permute_shape);
}

template<typename Dtype>
void permute_dimension_cpu(const size_t* dims, size_t ndim, size_t num,
                           Dtype* dst_data, size_t dst_dim, Dtype beta,
                           const Dtype* src_data, size_t src_dim, Dtype alpha){

  if (dst_dim == src_dim) return;


  for (size_t src_idx = 0; src_idx < num; ++src_idx){
    size_t src_dim_idx = 0;
    size_t dst_dim_idx = 0;
    for (int i = ndim - 1, p = src_idx; i >= 0; --i){
      size_t d = dims[i];
      if (i == src_dim) src_dim_idx = p % d;
      if (i == dst_dim) dst_dim_idx = p % d;
      p /= d;

    }
    size_t dst_idx = 0;
    for (int i = 0, p = src_idx, q = num; i < ndim; ++i){

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

template<typename Dtype>
void PermuteLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  permute_dimension_cpu((const size_t*)dims_.cpu_data(), bottom[0]->num_axes(), bottom[0]->count(),
                        top[0]->mutable_cpu_data(), second_dim_, (Dtype)0,
                        bottom[0]->cpu_data(), first_dim_, (Dtype)1);
}

template<typename Dtype>
void PermuteLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                       const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {

  if (propagate_down[0]) {
    permute_dimension_cpu((const size_t*)permute_dims_.cpu_data(), bottom[0]->num_axes(), bottom[0]->count(),
                          bottom[0]->mutable_cpu_diff(), second_dim_, (Dtype) 0,
                          top[0]->cpu_diff(), first_dim_, (Dtype) 1);
  }

}

#ifdef CPU_ONLY
STUB_GPU(PermuteLayer);
#endif

INSTANTIATE_CLASS(PermuteLayer);
REGISTER_LAYER_CLASS(Permute);
}