// --------------------------------------------------------
// R-FCN
// Written by Yi Li, 2016.
// --------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/vision_layers.hpp"
#include "caffe/util/gpu_util.cuh"

using std::max;
using std::min;

namespace caffe {

  template <typename Dtype>
  __global__ void PSROIPoolingForward(
    const int nthreads,
    const Dtype* bottom_data,
    const Dtype spatial_scale,
    const int channels,
    const int length,
    const int pooled_length,
    const int output_dim,
    const int group_size,
    Dtype* top_data,
    int* mapping_channel) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, pl, ctop)
      int ctop = index % output_dim;
      int pl = (index / output_dim) % pooled_length;
      int n = index / output_dim / pooled_length;

      // [start, end) interval for spatial sampling
      int roi_batch_ind = n;
      Dtype roi_start =
        static_cast<Dtype>(0) * spatial_scale;
      Dtype roi_end =
        static_cast<Dtype>(length) * spatial_scale;

      // Force too small ROIs to be 1x1
      Dtype roi_length = max((Dtype)length, 0.1);  // avoid 0

      // Compute location at bottom
      Dtype bin_size = roi_length / static_cast<Dtype>(pooled_length);

      int start = floor(static_cast<Dtype>(pl) * bin_size
                          + roi_start);
      int end = ceil(static_cast<Dtype>(pl + 1) * bin_size
                        + roi_start);

      // Add roi offsets and clip to input boundaries
      start = min(max(start, 0), length);
      end = min(max(end, 0), length);
      bool is_empty = (end <= start);

      int gl = pl;
      int c = gl * output_dim + ctop;

      bottom_data += (roi_batch_ind * length) * channels + c;
      Dtype out_sum = 0;
      for (int l = start; l < end; ++l) {
        int bottom_index = channels * l;
        out_sum += bottom_data[bottom_index];
      }

      Dtype bin_area = end - start;
      top_data[index] = is_empty? 0. : out_sum/bin_area;
      mapping_channel[index] = c;
    }
  }

// Bottom[0] shape: N, T, K*C
// Top[0] shape: N, K, C
  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int* mapping_channel_ptr = mapping_channel_.mutable_gpu_data();
    int count = top[0]->count();
    caffe_gpu_set(count, Dtype(0), top_data);
    caffe_gpu_set(count, -1, mapping_channel_ptr);
    // NOLINT_NEXT_LINE(whitespace/operators)
    PSROIPoolingForward<Dtype> <<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >>>(count, bottom_data, spatial_scale_,
      channels_, length_, pooled_length_,
      output_dim_, group_size_,
      top_data, mapping_channel_ptr);
    CUDA_POST_KERNEL_CHECK;
  }

  template <typename Dtype>
  __global__ void PSROIPoolingBackwardAtomic(
    const int nthreads,
    const Dtype* top_diff,
    const int* mapping_channel,
    const Dtype spatial_scale,
    const int channels,
    const int length,
    const int pooled_length,
    const int output_dim,
    Dtype* bottom_diff) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, pl, ctop)
      int ctop = index % output_dim;
      int pl = (index / output_dim) % pooled_length;
      int n = index / output_dim / pooled_length;

      // [start, end) interval for spatial sampling
      int roi_batch_ind = n;
      Dtype roi_start =
        static_cast<Dtype>(0) * spatial_scale;
      Dtype roi_end =
        static_cast<Dtype>(length ) * spatial_scale;

      // Force too small ROIs to be 1x1
      Dtype roi_length = max((Dtype)length, 0.1);  // avoid 0

      // Compute w and h at bottom
      Dtype bin_size = roi_length / static_cast<Dtype>(pooled_length);

      int start = floor(static_cast<Dtype>(pl)* bin_size
        + roi_start);
      int end = ceil(static_cast<Dtype>(pl + 1) * bin_size
        + roi_start);
      // Add roi offsets and clip to input boundaries
      start = min(max(start, 0), length);
      end = min(max(end, 0), length);
      bool is_empty = (end <= start);

      // Compute c at bottom
      int c = mapping_channel[index];
      Dtype* offset_bottom_diff = bottom_diff +
        (roi_batch_ind * length * channels) + c;
      Dtype bin_area = end - start;
      Dtype diff_val = is_empty ? 0. : top_diff[index] / bin_area;
      for (int l = start; l < end; ++l) {
        int bottom_index = channels * l;
        caffe_gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
      }
    }
  }

// Bottom[0] shape: N, T, K*C
// Top[0] shape: N, K, C
  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) {
      return;
    }

    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int bottom_count = bottom[0]->count();
    const int* mapping_channel_ptr = mapping_channel_.gpu_data();
    caffe_gpu_set(bottom_count, Dtype(0), bottom_diff);
    const int count = top[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    PSROIPoolingBackwardAtomic<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, top_diff, mapping_channel_ptr,
      spatial_scale_, channels_, length_,
      pooled_length_, output_dim_, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }

  INSTANTIATE_LAYER_GPU_FUNCS(PSROIPoolingLayer);

}  // namespace caffe
