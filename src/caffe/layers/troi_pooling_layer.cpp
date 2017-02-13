// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/vision_layers.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void TROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  op_ = this->layer_param_.batch_reduction_param().reduction_param().operation();
  axis_ = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.batch_reduction_param().reduction_param().axis());
  ticks_.push_back(1);
}

template <typename Dtype>
void TROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape;
  top_shape.push_back(bottom[1]->shape()[0]);
  for (int i = axis_ + 1; i < bottom[0]->shape().size(); ++i) {
      top_shape.push_back(bottom[0]->shape(i));
      step_ = bottom[0]->count(axis_+1);
      num_ = bottom[0]->count(0, axis_);
  }
  top[0]->Reshape(top_shape);
  ticks_[0] = bottom[0]->shape(axis_);
  if (op_ == ReductionParameter_ReductionOp_TOPK) {
    vector<int> argsort_shape;
    argsort_shape.push_back(bottom[1]->shape()[0]);
    for (int i = 1; i < bottom[0]->shape().size(); ++i) {
      argsort_shape.push_back(bottom[0]->shape()[i]);
    }
    argsort_idx_.Reshape(argsort_shape);
  } else {
    argsort_idx_.Reshape(1,1,1,1);
  }
  vector<int> masked_data_shape;
  for (int i = 1; i < bottom[0]->shape().size(); ++i) {
    masked_data_shape.push_back(bottom[0]->shape()[i]);
  }
  masked_data_.Reshape(masked_data_shape);
}

template <typename Dtype>
bool comparator(const std::pair<Dtype, int>& left, const std::pair<Dtype, int>& right) {
  return left.first >= right.first;
}

template <typename Dtype>
void TROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* idx_data = argsort_idx_.mutable_cpu_data();
  Dtype* mask = masked_data_.mutable_cpu_data();
  // Number of TROIs
  int num_rois = bottom[1]->shape()[0];
  int batch_size = bottom[0]->shape()[0];
  caffe_set(top[0]->count(), Dtype(0), top_data);
  
  if (op_ != ReductionParameter_ReductionOp_TOPK) {
    for (int n = 0; n < num_rois; ++n) {
      int roi_batch_ind = bottom_rois[0];
      int roi_start = bottom_rois[1];
      int roi_end = bottom_rois[2];
      CHECK_GE(roi_batch_ind, 0);
      CHECK_LT(roi_batch_ind, batch_size);
      CHECK_GE(roi_start, 0);
      CHECK_LT(roi_end, bottom[0]->shape()[1]);
      
      int tick = ticks_[0];
      //LOG(INFO) << "roi_batch_ind = " << roi_batch_ind << ", roi_start = " << roi_start << ", roi_end = " << roi_end << "tick = " << tick;
      // Create mask from bottom_data (b, 0:C, 0:S), and set 0 on (b, 0:c_1 || c_2:end, 0:S)
      bottom_data += tick * step_ * roi_batch_ind;
      caffe_copy(bottom[0]->count()/bottom[0]->shape()[0], bottom_data, mask);
      bottom_data -= tick * step_ * roi_batch_ind;
      for (int t = 0; t < tick; ++t) {
        if (t < roi_start || t >roi_end) {
          caffe_set(step_, Dtype(0), mask);
        }
        mask += step_;
      }
      mask -= tick * step_;
      Dtype coeff = (op_ == ReductionParameter_ReductionOp_MEAN) ? Dtype(1) / Dtype(tick) : Dtype(1);
      for (int t = 0; t < tick; ++t) {
        caffe_cpu_axpby(step_, coeff, mask, Dtype(1), top_data);
      }
      top_data += step_;
      bottom_rois += 3;
    }
  } else {
    caffe_set(bottom[1]->shape()[0] * bottom[0]->count() / bottom[0]->shape()[0], Dtype(-1), idx_data);
    int k = this->layer_param_.batch_reduction_param().reduction_param().k();
    int tick = ticks_[0];
    vector<std::pair<Dtype, int> > buffer;
    buffer.resize(tick);
    for (int n = 0; n < num_rois; ++n) {
      int roi_batch_ind = bottom_rois[0];
      int roi_start = bottom_rois[1];
      int roi_end = bottom_rois[2];
      CHECK_GE(roi_batch_ind, 0);
      CHECK_LT(roi_batch_ind, batch_size);
      CHECK_GE(roi_start, 0);
      CHECK_LT(roi_end, bottom[0]->shape()[1]);
      
      //LOG(INFO) << "roi_batch_ind = " << roi_batch_ind << ", roi_start = " << roi_start << ", roi_end = " << roi_end << " tick = " << tick << " step_ = " << step_;
      
      for (int i = 0; i < step_; ++i) {
        for (int t = 0; t < roi_end-roi_start+1; ++t) {
          buffer[t] = std::make_pair(bottom_data[(t+roi_start) * step_ + i], t);
        } 
        std::sort(buffer.begin(), buffer.begin()+roi_end-roi_start+1, comparator<Dtype>);

        k = (k > roi_end - roi_start + 1) ? (roi_end - roi_start + 1) : k;
        Dtype accum = 0;
        for (int k_out = 0; k_out < k; ++k_out) {
          std::pair<Dtype, int>& p = buffer[k_out];
          CHECK_GT(p.first, -FLT_MAX);
          accum += p.first;
          idx_data[(p.second+roi_start)*step_ + i] = k_out + 1;
        }
        top_data[i] = accum / Dtype(k);
      }
      top_data += step_;
      idx_data += tick * step_;
      bottom_rois += 3;
    }
  }
}

template <typename Dtype>
void TROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {return; }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  const Dtype* idx_data = argsort_idx_.cpu_data();
  int num_rois = bottom[1]->shape()[0];
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

  if (op_ != ReductionParameter_ReductionOp_TOPK) {
    for (int n = 0; n < num_rois; ++n) {
      int tick = ticks_[0];
      Dtype coeff = (op_ == ReductionParameter_ReductionOp_MEAN) ? Dtype(1) / Dtype(tick) : Dtype(1);
      int roi_batch_ind = bottom_rois[0];
      int roi_start = bottom_rois[1];
      int roi_end = bottom_rois[2];
      bottom_diff += tick * step_ * roi_batch_ind;
      for (int t = 0; t < tick; ++t) {
        if (t >= roi_start && t <= roi_end)
          caffe_cpu_axpby(step_, coeff, top_diff, Dtype(1), bottom_diff);
        bottom_diff += step_;
        top_diff += step_;
      }
      bottom_diff -= tick * step_ * roi_batch_ind;
    }
  } else {
    int tick = ticks_[0];
    int k = this->layer_param_.batch_reduction_param().reduction_param().k();
    for (int n = 0; n < num_rois; ++n) {
      int roi_batch_ind = bottom_rois[0];
      int roi_start = bottom_rois[1];
      int roi_end = bottom_rois[2];
      k = (k > roi_end - roi_start + 1) ? (roi_end - roi_start + 1) : k;
      bottom_diff += tick * step_ * roi_batch_ind;
      for (int i = 0; i < step_; ++i) {
        Dtype diff = top_diff[i] / Dtype(k);
        for (int t = 0; t < tick; ++t) {
          if (t >= roi_start && t <= roi_end) 
            bottom_diff[t * step_ + i] += (idx_data[t * step_ + i] >= 1) ? diff : 0;
          else
            CHECK_EQ(idx_data[t * step_ + i], -1);
        }
      }
      top_diff += step_;
      bottom_diff -= tick * step_ * roi_batch_ind;
      idx_data += tick * step_;  
      bottom_rois += 3;
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(TROIPoolingLayer);
#endif

INSTANTIATE_CLASS(TROIPoolingLayer);
REGISTER_LAYER_CLASS(TROIPooling);

}  // namespace caffe
