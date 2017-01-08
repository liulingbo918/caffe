#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void PerClassAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void PerClassAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  //vector<int> top_shape(2, 1);
  //const int num_labels = bottom[0]->shape(label_axis_);
  //top_shape[1] = num_labels-1;
  //top[0]->Reshape(top_shape);
  vector<int> top_shape(0);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void PerClassAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_gt = bottom[1]->cpu_data();
  const Dtype* bottom_label = bottom[2]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  //vector<Dtype> accuracy(num_labels, 0);
  //vector<int> count(num_labels, 0);
  Dtype accuracy = 0;
  int count = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);
      if ((bottom_data[i * dim + label_value * inner_num_ + j] > 0.5 
          && bottom_gt[i * inner_num_ + j] == label_value) 
          || (bottom_data[i * dim + label_value * inner_num_ + j] < 0.5 
          && bottom_gt[i * inner_num_ + j] == 0))  {
          //++accuracy[label_value-1];
          ++accuracy;
      }
      //++count[label_value-1];
      ++count;
    }
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  //for (int i = 0; i < num_labels-1; ++i) {
  //    top[0]->mutable_cpu_data()[i] = accuracy[i] / count[i];
  //    // Accuracy layer should not be used as a loss function.
  //}
  top[0]->mutable_cpu_data()[0] = accuracy / count;
}

INSTANTIATE_CLASS(PerClassAccuracyLayer);
REGISTER_LAYER_CLASS(PerClassAccuracy);

}  // namespace caffe
