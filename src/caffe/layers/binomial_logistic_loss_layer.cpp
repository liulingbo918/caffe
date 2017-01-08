#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void BinomialLogisticLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
}

template <typename Dtype>
void BinomialLogisticLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_gt = bottom[1]->cpu_data();
  const Dtype* bottom_label = bottom[2]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    int label = static_cast<int>(bottom_label[i]);
    if (label == 0) { // background, no need to include it in completeness loss
        continue;
    }
    Dtype prob = std::max(
        bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
    if (bottom_gt[i] == label) {
        loss -= log(prob);
    } else if (bottom_gt[i] == 0) {
        loss -= log(1-prob);
    } else {
        LOG(ERROR) << "Inconsistent label! "
                   << " categorical label = " << label
                   << " completeness label = " << bottom_gt[i];
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void BinomialLogisticLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_gt = bottom[1]->cpu_data();
    const Dtype* bottom_label = bottom[2]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
    const Dtype scale = - top[0]->cpu_diff()[0] / num;
    for (int i = 0; i < num; ++i) {
      int label = static_cast<int>(bottom_label[i]);
      if (label == 0) {
          continue;
      }
      Dtype prob = std::max(
          bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
      if (bottom_gt[i] == label) {
          bottom_diff[i * dim + label] = scale / prob;
      } else if (bottom_gt[i] == 0) {
          bottom_diff[i * dim + label] = scale / (prob-1);
      } else {
          LOG(ERROR) << "Inconsistent label! "
                     << "categorical label = " << label
                     << "completeness label = " << bottom_gt[i];
      }
    }
  }
}

INSTANTIATE_CLASS(BinomialLogisticLossLayer);
REGISTER_LAYER_CLASS(BinomialLogisticLoss);

}  // namespace caffe
