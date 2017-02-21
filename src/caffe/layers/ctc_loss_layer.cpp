#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/loss_layers.hpp"

#ifdef WITH_CTC

#define CTC_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    ctcStatus_t error = condition; \
    CHECK_EQ(error, 0) << " " << ctcGetStatusString(error); \
  } while (0)

namespace caffe {

template <typename Dtype>
void CTCLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  ctc_opt_.blank_label = 0;
}

template <typename Dtype>
void CTCLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//  LossLayer<Dtype>::Reshape(bottom, top);

  CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(0))<<" input batchsizes mismatch";
  CHECK_GE(bottom[0]->shape(0), bottom[1]->shape(1));

  if (Caffe::mode() == Caffe::GPU){
    ctc_opt_.loc = CTC_GPU;
    ctc_opt_.stream = 0;
  }else{
    ctc_opt_.loc = CTC_CPU;
    ctc_opt_.num_threads = 10;
  }

  input_length_.Resize(bottom[0]->shape(1) * sizeof(int));
  label_length_.Resize(bottom[1]->shape(0) * sizeof(int));

  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);

}

template <typename Dtype>
void CTCLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (sizeof(Dtype) == 4) {

    int num_class = bottom[0]->shape(2);
    int batchsize = bottom[0]->shape(1);

    int *ll_data = (int *) label_length_.mutable_cpu_data();
    int *il_data = (int *) input_length_.mutable_cpu_data();
    for (int n = 0; n < batchsize; ++n) {
      il_data[n] = bottom[0]->shape(0);
      ll_data[n] = bottom[1]->shape(1);
    }

    cpu_labels_.Resize(bottom[1]->count() * sizeof(int));
    int* label_data = (int*)cpu_labels_.mutable_cpu_data();
    const Dtype* src_label = bottom[1]->cpu_data();
    for (int i = 0; i < bottom[1]->count(); ++i) label_data[i] = src_label[i];

    size_t wp_size = 0;
    CTC_CHECK(get_workspace_size((const int *) label_length_.cpu_data(),
                                 (const int *) input_length_.cpu_data(),
                                 num_class, batchsize,
                                 ctc_opt_, &wp_size));
    workspace_.Resize(wp_size);
    costs_.Resize(batchsize * sizeof(float));
    CTC_CHECK(compute_ctc_loss((const float*)bottom[0]->cpu_data(),
                               (this->phase_ == TRAIN) ? (float*)bottom[0]->mutable_cpu_diff() : NULL,
                               (const int*)cpu_labels_.cpu_data(),
                               (const int *) label_length_.cpu_data(),
                               (const int *) input_length_.cpu_data(),
                               num_class, batchsize,
                               (float*)costs_.mutable_cpu_data(),
                               workspace_.mutable_cpu_data(), ctc_opt_));

    const float* loss_data = (float*)costs_.cpu_data();
    Dtype* loss_out = top[0]->mutable_cpu_data();
    *loss_out = 0;
    for (int n = 0; n < batchsize; ++n){
      *loss_out += loss_data[n] / Dtype(batchsize);
    }

  }else{
    NOT_IMPLEMENTED;
  }
}

template <typename Dtype>
void CTCLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (sizeof(Dtype) == 4) {
    int batchsize = bottom[0]->shape(1);
    Dtype loss_weight = top[0]->cpu_diff()[0];
    Dtype alpha = Dtype(1) / Dtype(batchsize);
    // scale gradients
    caffe_scal(bottom[0]->count(), alpha * loss_weight, bottom[0]->mutable_cpu_diff());
  }else{
    NOT_IMPLEMENTED;
  }
}

#ifdef CPU_ONLY
STUB_GPU(CTCLossLayer);
#endif

INSTANTIATE_CLASS(CTCLossLayer);
REGISTER_LAYER_CLASS(CTCLoss);

}  // namespace caffe
#endif