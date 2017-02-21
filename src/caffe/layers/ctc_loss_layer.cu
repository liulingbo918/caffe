
#ifdef WITH_CTC

#include "caffe/loss_layers.hpp"
#define CTC_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    ctcStatus_t error = condition; \
    CHECK_EQ(error, 0) << " " << ctcGetStatusString(error); \
  } while (0)

namespace caffe {


template <typename Dtype>
void CTCLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
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
        for (int i = 0; i < bottom[1]->count(); ++i) {
          label_data[i] = src_label[i];
        }

        size_t wp_size = 0;
        CTC_CHECK(get_workspace_size((const int *) label_length_.cpu_data(),
                                     (const int *) input_length_.cpu_data(),
                                     num_class, batchsize,
                                     ctc_opt_, &wp_size));
        workspace_.Resize(wp_size);
        costs_.Resize(batchsize * sizeof(float));
        CTC_CHECK(compute_ctc_loss((const float*)bottom[0]->gpu_data(),
                                   (this->phase_ == TRAIN) ? (float*)bottom[0]->mutable_gpu_diff() : NULL,
                                   (const int*)cpu_labels_.cpu_data(),
                                   (const int *) label_length_.cpu_data(),
                                   (const int *) input_length_.cpu_data(),
                                   num_class, batchsize,
                                   (float*)costs_.mutable_gpu_data(),
                                   workspace_.mutable_gpu_data(), ctc_opt_));

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
void CTCLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (sizeof(Dtype) == 4) {
        int batchsize = bottom[0]->shape(1);
        Dtype loss_weight = top[0]->cpu_diff()[0];
        Dtype alpha = Dtype(1) / Dtype(batchsize);
        // scale gradients
        caffe_gpu_scal(bottom[0]->count(), alpha * loss_weight, bottom[0]->mutable_gpu_diff());
    }else{
        NOT_IMPLEMENTED;
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(CTCLossLayer);

}  // namespace caffe

#endif