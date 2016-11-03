#include <opencv2/highgui/highgui_c.h>
#include <stdint.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


#ifdef USE_MPI
#include "mpi.h"
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
#endif

namespace caffe{
template <typename Dtype>
VideoWindowDataLayer<Dtype>:: ~VideoWindowDataLayer<Dtype>() {
    this->JoinPrefetchThread();
}

template <typename Dtype>
int VideoWindowDataLayer<Dtype>::ExactNumTopBlobs() const{
    return 2; // video window, label
}

template <typename Dtype>
void VideoWindowDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
    // LayerSetUp runs through the window file creates foreground and background windows
    // Foreground/background is decided by overlap threshold

    // window file format
    // repeated:
    //      # video_index
    //      video_path
    //      duration (time)
    //      fps (frame/sec)
    //      num_windows
    //      class_index overlap start_time end_time

    LOG(INFO) << "VideoWindowDataLayer: "<<std::endl
        << " foreground overlap threshold: "
        << this->layer_param_.video_window_data_param().fg_threshold() << std::endl
        << " background overlap threshold: "
        << this->layer_param_.video_window_data_param().bg_threshold() << std::endl
        << " foreground sampling fraction: "
        << this->layer_param_.video_window_data_param().fg_fraction() <<std::endl
        << " root_folder: "
        << this->layer_param_.video_window_data_param().root_folder() << std::endl
        << " name_pattern: "
        << this->layer_param_.video_window_data_param().name_pattern() << std::endl;

    root_folder_ = this->layer_param_.video_window_data_param().root_folder();
    name_pattern_ = this->layer_param_.video_window_data_param().name_pattern();

    fg_thresh_ = this->layer_param_.video_window_data_param().fg_threshold();
    bg_thresh_ = this->layer_param_.video_window_data_param().bg_threshold();
    fg_fraction_ = this->layer_param_.video_window_data_param().fg_fraction();

    float min_bg_coverage = this->layer_param_.video_window_data_param().min_bg_coverage();

    if (this->layer_param_.video_window_data_param().mode() == VideoWindowDataParameter_Mode_CLS){
        fg_fraction_ = 1.0;
        this->layer_param_.mutable_video_window_data_param()->set_boundary_frame(false);
    }

    if (this->layer_param_.video_window_data_param().mode() == VideoWindowDataParameter_Mode_PROP){
        this->layer_param_.mutable_video_window_data_param()->set_merge_positive(true);
    }

    const int gt_label_offset =
            (this->layer_param_.video_window_data_param().mode() == VideoWindowDataParameter_Mode_CLS)?
            -1 : 0;


    prefetch_rng_.reset(new Caffe::RNG(caffe_rng_rand()));

    std::ifstream infile(this->layer_param_.video_window_data_param().source().c_str());
    CHECK(infile.good()) << "Failed to open window file"
        << this->layer_param_.video_window_data_param().source() << std::endl;

    map<int, int> label_hist;
    label_hist.insert(std::make_pair(0, 0));

    int video_index;
    string hashtag;

    int video_cnt = 0;
    while(infile >> hashtag >> video_index){
        CHECK_EQ(hashtag, "#");
        video_cnt++;

        CHECK_EQ(video_cnt, video_index);

        string video_path;
        infile >> video_path;

        video_path = root_folder_ + video_path;
        vector<float> video_info(2);
        infile >> video_info[0] >> video_info[1];
        video_database_.push_back(std::make_pair(video_path, video_info));

        int num_gt_windows;
        infile >> num_gt_windows;
        vector<vector<float> > video_gt_windows;
        for (int i = 0; i < num_gt_windows; ++i){
            int label;
            float start_time, end_time;
            infile >> label >> start_time >> end_time;
            vector<float> window(VideoWindowDataLayer::NUM);
            window[VideoWindowDataLayer::VIDEO_INDEX] = video_index;
            window[VideoWindowDataLayer::LABEL] = label + gt_label_offset;
            window[VideoWindowDataLayer::OVERLAP] = 1.0;
            window[VideoWindowDataLayer::START] = start_time;
            window[VideoWindowDataLayer::END] = end_time;
            if (end_time - start_time > this->layer_param_.video_window_data_param().num_segments())
                video_gt_windows.push_back(window);
        }
        gt_windows_.push_back(video_gt_windows);
        flat_gt_windows_.insert(flat_gt_windows_.end(), video_gt_windows.begin(), video_gt_windows.end());

        int num_windows;
        infile >> num_windows;


        for (int i = 0; i < num_windows; ++i){
            int label;
            float overlap, overlap_self, start_time, end_time;
            infile >> label >> overlap >> overlap_self>> start_time >> end_time;

            vector<float> window(VideoWindowDataLayer::NUM);
            window[VideoWindowDataLayer::VIDEO_INDEX] = video_index;
            window[VideoWindowDataLayer::LABEL] = label;
            window[VideoWindowDataLayer::OVERLAP] = overlap;
            window[VideoWindowDataLayer::START] = start_time;
            window[VideoWindowDataLayer::END] = end_time;

            float coverage = (end_time - start_time) / video_info[0];

            float real_end_time = std::min(video_info[0], end_time);
            if (real_end_time - start_time <= this->layer_param_.video_window_data_param().snippet_length())
                continue;
            if (overlap >= fg_thresh_){
                int chk_label = window[VideoWindowDataLayer::LABEL];
                CHECK_GT(chk_label, 0);
                fg_windows_.push_back(window);
                label_hist.insert(std::make_pair(label, 0));
                label_hist[label]++;
            }else if (overlap_self <= bg_thresh_ && coverage >= min_bg_coverage){
                // background window
                window[VideoWindowDataLayer::LABEL] = 0;
                window[VideoWindowDataLayer::OVERLAP] = overlap_self;
                bg_windows_.push_back(window);
                label_hist[0]++;
            }
        }

        if (video_index % 1000 == 0){
            LOG(INFO) << " num: "<<video_index<<" "<<video_path
                <<" duration: "
                << video_info[0]
                <<" fps: "
                << video_info[1]
                << " #windows: "<< num_windows;
        }
    }

    CHECK_GT(video_cnt, 0)<<" window file is empty";

    LOG(INFO)<<" Number of videos: "<<video_cnt;

    for (map<int, int>::iterator it = label_hist.begin(); it != label_hist.end(); ++it){
        LOG(INFO) <<" class "<<it->first
            << " has "<<label_hist[it->first]<<" samples";
    }

    LOG(INFO) << "Segment sampling mode: "
        <<this->layer_param_.video_window_data_param().segment_mode();

    // compute shapes
    const int crop_size = this->transform_param_.crop_size();
    CHECK_GT(crop_size, 0);
    const int batch_size = this->layer_param_.video_window_data_param().batch_size();
    const int num_segments = this->layer_param_.video_window_data_param().num_segments();

    const int snippet_len = this->layer_param_.video_window_data_param().snippet_length();
    const int channels =
            ((this->layer_param_.video_window_data_param().modality()== VideoWindowDataParameter_Modality_FLOW) ? 2 : 3)
            * snippet_len
            * (num_segments
               + 2 * (this->layer_param_.video_window_data_param().boundary_frame()));
    top[0]->Reshape(batch_size, channels, crop_size, crop_size);
    this->prefetch_data_.Reshape(batch_size, channels, crop_size, crop_size);
    this->transformed_data_.Reshape(1, channels, crop_size, crop_size);

    LOG(INFO) << "output data size: " << top[0]->num() << ","
    << top[0]->channels() << "," << top[0]->height() << ","
    << top[0]->width();
    // label
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    this->prefetch_label_.Reshape(label_shape);
}


template <typename Dtype>
unsigned int VideoWindowDataLayer<Dtype>::PrefetchRand() {
    CHECK(prefetch_rng_);
    caffe::rng_t* prefetch_rng =
            static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    return (*prefetch_rng)();
}


template <typename Dtype>
vector<int> VideoWindowDataLayer<Dtype>::SampleSegments(
        const int start_frame, const int end_frame, const int context_pad,
        const int total_frame, const int num_segments, const int snippet_len,
        const bool random_shift, const bool boundary_frame){

    int real_start_frame = std::max(0, start_frame);
    int real_end_frame = std::min(end_frame, total_frame);

    int duration = real_end_frame - real_start_frame;

    CHECK_GT(duration, 0)<<real_end_frame<<" "<<real_start_frame<<" "<<total_frame;

    int average_duration = duration / num_segments;

    vector<int> offsets;

    if (boundary_frame) {
        int boundary_offset = PrefetchRand() % (average_duration - snippet_len + 1) - average_duration / 2;
        int left_boundary_index = std::max(0, real_start_frame + boundary_offset);
        offsets.push_back(left_boundary_index);
    }
    for (int i = 0; i < num_segments; i++){
        if (random_shift){
            if (average_duration >= snippet_len){
                const unsigned int rand_idx = PrefetchRand();
                int offset = rand_idx % (average_duration - snippet_len + 1);
                CHECK_GE(offset, 0);
                offsets.push_back(offset + i*average_duration + real_start_frame);
            } else if (duration >= snippet_len) {
                // randomly sample snippet from remaining slots if cannot uniformly span them
                const unsigned int rand_idx = PrefetchRand();
                int offset = rand_idx % (duration - snippet_len + 1);
                CHECK_GE(offset, 0);
                offsets.push_back(real_start_frame + offset);
            } else {
                LOG(FATAL)<<"Insufficient frames to build snippet, need :"<<snippet_len<<" got: "<<duration<<" window: "
                        <<start_frame<<" -> "<<end_frame<<", duration: "<<total_frame;
            }
        }else{
            if (average_duration >= snippet_len) {
                offsets.push_back((average_duration - snippet_len) / 2 + i * average_duration + real_start_frame);
            }
            else if (duration >= snippet_len)
                offsets.push_back(real_start_frame);
            else
                LOG(FATAL)<<"Insufficient frames to build snippet, need :"<<snippet_len<<" got: "<<duration;
        }
        //LOG(INFO)<<offsets.back()<<" "<<average_duration<<" "<<snippet_len;
    }
    if (boundary_frame){
        int boundary_offset = PrefetchRand() % (average_duration - snippet_len + 1) - average_duration / 2;
        int right_boundary_index = std::min(real_end_frame + boundary_offset, total_frame);
        offsets.push_back(right_boundary_index - snippet_len);
    }
    return offsets;
}

template <typename Dtype>
void VideoWindowDataLayer<Dtype>::InternalThreadEntry(){

    // at each iteration, we sample N windows where N * p are foreground
    // and N*(1-p) are backgorund

    Datum datum;
    CHECK(this->prefetch_data_.count());

    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;

    CPUTimer timer;
    Dtype *top_data = this->prefetch_data_.mutable_cpu_data();
    Dtype *top_label = this->prefetch_label_.mutable_cpu_data();
    const int batch_size = this->layer_param_.video_window_data_param().batch_size();
    const int num_segments = this->layer_param_.video_window_data_param().num_segments();
    const int snippet_len = this->layer_param_.video_window_data_param().snippet_length();
    const int new_height = this->layer_param_.video_window_data_param().new_height();
    const int new_width = this->layer_param_.video_window_data_param().new_width();
    const int is_diff = this->layer_param_.video_window_data_param().modality()
                         == VideoWindowDataParameter_Modality_DIFF;

    // zero out batch
    caffe_set(this->prefetch_data_.count(), Dtype(0), top_data);

    const int num_fg = static_cast<int>(static_cast<float>(batch_size) * fg_fraction_);
    const int num_samples[2] = {batch_size - num_fg, num_fg};

    int item_id = 0;
    // first, sample background, then foreground
    for (int is_fg = 0; is_fg < 2; ++is_fg){
        for (int sample_idx = 0; sample_idx < num_samples[is_fg]; ++sample_idx){
            // sample a window
            timer.Start();
            const unsigned rand_index = PrefetchRand();
            vector<float> window;
            switch (this->layer_param_.video_window_data_param().mode()){
                case VideoWindowDataParameter_Mode_PROP:
                case VideoWindowDataParameter_Mode_DET: {
                    if (!is_fg){
                        window = bg_windows_[rand_index % bg_windows_.size()];
                    }else{
                        if (this->layer_param_.video_window_data_param().gt_fg()) {
                            window = flat_gt_windows_[rand_index % flat_gt_windows_.size()];
                        }else{
                            window = fg_windows_[rand_index % fg_windows_.size()];
                        }
                    }
                    break;
                }
                case VideoWindowDataParameter_Mode_CLS:{
                    window = flat_gt_windows_[rand_index % flat_gt_windows_.size()];
                    break;
                }
            }

            int video_index = window[VideoWindowDataLayer::VIDEO_INDEX]-1;

            pair<string, vector<float> > video_info =
                    video_database_[video_index];

            //std::cout<<video_info.first<<" "<<video_info.second[0]<<" "<<window[0]<<"\n";

            string video_path = video_info.first;
            const float fps = video_info.second[1];
            const int total_frame = static_cast<int>(video_info.second[0] * fps);
            const int start_frame = static_cast<int>(window[VideoWindowDataLayer::START] * fps);
            const int end_frame = static_cast<int>(window[VideoWindowDataLayer::END] * fps);
            const int label = (this->layer_param_.video_window_data_param().merge_positive())
                              ? is_fg : window[VideoWindowDataLayer::LABEL];


            CHECK_GE(end_frame, start_frame)<<"incorrect window in video path: "<<video_path;
            vector<int> offsets = this->SampleSegments(start_frame, end_frame, 0,
                                                       total_frame, num_segments,
                                                       snippet_len + is_diff, // diff needs one more
                                                       this->phase_ == TRAIN, this->layer_param_.video_window_data_param().boundary_frame());

            switch(this->layer_param_.video_window_data_param().modality()){
                case VideoWindowDataParameter_Modality_FLOW:
                    ReadSegmentFlowToDatum(video_path, label,
                                           offsets, new_height, new_width, snippet_len,
                                           &datum, name_pattern_.c_str());
                    break;
                case VideoWindowDataParameter_Modality_RGB:
                    ReadSegmentRGBToDatum(video_path, label,
                                          offsets, new_height, new_width, snippet_len,
                                          &datum, true, name_pattern_.c_str());
                    break;
                case VideoWindowDataParameter_Modality_DIFF:
                    ReadSegmentRGBDiffToDatum(video_path, label,
                                          offsets, new_height, new_width, snippet_len,
                                          &datum, true, name_pattern_.c_str());
                    break;
            };
            read_time += timer.MicroSeconds();
            timer.Start();

            int offset1 = this->prefetch_data_.offset(item_id);
            this->transformed_data_.set_cpu_data(top_data + offset1);
            this->data_transformer_->Transform(datum, &(this->transformed_data_));
            top_label[item_id] = label;


            item_id++;


#if 0
            if (is_fg){
                LOG(INFO)<<"Foreground window. video_index: "<<window[VideoWindowDataLayer::VIDEO_INDEX]
                <<", total_frame: "<<total_frame
                <<", start_frame: "<<start_frame<<", end_frame:"<<end_frame
                <<", overlap: "<<window[VideoWindowDataLayer::OVERLAP];
            }else{
                LOG(INFO)<<"Background window. , video_index: "<<window[VideoWindowDataLayer::VIDEO_INDEX]
                <<", total_frame: "<<total_frame
                <<", start_frame: "<<start_frame<<", end_frame:"<<end_frame
                <<", overlap: "<<window[VideoWindowDataLayer::OVERLAP];
            }

            LOG(INFO)<<"Sampled frame indices:";
            for (int fi = 0; fi < offsets.size(); ++fi) {
                LOG(INFO) << "\t" << offsets[fi];
            }


            vector< vector<float> > video_gt_windows = gt_windows_[video_index];
            LOG(INFO)<<"Video groundtruth windows:";
            for (int gt_idx = 0; gt_idx < video_gt_windows.size(); ++gt_idx){
                LOG(INFO)<<"\t"
                <<video_gt_windows[gt_idx][VideoWindowDataLayer::LABEL]
                <<" "<<video_gt_windows[gt_idx][VideoWindowDataLayer::START]
                <<" "<<video_gt_windows[gt_idx][VideoWindowDataLayer::END];
            }
#endif

#if 0
            // write loaded frames to disk for debug
            if (!is_fg) continue;
            string base_path = "local/debug_frames/snippet_%d_frame_idx_%d_class_%d.jpg";
            int l = 1;
            char tmp[1024];
            cv::Mat out_img;
            const int crop_size = 224;
            int img_size[3] = {crop_size, crop_size};
            out_img.create(2, img_size, CV_8UC3);

            size_t offset = crop_size * crop_size * 3;
            const int mean_val[3] = {104, 117, 123};
            for (int fi = 0; fi < l * (num_segments+2); ++fi){
                int seg_idx = fi / l;
                int frame_idx = offsets[seg_idx] + fi % (l);
                sprintf(tmp, base_path.c_str(), seg_idx, frame_idx, label-1);


                for (int c = 0; c < 3; ++c) {
                    for (int h = 0; h < crop_size; ++h) {
                        for (int w = 0; w < crop_size; ++w) {
                            float v = top_data[offset1 + offset * fi + w + (h + c * crop_size) * crop_size] + mean_val[c];
//                            LOG(INFO)<<"fi: "<<fi<<" c: "<<c<<" h: "<<h<<" w: "<<w;
//                            LOG(INFO)<<int(v);
                            cv::Vec3b& color = out_img.at<cv::Vec3b>(h, w);
                            color[c] = static_cast<uint8_t>(v);
                        }
                    }
                }

                cv::imwrite(tmp, out_img);
            }
            exit(0);
#endif
        }
    }
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(VideoWindowDataLayer);
REGISTER_LAYER_CLASS(VideoWindowData);

}
