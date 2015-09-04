#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PairRankLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->channels(), bottom[0]->channels());
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
}

template <typename Dtype>
void PairRankLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // b
      diff_.mutable_cpu_data());  // a_i-b_i
      
  caffe_mul(count, diff_.cpu_data(), bottom[2]->cpu_data(), diff_.mutable_cpu_data());
  caffe_cpu_axpby(count, Dtype(-1), diff_.cpu_data(), Dtype(0), diff_.mutable_cpu_data());
  caffe_add_scalar(count, Dtype(1), diff_.mutable_cpu_data());
  
  const int channels = bottom[0]->channels();
  
  Dtype loss(0.0);
  
  for (int i = 0; i < bottom[0]->num(); ++i) {
    for(int j = 0; j < channels; j ++) {
      loss += std::max(Dtype(0), diff_.cpu_data()[i*channels+j]);
    } 
  }
  
  //printf("loss is %f\n", loss);
  loss /= static_cast<Dtype>(bottom[0]->num());
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void PairRankLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      Dtype sign = (i == 0) ? -1 : 1;
      Dtype* bout = bottom[i]->mutable_cpu_diff();
      
      int num = bottom[i]->num();
      int channels = bottom[i]->channels();      
      
      sign *= top[0]->cpu_diff()[0] / num;
      
      for (int j = 0; j < num; ++j) {
        for (int k = 0; k < channels; k ++) {
          Dtype tt = diff_.cpu_data()[j*channels+k] >= 0 ? Dtype(1) : Dtype(0);
          bout[j*channels+k] = tt * sign * bottom[2]->cpu_data()[j*channels+k];
        }
      }
    }
  }  
}

#ifdef CPU_ONLY
STUB_GPU(PairRankLossLayer);
#endif

INSTANTIATE_CLASS(PairRankLossLayer);
REGISTER_LAYER_CLASS(PairRankLoss);

}  // namespace caffe
