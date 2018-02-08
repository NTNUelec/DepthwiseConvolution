/*
 * depthwise_conv_layer.hpp
 *
 *  Created on: May 23, 2017
 *      Author: liuhao
 *  Edited on: Dec 16, 2017
 *      Author: HongYen Chen 
 */

#include <vector>
#include <algorithm>
#include <cfloat>
#include "caffe/layers/depthwise_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"


/*
 * The depthwise layer for mobilenet. depth_multiplier = out_channels / in_channels. depth_multiplier can only be an integer. 
 */

namespace caffe {

template <typename Dtype>
__global__ void ConvForward(const int nthreads,
	const Dtype* const bottom_data, const int num, const int in_channels,
	const int in_height, const int in_width,const int conved_height,
	const int conved_width, const int kernel_h, const int kernel_w,
	const int stride_h, const int stride_w, const int pad_h, const int pad_w,
	Dtype* const top_data, const Dtype* const weight, const Dtype* const bias, const bool bias_term_, const int depth_multiplier_) {

	CUDA_KERNEL_LOOP(index, nthreads) {//index is from 0 to nthread - 1, parallel execute. nthread is output's count.

		const int pw = index % conved_width;// 该线程对应的top blob（N,C,H,W）中的w,输出Feature Map的中的高的坐标
		const int ph = (index / conved_width) % conved_height;// 该线程对应的top blob（N,C,H,W）中的h,输出Feature Map的中的宽的坐标
		const int c = (index / conved_width / conved_height) % (in_channels * depth_multiplier_);// 该线程对应的top blob（N,C,H,W）中的C,即第c个Channel(number of feature maps)
		const int n = index / conved_width / conved_height / (in_channels * depth_multiplier_);// 该线程对应的top blob（N,C,H,W）中的N,即n个样本个数

        // hstart,wstart,hend,wend分别为bottom blob（上一层feature map）中的点的坐标范围
   	    // 由这些点计算出该线程对应的点（top blob中的点）
		int hstart = ph * stride_h - pad_h;
		int wstart = pw * stride_w - pad_w;
		int hend = min(hstart + kernel_h, in_height + pad_h);
		int wend = min(wstart + kernel_w, in_width + pad_w);
		// const int filter_size = (hend - hstart) * (wend - wstart);
		hstart = max(hstart, 0);// pad may let hstart < 0, so we need to adjust hstart and let it not smaller than 0
		wstart = max(wstart, 0);// pad may let wstart < 0, so we need to adjust wstart and let it not smaller than 0
		hend = min(hend, in_height);// pad may let hend > in_height, so we need to adjust hend and let it not bigger than in_height
		wend = min(wend, in_width);// pad may let wend > in_width, so we need to adjust wend and let it not bigger than in_height
		Dtype val = 0;
		const Dtype* const bottom_slice = bottom_data + (n * in_channels + (c % in_channels)) * in_height * in_width;
		const Dtype* const weight_slice = weight + c * kernel_h * kernel_w;                

		int khstart = hend < kernel_h ? kernel_h-hend:0;// if hend < kernel_h, it means the filter will calculate with input feature map which includes pad value 0, so we can skip it.
		int kwstart = wend < kernel_w ? kernel_w-wend:0;// if wend < kernel_w, it means the filter will calculate with input feature map which includes pad value 0, so we can skip it.
                
		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {
				val += bottom_slice[h * in_width + w] * weight_slice[(khstart+h-hstart) * kernel_w + (kwstart+w-wstart)];        
			}
		}
		if(bias_term_) {
			val += bias[c];
		}
		top_data[index] = val;                
	}	   
}

template<typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::Forward_gpu(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	const Dtype* weight = this->blobs_[0]->gpu_data();
	int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
	int* stride_data = this->stride_.mutable_cpu_data();
	int* pad_data = this->pad_.mutable_cpu_data();
        
	for (int i = 0; i < bottom.size(); ++i) {
		const Dtype* bottom_data = bottom[i]->gpu_data();
		Dtype* top_data = top[i]->mutable_gpu_data();
		const int count = top[i]->count();
		vector<int> in_shape_ = bottom[i]->shape();
        vector<int> out_shape_ = top[i]->shape();
		const int in_channels_ = in_shape_[1];
		const int in_height_ = in_shape_[2];
		const int in_width_ = in_shape_[3];
                
        const int out_channels = out_shape_[1];

		const int kernel_h_ = kernel_shape_data[0];
		const int kernel_w_ = kernel_shape_data[1];
		const int stride_h_ = stride_data[0];
		const int stride_w_ = stride_data[1];
		const int pad_h_ = pad_data[0];
		const int pad_w_ = pad_data[1];

		const int conved_height = this->output_shape_[0];// The spatial dimensions of the output, after convolution.
		const int conved_width = this->output_shape_[1];// The spatial dimensions of the output, after convolution.

		const bool bias_term_ = this->bias_term_;

        const int depth_multiplier_ = out_channels / in_channels_;

		if (bias_term_) {
			const Dtype* const bias = this->blobs_[1]->gpu_data();
			ConvForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
					count, bottom_data, bottom[i]->num(), in_channels_,
					in_height_, in_width_,conved_height,conved_width,kernel_h_,
					kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data, weight, bias, bias_term_, depth_multiplier_);
		} 
		else {
			ConvForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
					count, bottom_data, bottom[i]->num(), in_channels_,
					in_height_, in_width_,conved_height,conved_width,kernel_h_,
					kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data, weight, 0, bias_term_, depth_multiplier_);
		}
	}
}

template <typename Dtype>
__global__ void ConvBackward(const int nthreads,
const Dtype* const top_diff,
const int num, const int in_channels, const int height,
const int width, const int conved_height, const int conved_width,
const int kernel_h, const int kernel_w, const int stride_h,
const int stride_w, const int pad_h, const int pad_w,
Dtype* const bottom_diff,const Dtype* const weight, const int depth_multiplier_) {

	CUDA_KERNEL_LOOP(index, nthreads) { // index is from 0 to nthread-1, parallel execute. nthread = count_buttom
		const int w = index % width + pad_w;
		const int h = (index / width) % height + pad_h;
		const int c = (index / width / height) % in_channels;
		const int n = index / width / height / in_channels;
		
		const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
		const int phend = min(h / stride_h + 1, conved_height);
		const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
		const int pwend = min(w / stride_w + 1, conved_width);
		
		const int khstart = (h >= kernel_h) ? ((h-kernel_h) % stride_h) + (kernel_h-stride_h) : h;
		const int kwstart = (w >= kernel_w) ? ((w-kernel_w) % stride_w) + (kernel_w-stride_w) : w;
	    		
		Dtype error  = 0; // for back layer diff, error of back layer = error of top layer * weight
									
		for (int ph = phstart; ph < phend; ++ph) {
			for (int pw = pwstart; pw < pwend; ++pw) {
				int kh = khstart - (ph-phstart) * stride_h;
				int kw = kwstart - (pw-pwstart) * stride_w;
				for(int i = 0; i < depth_multiplier_; i++){
					const Dtype* const top_diff_slice = top_diff + (n * (in_channels * depth_multiplier_) + c + in_channels * i) * conved_height * conved_width;		
					const Dtype* const weight_slice = weight + (c + in_channels * i) * kernel_h * kernel_w;	
					error += top_diff_slice[ph * conved_width + pw] * weight_slice[kh * kernel_w + kw];									
				}
			}
		}
		bottom_diff[index] = error;		
	}
}

__device__ float atomicAddme(float* address, float val)
{
    return atomicAdd(address,val);
}

__device__ double atomicAddme(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

#define DIVIDE_CEIL(a,b) a/b+((a/b*b)<a)

template <typename Dtype>
__global__ void ConvBackwardWeight(const int nthreads,
const Dtype* const top_diff,
const int num, const int out_channels, const int in_height,
const int in_width, const int conved_height, const int conved_width,
const int kernel_h, const int kernel_w, const int stride_h,
const int stride_w, const int pad_h, const int pad_w,
Dtype* const weight_diff, const Dtype* const bottom_data, const int depth_multiplier_) {

	CUDA_KERNEL_LOOP(index, nthreads) { //nthread = count_weight
		const int kw = index % kernel_w;
		const int kh = (index / kernel_w) % kernel_h;
		const int c = index / kernel_w / kernel_h;
		
		Dtype gradient = 0; // gradient = error of top layer * input of bottom layer
		for(int n=0;n<num;n++) {
			
			const Dtype* const top_diff_slice = top_diff + (n * out_channels + c) * conved_height * conved_width;
			const Dtype* const bottom_data_slice = bottom_data + (n * (out_channels/depth_multiplier_) + (c % (out_channels/depth_multiplier_))) * in_height * in_width;
					
			const int phstart = max(DIVIDE_CEIL((pad_h-kh),stride_h),0);
			const int phend = min(DIVIDE_CEIL((in_height+pad_h-kh),stride_h),conved_height);		
			const int pwstart = max(DIVIDE_CEIL((pad_w-kw),stride_w),0);			
			const int pwend = min(DIVIDE_CEIL((in_width+pad_w-kw),stride_w),conved_width);

			for(int ph = phstart;ph<phend;ph++){
				for (int pw = pwstart;pw<pwend;pw++){
					const int h = ph*stride_h+kh-pad_h;
					const int w = pw*stride_w+kw-pad_w;
					gradient += top_diff_slice[ph * conved_width + pw] * bottom_data_slice[h * in_width + w];
				}
			}
		}
		weight_diff[c * kernel_h * kernel_w + kh * kernel_w + kw] += gradient;
	}
}

template <typename Dtype>
__global__ void ConvBackwardBias(const int nthreads,const Dtype* const top_diff,
const int num, const int out_channels_, const int in_height,
const int in_width, const int conved_height, const int conved_width,
const int kernel_h, const int kernel_w, const int stride_h,
const int stride_w, const int pad_h, const int pad_w, Dtype* const bias_diff) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		const int c = index;
		Dtype gradient=0;
		for( int n=0;n<num;n++) {
			const Dtype* const top_diff_slice =	top_diff + (n * out_channels_ + c) * conved_height * conved_width;
			for(int ph=0;ph<conved_height;ph++) {
				for (int pw=0;pw<conved_width;pw++) {
					gradient += top_diff_slice[ph * conved_width + pw];
				}
			}
		}
		bias_diff[c]+=gradient;
	}
}

template<typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::Backward_gpu(
const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
const vector<Blob<Dtype>*>& bottom) {

	int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
	int* stride_data = this->stride_.mutable_cpu_data();
	int* pad_data = this->pad_.mutable_cpu_data();

	const Dtype* weight = this->blobs_[0]->gpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();

	const bool bias_term_ = this->bias_term_;
	Dtype* bias_diff = bias_term_ ? this->blobs_[1]->mutable_gpu_diff() : 0;
	const bool bias_propagate_down_ = this->param_propagate_down_[1];
	const bool weight_propagate_down_ = this->param_propagate_down_[0];


	const int kernel_h_ = kernel_shape_data[0];
	const int kernel_w_ = kernel_shape_data[1];
	const int stride_h_ = stride_data[0];
	const int stride_w_ = stride_data[1];
	const int pad_h_ = pad_data[0];
	const int pad_w_ = pad_data[1];

	const int conved_height = this->output_shape_[0];
	const int conved_weight = this->output_shape_[1];

	for (int i = 0; i < top.size(); ++i) {

		const Dtype* top_diff = top[i]->gpu_diff();
		const Dtype* bottom_data = bottom[i]->gpu_data();
		Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();

		vector<int> in_shape_ = bottom[i]->shape();
		const int in_channels_ = in_shape_[1];
		const int in_height_ = in_shape_[2];
		const int in_width_ = in_shape_[3];

		vector<int> out_shape_ = top[i]->shape();
		const int out_channels_ = out_shape_[1];

		const int depth_multiplier_ = out_channels_ / in_channels_;

		// Bias gradient, if necessary.
		if (bias_term_ && bias_propagate_down_) {
			const int count_bias = out_channels_;
			ConvBackwardBias<Dtype><<<CAFFE_GET_BLOCKS(count_bias), CAFFE_CUDA_NUM_THREADS>>>(
				count_bias, top_diff, bottom[i]->num(), out_channels_,
				in_height_, in_width_, conved_height, conved_weight, kernel_h_,
				kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bias_diff);
		}
		// gradient w.r.t. weight. Note that we will accumulate diffs.
		if (weight_propagate_down_) {
			const int count_weight = out_channels_ * kernel_h_ * kernel_w_;
			ConvBackwardWeight<Dtype><<<CAFFE_GET_BLOCKS(count_weight), CAFFE_CUDA_NUM_THREADS>>>(
				count_weight, top_diff, bottom[i]->num(), out_channels_,
				in_height_, in_width_, conved_height, conved_weight, kernel_h_,
				kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
				weight_diff, bottom_data, depth_multiplier_);
		}
		// gradient w.r.t. bottom data, if necessary.
		if (propagate_down[i]) {
			const int count_bottom = bottom[i]->count();
			ConvBackward<Dtype><<<CAFFE_GET_BLOCKS(count_bottom), CAFFE_CUDA_NUM_THREADS>>>(
				count_bottom, top_diff, bottom[i]->num(), in_channels_,
				in_height_, in_width_, conved_height, conved_weight, kernel_h_,
				kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, 
				bottom_diff, weight, depth_multiplier_);
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS (DepthwiseConvolutionLayer);

}  // namespace caffe
