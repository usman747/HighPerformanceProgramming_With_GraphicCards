#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

int getIndexGlobal(size_t countX, int i, int j) {
	return j * countX + i;
}
float getValueGlobal(__global float* a, size_t countX, size_t countY, int i, int j) {
	if (i < 0 || (size_t) i >= countX || j < 0 || (size_t) j >= countY)
		return 0;
	else
		return a[getIndexGlobal(countX, i, j)];
}

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE| CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void gaussiankernal_device(__global float* h_input, __global float* h_output, __global float* kernal){
	
	
	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);

    size_t i = get_global_id(0); 
    size_t j = get_global_id(1); 
	float sum=0.0;
	for (size_t i_ = 0; i_ < 5; i_++)
        for (size_t j_ = 0; j_ < 5; j_++)
			sum += kernal[i_*5+j_] * getValueGlobal(h_input,countX,countY,(i_+i+-1), (j_+j+-1));
	if (sum < 0.0)
		h_output[getIndexGlobal(countX, i, j)] = 0.0;
	else if (sum>255.0)
		h_output[getIndexGlobal(countX, i, j)] = 255.0;
	else
		h_output[getIndexGlobal(countX, i, j)] = sum;
		
}

__kernel void sobel_kernal(__global float* h_input, __global float* h_output, __global float* grad_dire) {

	const float PI = 3.14159265;
	size_t countX = get_global_size(0);
    size_t countY = get_global_size(1);
    size_t i = get_global_id(0); 
    size_t j = get_global_id(1); 
    float corner_1 = getValueGlobal(h_input, countX, countY, i-1, j-1);
    float corner_2 = getValueGlobal(h_input, countX, countY, i-1, j+1);
    float corner_3 = getValueGlobal(h_input, countX, countY, i+1, j-1);
    float corner_4 = getValueGlobal(h_input, countX, countY, i+1, j+1);
    
	float Gx = corner_1+2*getValueGlobal(h_input, countX, countY, i-1, j)+corner_2
					-corner_3-2*getValueGlobal(h_input, countX, countY, i+1, j)-corner_4;
					
	float Gy = corner_1+2*getValueGlobal(h_input, countX, countY, i, j-1)+corner_3
					-corner_2-2*getValueGlobal(h_input, countX, countY, i, j+1)-corner_4;
	h_output[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
	float angle = atan2(Gx, Gy);
	angle = angle * (360.0 / (2.0 * PI)); 
	if (angle <0)
	grad_dire[getIndexGlobal(countX, i, j)] = (angle + 180);
	else
	grad_dire[getIndexGlobal(countX, i, j)] = angle;

}

__kernel void non_max_suppression(__global float* h_input,__global float* grad_dire, __global float* h_output){

int q = 255, r=255;
size_t countX = get_global_size(0);
size_t countY = get_global_size(1);
size_t i = get_global_id(0); 
size_t j = get_global_id(1); 
float point =  getValueGlobal(grad_dire, countX, countY,i,j);
	if (((112.5<=point) && (point<157.5)) ){
			q = getValueGlobal(h_input, countX, countY,i-1,j-1);
			r = getValueGlobal(h_input, countX, countY,i+1,j+1);
	}
	else if (((67.5<=point) && (point<112.5)) ){
			q = getValueGlobal(h_input, countX, countY,i+1,j);
			r = getValueGlobal(h_input, countX, countY,i-1,j);
	
	}
	
	else if (((22.5<=point) && (point<67.5)) ){
			q = getValueGlobal(h_input, countX, countY,i+1,j-1);
			r = getValueGlobal(h_input, countX, countY,i-1,j+1);
	}
			
	else if (((0<=point) && (point<22.5)) || ((157.5<=point) && (point <= 180))){
		q = getValueGlobal(h_input, countX, countY,i,j+1);
		r = getValueGlobal(h_input, countX, countY,i,j-1);
	}
						
	if (getValueGlobal(h_input, countX, countY,i,j)>=q && getValueGlobal(h_input, countX, countY,i,j)>=r){
					h_output[getIndexGlobal( countX,i,j)] = getValueGlobal(h_input, countX, countY,i,j); 
	}
	else 
		h_output[getIndexGlobal( countX,i,j)] = 0;
			
}

__kernel void double_threshold(__global float* h_input, __global float* h_output,   float max_, float low_thre_ratio, float high_thre_ratio){

	float ht = max_ * high_thre_ratio;
	float lt = ht * low_thre_ratio;
	float low = 0;
	float high = 0.5;
	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);
	size_t i = get_global_id(0);
	size_t j = get_global_id(1); 
	float point = getValueGlobal(h_input, countX, countY,i,j);
	int loc = getIndexGlobal( countX,i,j);
	
	if (point >= ht)
		h_output[loc] =  high;

	else if ((point <= ht) && (point >= lt))
		h_output[loc] =  low;
	
	else 
		h_output[loc] =  point;
		
}

__kernel void hysterisis(__global float* h_input,   float low, float high){
	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);
	size_t i = get_global_id(0); 
	size_t j = get_global_id(1); 
	float point = getValueGlobal(h_input, countX, countY,i,j);
	int loc = getIndexGlobal( countX,i,j);
			if (point == low){
				if ((getValueGlobal(h_input, countX, countY,i+1,j-1) == high) || (getValueGlobal(h_input, countX, countY,i+1,j) == high) || (getValueGlobal(h_input, countX, countY,i+1,j+1)== high) || (getValueGlobal(h_input, countX, countY,i,j-1)== high) || (getValueGlobal(h_input, countX, countY,i,j+1)== high) || (getValueGlobal(h_input, countX, countY,i-1,j-1)== high) || (getValueGlobal(h_input, countX, countY,i-1,j)== high) || (getValueGlobal(h_input, countX, countY,i-1,j+1)== high))
					h_input[loc] = high;
				else 
				h_input[loc] = 0;
			}
}