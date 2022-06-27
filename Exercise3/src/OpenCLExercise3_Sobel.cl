#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

//TODO TASK 1

int getIndexGlobal(size_t countX, int i, int j) 
{
	return j * countX + i;
}


// Read value from global array a, return 0 if outside image

float getValueGlobal(__global float* a, size_t countX, size_t countY, int i, int j) 
{
	if (i < 0 || (size_t) i >= countX || j < 0 || (size_t) j >= countY)
		return 0;
	else
		return a[getIndexGlobal(countX, i, j)];
}

__kernel void sobelKernel1(__global float* h_input, __global float* h_output) 
{	
	size_t countX = get_global_size(0);         //  limit of X/i direction loop in GPU
    size_t countY = get_global_size(1);         //  limit of Y/j direction loop in GPU


    size_t i = get_global_id(0); // Get global index of the current work item in the x-direction
    size_t j = get_global_id(1); // Get global index of the current work item in the y-direction


	float Gx = getValueGlobal(h_input, countX, countY, i-1, j-1)+2*getValueGlobal(h_input, countX, countY, i-1, j)+getValueGlobal(h_input, countX, countY, i-1, j+1)
					-getValueGlobal(h_input, countX, countY, i+1, j-1)-2*getValueGlobal(h_input, countX, countY, i+1, j)-getValueGlobal(h_input, countX, countY, i+1, j+1);
	float Gy = getValueGlobal(h_input, countX, countY, i-1, j-1)+2*getValueGlobal(h_input, countX, countY, i, j-1)+getValueGlobal(h_input, countX, countY, i+1, j-1)
					-getValueGlobal(h_input, countX, countY, i-1, j+1)-2*getValueGlobal(h_input, countX, countY, i, j+1)-getValueGlobal(h_input, countX, countY, i+1, j+1);

	h_output[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);	
	
}


//TODO TASK 2

__kernel void sobelKernel2(__global float* h_input, __global float* h_output) 
{	
	size_t countX = get_global_size(0);         //  limit of X/i direction loop in GPU
    size_t countY = get_global_size(1);         //  limit of Y/j direction loop in GPU


    size_t i = get_global_id(0); // Get global index of the current work item in the x-direction
    size_t j = get_global_id(1); // Get global index of the current work item in the y-direction


	float c1 = getValueGlobal(h_input, countX, countY, i-1, j-1);
    float c2 = getValueGlobal(h_input, countX, countY, i-1, j+1);
    float c3 = getValueGlobal(h_input, countX, countY, i+1, j-1);
    float c4 = getValueGlobal(h_input, countX, countY, i+1, j+1);


	float Gx = c1 + 2*getValueGlobal(h_input, countX, countY, i-1, j) + c2 - c3 - 2*getValueGlobal(h_input, countX, countY, i+1, j) - c4;
	float Gy = c1 + 2*getValueGlobal(h_input, countX, countY, i, j-1) + c3 - c2 - 2*getValueGlobal(h_input, countX, countY, i, j+1) - c4;

	h_output[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);	
	
}




const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE| CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void sobelKernel3(__read_only image2d_t h_input, __global float* h_output) {

	
	size_t countX = get_global_size(0);
    size_t countY = get_global_size(1);
    size_t i = get_global_id(0); 
    size_t j = get_global_id(1); 
    float corner_1 = read_imagef(h_input, sampler, (int2){i-1, j - 1}).x;
    float corner_2 = read_imagef(h_input, sampler, (int2){i-1, j + 1}).x;
    float corner_3 = read_imagef(h_input, sampler, (int2){i+1, j - 1}).x;
    float corner_4 = read_imagef(h_input, sampler, (int2){i+1, j + 1}).x;
    
    
	float Gx = corner_1 + 2 * read_imagef(h_input, sampler, (int2){i-1, j}).x + corner_2
					-corner_3-2*read_imagef(h_input, sampler, (int2){i+1, j}).x-corner_4;
					
	float Gy = corner_1 + 2 * read_imagef(h_input, sampler, (int2){i, j - 1}).x + corner_3
					-corner_2-2*read_imagef(h_input, sampler, (int2){i, j+1}).x-corner_4;
	
	//write_imagef(h_output, (int2){i,j}, (float4)(sqrt(Gx * Gx + Gy * Gy)));
	h_output[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
		
	
}
