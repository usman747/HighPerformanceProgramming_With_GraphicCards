#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack To make syntax highlighting In Eclipse work
#End If

__kernel void kernel1 (__global Const float* d_input, __global float* d_output) 

{
	//TODO
	// d_input & d_output Is shared memoery

	int index = get_global_id(0);                  // 0 returns address Of X axis While 1 returns address Of Y axi..  (X,Y) Or (I,J) start location Of GPU Matrix Map 
	//d_output[index] = cos(d_input[index]);        // for CPU

	d_output[index] = native_cos(d_input[index]);      // For Gpu


}