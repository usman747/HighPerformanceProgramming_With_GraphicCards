#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

__kernel void matrixMulKernel1(__global float *h_inputA, __global float*h_inputB, __global float *h_outputC, size_t countAX_BY) {
	//TODO
	
    size_t countBX = get_global_size(0);
	size_t i = get_global_id(1);
    size_t j = get_global_id(0);
	
			float sum = 0;
			for (size_t k = 0; k < countAX_BY; k++) {
				float a = h_inputA[k + j * countAX_BY];
				float b = h_inputB[i + k * countBX];
				sum += a * b;
			}
			h_outputC[i + j * countBX] = sum;
}


// The preprocessor constant WG_SIZE will contain the size of a work group in X/Y-direction

//define WG_SIZE = 16;
//__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matrixMulKernel2(__global float *h_inputA, __global float*h_inputB, __global float *h_outputC, size_t countAX_BY) {
    size_t countBX = get_global_size(0);
	size_t i = get_global_id(1);
    size_t j = get_global_id(0);
    uint wg=16;
	float sum = 0;
	int k = get_local_id (1);
	int g = get_local_id (0);
	__local float l_A[16][16];
	__local float l_B[16][16];
	 // loop over the submatrices
	for (uint bs = 0; bs < countAX_BY; bs += wg) {
		 //Copy blocks of d_inputA , d_inputB to local memory
		l_A[g][k] = h_inputA[(k+bs) + j * countAX_BY];
		l_B[g][k] = h_inputB[i + (g+bs) * countBX];
	
	    barrier(CLK_LOCAL_MEM_FENCE);
		for (uint m = 0; m < wg; m++)
	        sum += l_A[g][m] * l_B[m][k];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	h_outputC[i + j * countBX] = sum;
}