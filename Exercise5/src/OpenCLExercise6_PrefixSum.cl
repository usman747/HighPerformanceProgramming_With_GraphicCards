
#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack To make syntax highlighting In Eclipse work
#endif

__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
__kernel void prefixSumKernel(__global int* d_input,__global int* d_output,__global int* sum) {
//TODO

    __local int data[WG_SIZE];
    int i = get_global_id(0);
    int j = get_local_id(0);
    int k = get_group_id(0);

    data[j] = d_input[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = 1; offset < WG_SIZE; offset*= 2){
    int temp = data[j];
    if (j >= offset){
        temp += data[j - offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    data[j] = temp;
    barrier(CLK_LOCAL_MEM_FENCE);
    }
    d_output[i] = data[j];
    if (j == WG_SIZE - 1 && sum){
    sum[k] = data[j];
    }
    
}

__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
__kernel void blockAddKernel(__global int* d_output, __global int* sum) {
	//TODO
    int i = get_global_id(0);
    int k = get_group_id(0);
                             
    if (k == 0)
        return;


    d_output[i] = d_output[i] + sum[k-1];
}