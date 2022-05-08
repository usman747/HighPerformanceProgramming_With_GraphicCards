//////////////////////////////////////////////////////////////////////////////
// OpenCL exercise 1: Basics
//////////////////////////////////////////////////////////////////////////////

// includes
#include <stdio.h>

#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////
void calculateHost(const std::vector<float>& h_input, std::vector<float>& h_output) {
	for (std::size_t i = 0; i < h_output.size(); i++)
		h_output[i] = std::cos(h_input[i]);
}

//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	// Create a context
	//cl::Context context(CL_DEVICE_TYPE_GPU);
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cerr << "No platforms found" << std::endl;
		return 1;
	}
	int platformId = 0;
	for (size_t i = 0; i < platforms.size(); i++) {
		if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing") {
			platformId = i;
			break;
		}
	}
	cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[platformId](), 0, 0 };
	std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
	cl::Context context(CL_DEVICE_TYPE_GPU, prop);

	// Get the first device of the context
	std::cout << "Context has " << context.getInfo<CL_CONTEXT_DEVICES>().size() << " devices" << std::endl;
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);    // passing GPU device 

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "D:/DATA/University of stuttgart/COURSES/4th semester/GPU Lab/Ex_1/Opencl-Basics-Windows/Opencl-ex1/src/OpenCLExercise1_Basics.cl");
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	// Create a kernel object
	cl::Kernel kernel1(program, "kernel1");    //// calling specific kernal in the CL 

	// Declare some values
	std::size_t wgSize = 128; // Number of work items per work group    
	std::size_t count = wgSize * 100000; // Overall number of work items = Number of elements
	std::size_t size = count * sizeof(float); // Size of data in bytes    



	// 128*100000 kernals count
	// each byte has 4 bits
	// so total bytes = 128*100000*4



	// Allocate space for input data and for output data from CPU and GPU on the host 
	std::vector<float> h_input(count);
	std::vector<float> h_outputCpu(count);
	std::vector<float> h_outputGpu(count);   // read output from GPU

	// Allocate space for input and output data on the device
	//TODO
	// 
	//declare variables/buffer for GPU to store data
	//GPU varibles d_input, d_output

	cl::Buffer d_input(context, CL_MEM_READ_ONLY, size);
	cl::Buffer d_output(context, CL_MEM_WRITE_ONLY, size);


	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
	memset(h_input.data(), 255, size);
	memset(h_outputCpu.data(), 255, size);
	memset(h_outputGpu.data(), 255, size);
	//TODO

	// h_input contains data in CPU
	// Copying data from CPU to GPU buffers

	queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data());
	queue.enqueueWriteBuffer(d_output, true, 0, size, h_outputGpu.data());

	// Initialize input data with more or less random values
	for (std::size_t i = 0; i < count; i++)
		h_input[i] = ((i * 1009) % 31) * 0.1;

	// Do calculation on the host side
	// Now we will checck how much time does CPU and GPU take to excute 
	// First note CPU time

	Core::TimeSpan time1 = Core::getCurrentTime();
	calculateHost(h_input, h_outputCpu);
	Core::TimeSpan time2 = Core::getCurrentTime();
	Core::TimeSpan out = time2 - time1;    // total time CPU took to excute instructiuons

	// Copy input data to device (device is GPU)

	//TODO: enqueueWriteBuffer()

	cl::Event dataWriteTime;

	queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data(), NULL, &dataWriteTime);
	queue.enqueueWriteBuffer(d_output, true, 0, size, h_outputGpu.data(), NULL, &dataWriteTime);

	Core::TimeSpan timeW = OpenCL::getElapsedTime(dataWriteTime);    // time taken for data to write



	// Launch kernel on the device
	//TODO
	// Kernal 1 function in CL have 2 arguments 
	// we will set its arguments 

	kernel1.setArg(0, d_input);    // first argument is d_input 
	kernel1.setArg(1, d_output);


	// Copy output data back to host
	//TODO: enqueueReadBuffer()
	// now excute kernal using command below



	cl::Event KernalEx; // make Event class object to campute time
	queue.enqueueNDRangeKernel(kernel1, 0, count, wgSize, NULL, &KernalEx);   // count is no of cores, workgroup size, &KernalEx to store event times 

	// Now the GPU kernal is executed and output data (d_output) is updated
	// Now bring back the latest output_d updated by GPU
	// calculate excution time on GPU

	cl::Event dataReadTime;

	queue.enqueueReadBuffer(d_output, true, 0, size, h_outputGpu.data(), NULL, &dataReadTime);
	queue.finish();


	// Print performance data
	//TODO
	Core::TimeSpan timeR = OpenCL::getElapsedTime(dataReadTime);    // timespan class has function getElapsedTime() that convert event class object to time 


	Core::TimeSpan cpuTime = out;    // CPU time
	Core::TimeSpan gpuTime = OpenCL::getElapsedTime(KernalEx);    /// GPU time 
	Core::TimeSpan copyTime1 = timeW;      // data write time
	Core::TimeSpan copyTime2 = timeR;      // data read time
	Core::TimeSpan copyTime = copyTime1 + copyTime2;    /// total transfer time
	Core::TimeSpan overallGpuTime = gpuTime + copyTime;

	std::cout << "CPU Time: " << cpuTime.toString() << std::endl;
	std::cout << "Memory copy Time: " << copyTime.toString() << std::endl;
	std::cout << "GPU Time w/o memory copy: " << gpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / gpuTime.getSeconds()) << ")" << std::endl;  // 
	std::cout << "GPU Time with memory copy: " << overallGpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / overallGpuTime.getSeconds()) << ")" << std::endl;




	// Check whether results are correct
	std::size_t errorCount = 0;
	for (std::size_t i = 0; i < count; i++) {
		// Allow small differences between CPU and GPU results (due to different rounding behavior)
		if (!(std::abs(h_outputCpu[i] - h_outputGpu[i]) <= 10e-5))
		{
			if (errorCount < 15)
				std::cout << "Result for " << i << " is incorrect: GPU value is " << h_outputGpu[i] << ", CPU value is " << h_outputCpu[i] << std::endl;
			else if (errorCount == 15)
				std::cout << "..." << std::endl;
			errorCount++;
		}
	}
	if (errorCount != 0) {
		std::cout << "Found " << errorCount << " incorrect results" << std::endl;
		return 1;
	}

	std::cout << "Success" << std::endl;

	return 0;
}
