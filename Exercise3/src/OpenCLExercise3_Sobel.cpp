//////////////////////////////////////////////////////////////////////////////
// OpenCL exercise 3: Sobel filter
//////////////////////////////////////////////////////////////////////////////

// includes
#include <stdio.h>

#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

#include <boost/lexical_cast.hpp>

//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////
int getIndexGlobal(std::size_t countX, int i, int j) {
	return j * countX + i;
}
// Read value from global array a, return 0 if outside image
float getValueGlobal(const std::vector<float>& a, std::size_t countX, std::size_t countY, int i, int j) {
	if (i < 0 || (size_t)i >= countX || j < 0 || (size_t)j >= countY)
		return 0;
	else
		return a[getIndexGlobal(countX, i, j)];
}
void sobelHost(const std::vector<float>& h_input, std::vector<float>& h_outputCpu, std::size_t countX, std::size_t countY)
{
	for (int i = 0; i < (int)countX; i++) {
		for (int j = 0; j < (int)countY; j++) {
			float Gx = getValueGlobal(h_input, countX, countY, i - 1, j - 1) + 2 * getValueGlobal(h_input, countX, countY, i - 1, j) + getValueGlobal(h_input, countX, countY, i - 1, j + 1)
				- getValueGlobal(h_input, countX, countY, i + 1, j - 1) - 2 * getValueGlobal(h_input, countX, countY, i + 1, j) - getValueGlobal(h_input, countX, countY, i + 1, j + 1);
			float Gy = getValueGlobal(h_input, countX, countY, i - 1, j - 1) + 2 * getValueGlobal(h_input, countX, countY, i, j - 1) + getValueGlobal(h_input, countX, countY, i + 1, j - 1)
				- getValueGlobal(h_input, countX, countY, i - 1, j + 1) - 2 * getValueGlobal(h_input, countX, countY, i, j + 1) - getValueGlobal(h_input, countX, countY, i + 1, j + 1);
			h_outputCpu[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
		}
	}
}

//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
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


	// Get a device of the context
	int deviceNr = argc < 2 ? 1 : atoi(argv[1]);
	std::cout << "Using device " << deviceNr << " / " << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
	ASSERT(deviceNr > 0);
	ASSERT((size_t)deviceNr <= context.getInfo<CL_CONTEXT_DEVICES>().size());
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceNr - 1];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "../../../src/OpenCLExercise3_Sobel.cl");
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	// Declare some values
	std::size_t wgSizeX = 16; // Number of work items per work group in X direction
	std::size_t wgSizeY = 16;
	std::size_t countX = wgSizeX * 40; // Overall number of work items in X direction = Number of elements in X direction
	std::size_t countY = wgSizeY * 30;
	//countX *= 3; countY *= 3;
	std::size_t count = countX * countY; // Overall number of elements
	std::size_t size = count * sizeof(float); // Size of data in bytes

	// Allocate space for output data from CPU and GPU on the host
	std::vector<float> h_input(count);
	std::vector<float> h_outputCpu(count);
	std::vector<float> h_outputGpu(count);


	// Allocate space for input and output data on the device..................................................................................................................................................
	//TODO
	//TODO
	// 
	//declare variables/buffer for GPU to store data
	//GPU varibles d_input, d_output

	cl::Buffer d_input(context, CL_MEM_READ_ONLY, size);
	cl::Buffer d_output(context, CL_MEM_WRITE_ONLY, size);
	cl::Image2D d_image(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_R, CL_FLOAT), countX, countY);     //for task3
	cl::size_t<3> origin; ////for task2,3
	origin[0] = origin[1] = origin[2] = 0; //for task2,3
	cl::size_t<3> region; //for task2,3
	region[0] = countX; //for task2,3
	region[1] = countY; //for task2,3
	region[2] = 1; //for task2,3



	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
	memset(h_input.data(), 255, size);
	memset(h_outputCpu.data(), 255, size);
	memset(h_outputGpu.data(), 255, size);
	//TODO: GPU

	//////// Load input data ////////////////////////////////
	// Use random input data
	/*
	for (int i = 0; i < count; i++)
		h_input[i] = (rand() % 100) / 5.0f - 10.0f;
	*/
	// Use an image (Valve.pgm) as input data
	{
		std::vector<float> inputData;
		std::size_t inputWidth, inputHeight;
		Core::readImagePGM("../../../src/Valve.pgm", inputData, inputWidth, inputHeight);
		for (size_t j = 0; j < countY; j++) {
			for (size_t i = 0; i < countX; i++) {
				h_input[i + countX * j] = inputData[(i % inputWidth) + inputWidth * (j % inputHeight)];
			}
		}
	}

	// Do calculation on the host side..........................................................................................................................................................................
	// Now we will checck how much time does CPU and GPU take to excute 
	// First note CPU time

	Core::TimeSpan time1 = Core::getCurrentTime();
	sobelHost(h_input, h_outputCpu, countX, countY);
	Core::TimeSpan time2 = Core::getCurrentTime();
	Core::TimeSpan cpuTime = time2 - time1;

	//////// Store CPU output image ///////////////////////////////////
	Core::writeImagePGM("output_sobel_cpu.pgm", h_outputCpu, countX, countY);

	std::cout << std::endl;
	// Iterate over all implementations (task 1 - 3)
	for (int impl = 1; impl <= 1; impl++)
	{
		std::cout << "Implementation #" << impl << ":" << std::endl;

		// Reinitialize output memory to 0xff
		memset(h_outputGpu.data(), 255, size);



		//TODO: GPU
		//TODO: enqueueWriteBuffer()

		//----------------------------------------TASK1----------------------- ..uncomment this section and comment task3 section below
		//FOR Task1
		// Copy input data to device........................................................................................................................................................................
		//TODO
		/*
		queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data(), NULL);    // writing CPU to GPU
		queue.enqueueWriteBuffer(d_output, true, 0, size, h_outputGpu.data(), NULL);    
		*/
		//----------------------------------------TASK1-----------------------//


		//----------------------------------------TASK3----------------------- ..uncomment this section and comment task1 section above
		//For TASk3
		if (impl == 3)
			queue.enqueueWriteImage(d_image, true, origin, region, 0, 0, h_input.data(), NULL);
		else
			queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data(), NULL);
		queue.enqueueWriteBuffer(d_output, true, 0, size, h_outputGpu.data(), NULL);
		//----------------------------------------TASK3-----------------------//

		// Create a kernel object
		std::string kernelName = "sobelKernel" + boost::lexical_cast<std::string> (impl);
		cl::Kernel sobelKernel(program, kernelName.c_str());

		//----------------------------------------TASK1----------------------- ..uncomment this section and comment task3 section below
		//For Task1
		/*
		sobelKernel.setArg(0, d_input);
		sobelKernel.setArg(1, d_output);
		*/
		//----------------------------------------TASK1-----------------------//


		//----------------------------------------TASK3----------------------- ..uncomment this section and comment task1 section above
		//For Task3
		if (impl == 3)
			sobelKernel.setArg(0, d_image);
		else
			sobelKernel.setArg(0, d_input);
		
		sobelKernel.setArg(1, d_output);
		//----------------------------------------TASK3-----------------------//

		// for Task 2
		//std::string kernelName = "sobelKernel2" + boost::lexical_cast<std::string> (impl);
		//cl::Kernel sobelKernel2(program, kernelName.c_str ());
		//sobelKernel2.setArg(0, d_input);
		//sobelKernel2.setArg(1, d_output);


		// Launch kernel on the device.....................................................................................................................................................................
		//TODO
		cl::Event KernalEx;
		queue.enqueueNDRangeKernel(sobelKernel, 0, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &KernalEx);


		// For Task 2
		//queue.enqueueNDRangeKernel(sobelKernel2, 0, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &KernalEx);


		// Copy output data in GPU back to host...............................................................................................................................................................
		//TODO
		queue.enqueueReadBuffer(d_output, true, 0, size, h_outputGpu.data(), NULL);      // Reading from output GPU

		// Print performance data...............................................................................................................................................................................
		//TODO
		Core::TimeSpan gpuTime = OpenCL::getElapsedTime(KernalEx);
		std::cout << "CPU Time: " << cpuTime.toString() << std::endl;
		std::cout << "GPU Time w/o memory copy: " << gpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / gpuTime.getSeconds()) << ")" << std::endl;


		//////// Store GPU output image ///////////////////////////////////
		Core::writeImagePGM("output_sobel_gpu_" + boost::lexical_cast<std::string> (impl) + ".pgm", h_outputGpu, countX, countY);

		// Check whether results are correct
		std::size_t errorCount = 0;
		for (size_t i = 0; i < countX; i = i + 1) { //loop in the x-direction
			for (size_t j = 0; j < countY; j = j + 1) { //loop in the y-direction
				size_t index = i + j * countX;
				// Allow small differences between CPU and GPU results (due to different rounding behavior)
				if (!(std::abs(h_outputCpu[index] - h_outputGpu[index]) <= 1e-5)) {
					if (errorCount < 15)
						std::cout << "Result for " << i << "," << j << " is incorrect: GPU value is " << h_outputGpu[index] << ", CPU value is " << h_outputCpu[index] << std::endl;
					else if (errorCount == 15)
						std::cout << "..." << std::endl;
					errorCount++;
				}
			}
		}
		if (errorCount != 0) {
			std::cout << "Found " << errorCount << " incorrect results" << std::endl;
			return 1;
		}

		std::cout << std::endl;
	}

	std::cout << "Success" << std::endl;

	return 0;
}
