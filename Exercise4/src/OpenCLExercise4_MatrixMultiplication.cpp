//////////////////////////////////////////////////////////////////////////////
// OpenCL exercise 4: Matrix multiplication
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
#include <iomanip>
#include <sstream>

#include <boost/lexical_cast.hpp>

//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////
void matrixMulHost(const std::vector<float>& h_inputA, const std::vector<float>& h_inputB, std::vector<float>& h_outputC, std::size_t countAX_BY, std::size_t countAY, std::size_t countBX) {
	for (std::size_t j = 0; j < countAY; j++) {
		for (std::size_t i = 0; i < countBX; i++) {
			float sum = 0;
			for (std::size_t k = 0; k < countAX_BY; k++) {
				float a = h_inputA[k + j * countAX_BY];
				float b = h_inputB[i + k * countBX];
				sum += a * b;
			}
			h_outputC[i + j * countBX] = sum;
		}
	}
}

void printPerformanceHeader() {
	std::cout << "Implementation           CPU       Calc       MT      GPU+MT  Speedup (w/o MT)" << std::endl;
}
void printPerformance(const std::string& name, Core::TimeSpan timeCalc, Core::TimeSpan timeMem, Core::TimeSpan timeCpu, bool showMem = true) {
	Core::TimeSpan overallTime = timeCalc + timeMem;
	std::stringstream str;
	str << std::setiosflags(std::ios::left) << std::setw(20) << name;
	str << std::setiosflags(std::ios::right);
	str << " " << std::setw(9) << timeCpu;
	str << " " << std::setw(9) << timeCalc;
	if (showMem)
		str << " " << std::setw(9) << timeMem;
	else
		str << " " << std::setw(9) << "";
	str << " " << std::setw(9) << overallTime;
	str << "  " << (timeCpu.getSeconds() / overallTime.getSeconds());
	if (showMem)
		str << " (" << (timeCpu.getSeconds() / timeCalc.getSeconds()) << ")";
	std::cout << str.str() << std::endl;
}
void printPerformance(const std::string& name, Core::TimeSpan timeCalc, Core::TimeSpan timeCpu) {
	printPerformance(name, timeCalc, Core::TimeSpan::fromSeconds(0), timeCpu, false);
}

bool compareMatrices(const std::vector<float>& matrix1, const std::string& matrix1N, const std::vector<float>& matrix2, const std::string& matrix2N, std::size_t countX, std::size_t countY) {
	std::size_t errorCount = 0;
	for (size_t j = 0; j < countY; j = j + 1) { //loop in the y-direction
		for (size_t i = 0; i < countX; i = i + 1) { //loop in the x-direction
			size_t index = i + j * countX;
			// Allow small differences between results (due to rounding)
			if (!(std::abs(matrix1[index] - matrix2[index]) <= 1e-2)) {
				if (errorCount < 15)
					std::cout << "Result for " << i << "," << j << " is incorrect: " << matrix1N << " value is " << matrix1[index] << ", " << matrix2N << " value is " << matrix2[index] << std::endl;
				else if (errorCount == 15)
					std::cout << "..." << std::endl;
				errorCount++;
			}
		}
	}
	if (errorCount != 0) {
		std::cout << "Found " << errorCount << " incorrect results" << std::endl;
		return false;
	}
	return true;
}

void dumpMatrix(const std::string& name, const std::vector<float>& matrix, std::size_t countX, std::size_t countY) {
	std::cout << name << " =" << std::endl;
	for (size_t j = 0; j < countY; j = j + 1) { //loop in the y-direction
		std::stringstream str;
		for (size_t i = 0; i < countX; i = i + 1) { //loop in the x-direction
			if (i)
				str << " ";
			str.width(8);
			str << matrix[i + countX * j];
		}
		std::cout << "(" << str.str() << ")" << std::endl;
	}
	std::cout << std::endl;
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

	// Declare some values
	// Declare some values
	std::size_t wgSize = 16;
	std::size_t countAX_BY = 128;
	std::size_t countAY = 128;
	std::size_t countBX = 128;


	std::size_t countCX = countBX;
	std::size_t countCY = countAY;
	std::size_t countA = countAX_BY * countAY;
	std::size_t countB = countBX * countAX_BY;
	std::size_t countC = countCX * countCY;
	std::size_t sizeA = countA * sizeof(float);
	std::size_t sizeB = countB * sizeof(float);
	std::size_t sizeC = countC * sizeof(float);

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "../../../src/OpenCLExercise4_MatrixMultiplication.cl");
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	// This will pass the value of wgSize as a preprocessor constant "WG_SIZE" to the OpenCL C compiler
	// 
	// 
	// 
	// 
	//std::cout << "Hellooooo";
	//OpenCL::buildProgram(program, devices, "-DWG_SIZE=" + boost::lexical_cast<std::string>(wgSize));


	// Allocate space for output data from CPU and GPU on the host
	std::vector<float> h_inputA(countA);
	std::vector<float> h_inputB(countB);
	std::vector<float> h_outputCCpu(countC);
	std::vector<float> h_outputCGpu(countC);

	// Allocate space for input and output data on the device.................................................................................................................................................
	//TODO
	cl::Buffer d_inputA(context, CL_MEM_READ_WRITE, sizeA);
	cl::Buffer d_inputB(context, CL_MEM_READ_WRITE, sizeB);
	cl::Buffer d_outputC(context, CL_MEM_READ_WRITE, sizeC);

	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
	memset(h_inputA.data(), 255, sizeA);
	memset(h_inputB.data(), 255, sizeB);
	memset(h_outputCCpu.data(), 255, sizeC);
	memset(h_outputCGpu.data(), 255, sizeC);
	//TODO: GPU....................................................................................................................................................................................................
	queue.enqueueWriteBuffer(d_inputA, true, 0, sizeA, h_inputA.data());
	queue.enqueueWriteBuffer(d_inputB, true, 0, sizeB, h_inputB.data());
	queue.enqueueWriteBuffer(d_outputC, true, 0, sizeC, h_outputCGpu.data());


	//////// Generate input data ////////////////////////////////
	// Use random input data
	for (std::size_t i = 0; i < countA; i++)
		h_inputA[i] = (rand() % 100) / 5.0f - 10.0f;
	for (std::size_t i = 0; i < countB; i++)
		h_inputB[i] = (rand() % 100) / 5.0f - 10.0f;
	// Use integer numbers as data
	/*
	for (std::size_t i = 0; i < countA; i++)
		h_inputA[i] = i;
	for (std::size_t i = 0; i < countB; i++)
		h_inputB[i] = (int)i - 5;
	*/

	// Do calculation on the host side.........................................................................................................................................................

	Core::TimeSpan time1 = Core::getCurrentTime();
	matrixMulHost(h_inputA, h_inputB, h_outputCCpu, countAX_BY, countAY, countBX);
	Core::TimeSpan time2 = Core::getCurrentTime();
	Core::TimeSpan cpuTime = time2 - time1;


	//Core::TimeSpan cpuTime = Core::TimeSpan::fromSeconds(0); //TODO
//Core::TimeSpan atlasTime = Core::TimeSpan::fromSeconds(0); //TODO
	//printPerformanceHeader();
	//printPerformance("CPU", cpuTime);
//printPerformance("CPU", cpuTime, atlasTime);


	// Copy input data to device.............................................................................................................................................................
	//TODO
	cl::Event dataWriteTime;
	queue.enqueueWriteBuffer(d_inputA, true, 0, sizeA, h_inputA.data(), NULL, &dataWriteTime);
	Core::TimeSpan timeW1 = OpenCL::getElapsedTime(dataWriteTime);

	queue.enqueueWriteBuffer(d_inputB, true, 0, sizeB, h_inputB.data(), NULL, &dataWriteTime);
	Core::TimeSpan timeW2 = OpenCL::getElapsedTime(dataWriteTime);

	queue.enqueueWriteBuffer(d_outputC, true, 0, sizeC, h_outputCGpu.data(), NULL, &dataWriteTime);
	Core::TimeSpan timeW3 = OpenCL::getElapsedTime(dataWriteTime);
	Core::TimeSpan CopyTime = timeW1 + timeW2 + timeW3;

	// Iterate over all implementations (task 1 - 2)
	for (int impl = 1; impl <= 1; impl++) {
		// Reinitialize output memory to 0xff
		memset(h_outputCGpu.data(), 255, sizeC);
		//TODO.............................................................................................................................................
		queue.enqueueWriteBuffer(d_outputC, true, 0, sizeC, h_outputCGpu.data());

		// Create a kernel object
		std::string kernelName = "matrixMulKernel" + boost::lexical_cast<std::string> (impl);
		cl::Kernel matrixMulKernel(program, kernelName.c_str());

		// Launch kernel on the device....................................................................................................................................................................................
		//TODO

		matrixMulKernel.setArg(0, d_inputA);
		matrixMulKernel.setArg(1, d_inputB);
		matrixMulKernel.setArg(2, d_outputC);
		matrixMulKernel.setArg(3, countAX_BY);


		cl::Event KernalEx;
		queue.enqueueNDRangeKernel(matrixMulKernel, 0, cl::NDRange(countAY, countBX), cl::NDRange(wgSize, wgSize), NULL, &KernalEx);
		//queue.finish();

		// Copy output data back to host.....................................................................................................................................................................................
		//TODO
		queue.enqueueReadBuffer(d_outputC, true, 0, sizeC, h_outputCGpu.data(), NULL);
		Core::TimeSpan gpuTime = OpenCL::getElapsedTime(KernalEx);

		// Print performance data................................................................................................................................................................
		std::cout << "####Implementation#" << impl << "#####" << std::endl;
		std::cout << "CPU Time: " << cpuTime.toString() << std::endl;
		std::cout << "GPU Time w/o memory copy: " << gpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / gpuTime.getSeconds()) << ")" << std::endl;
		//Core::TimeSpan gpuTime(0); //TODO
		//Core::TimeSpan copyTime(0); //TODO
		//printPerformance(kernelName, gpuTime, copyTime);

		// Check whether results are correct
		if (!compareMatrices(h_outputCCpu, "CPU", h_outputCGpu, "GPU", countCX, countCY))
			return 1;
	}

	std::cout << "Success" << std::endl;

	//dumpMatrix ("A", h_inputA, countAX_BY, countAY);
	//dumpMatrix ("B", h_inputB, countBX, countAX_BY);
	//dumpMatrix ("C", h_outputCCpu, countCX, countCY);

	return 0;
}
