//////////////////////////////////////////////////////////////////////////////
// OpenCL Project : Canny Edge Detection
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
#include <boost/lexical_cast.hpp>
using namespace std;
const float PI = 3.14159265;

int getIndexGlobal(std::size_t countX, int i, int j) {
	return j * countX + i;
}

float getValueGlobal(const std::vector<float>& a, std::size_t countX, std::size_t countY, int i, int j) {
	if (i < 0 || (size_t) i >= countX || j < 0 || (size_t) j >= countY)
		return 0;
	else
		return a[getIndexGlobal(countX, i, j)];
}

void hysteris( std::vector<float>& h_input, float weak, float strong,std::size_t countX, std::size_t countY){
	// Source: https://github.com/FienSoP/canny_edge_detector/blob/master/canny_edge_detector.py

	for (int i=0; i<countX; i++)
		for (int j=0; j<countY; j++){
		
			if (getValueGlobal(h_input, countX, countY,i,j) == weak){
				if ((getValueGlobal(h_input, countX, countY,i+1,j-1) == strong) || (getValueGlobal(h_input, countX, countY,i+1,j) == strong) || (getValueGlobal(h_input, countX, countY,i+1,j+1)== strong) || (getValueGlobal(h_input, countX, countY,i,j-1)== strong) || (getValueGlobal(h_input, countX, countY,i,j+1)== strong) || (getValueGlobal(h_input, countX, countY,i-1,j-1)== strong) || (getValueGlobal(h_input, countX, countY,i-1,j)== strong) || (getValueGlobal(h_input, countX, countY,i-1,j+1)== strong))
					h_input[getIndexGlobal( countX,i,j)] = strong;
				else 
				h_input[getIndexGlobal( countX,i,j)] = 0;
			}
		}
}

void double_theshold(const std::vector<float>& h_input,  std::vector<float>& h_output, float low_thre_ratio, float high_thre_ratio, std::size_t countX, std::size_t countY){
	// Source: https://github.com/FienSoP/canny_edge_detector/blob/master/canny_edge_detector.py
	float max = *max_element(h_input.begin(), h_input.end());
	float ht = max * high_thre_ratio;
	float lt = ht * low_thre_ratio;
	float weak = 0;
	float strong = 0.5;
	
	for (int i=0; i<countX; i++)
		for (int j=0; j<countY; j++){
		     float point = getValueGlobal(h_input, countX, countY,i,j);
			 if (point >= ht)
				h_output[getIndexGlobal( countX,i,j)] =  strong;

			 else if ((point <= ht) && (point >= lt))
				h_output[getIndexGlobal( countX,i,j)] =  weak;
			
			else
			    h_output[getIndexGlobal( countX,i,j)] =  point;

			    
		  
		}
}

void non_max_suppression(const std::vector<float>& h_input,std::vector<float>&grad, std::vector<float>&non_max,std::size_t countX, std::size_t countY){
	
	// Source: https://github.com/FienSoP/canny_edge_detector/blob/master/canny_edge_detector.py
	int pixel1 = 255, pixel2=255;
	
	for (int i = 0; i < (int) countX; i++) {
		for (int j = 0; j < (int) countY; j++) {
			pixel1 = 255;
			pixel2= 255;
			
			
			if (((112.5<=getValueGlobal(grad, countX, countY,i,j)) && (getValueGlobal(grad, countX, countY,i,j)<157.5)) )
			{
				pixel1 = getValueGlobal(h_input, countX, countY,i-1,j-1);
				pixel2 = getValueGlobal(h_input, countX, countY,i+1,j+1);

			}
			
			else if (((67.5<=getValueGlobal(grad, countX, countY,i,j)) && (getValueGlobal(grad, countX, countY,i,j)<112.5)) )
			{
				pixel1 = getValueGlobal(h_input, countX, countY,i+1,j);
				pixel2 = getValueGlobal(h_input, countX, countY,i-1,j);

			}
			
			else if (((22.5<=getValueGlobal(grad, countX, countY,i,j)) && (getValueGlobal(grad, countX, countY,i,j)<67.5)) )
			{
				pixel1 = getValueGlobal(h_input, countX, countY,i+1,j-1);
				pixel2 = getValueGlobal(h_input, countX, countY,i-1,j+1);


			}
			
			else if (((0<=getValueGlobal(grad, countX, countY,i,j)) && (getValueGlobal(grad, countX, countY,i,j)<22.5)) || ((157.5<=getValueGlobal(grad, countX, countY,i,j)) && (getValueGlobal(grad, countX, countY,i,j) <= 180)))
			{
				pixel1 = getValueGlobal(h_input, countX, countY,i,j+1);
				pixel2 = getValueGlobal(h_input, countX, countY,i,j-1);

			}	
			
			if (getValueGlobal(h_input, countX, countY,i,j)>=pixel1 && getValueGlobal(h_input, countX, countY,i,j)>=pixel2){
					non_max[getIndexGlobal( countX,i,j)] = getValueGlobal(h_input, countX, countY,i,j); 
			}
			else 
				non_max[getIndexGlobal( countX,i,j)] = 0;
			
		}}
	
}

void sobelHost(const std::vector<float>& h_input, std::vector<float>& h_outputCpu,std::vector<float>&grad,std::size_t countX, std::size_t countY) {
	// From Lab
	for (int i = 0; i < (int) countX; i++) {
		for (int j = 0; j < (int) countY; j++) {
			float Gx = getValueGlobal(h_input, countX, countY, i-1, j-1)+2*getValueGlobal(h_input, countX, countY, i-1, j)+getValueGlobal(h_input, countX, countY, i-1, j+1)
					-getValueGlobal(h_input, countX, countY, i+1, j-1)-2*getValueGlobal(h_input, countX, countY, i+1, j)-getValueGlobal(h_input, countX, countY, i+1, j+1);
			float Gy = getValueGlobal(h_input, countX, countY, i-1, j-1)+2*getValueGlobal(h_input, countX, countY, i, j-1)+getValueGlobal(h_input, countX, countY, i+1, j-1)
					-getValueGlobal(h_input, countX, countY, i-1, j+1)-2*getValueGlobal(h_input, countX, countY, i, j+1)-getValueGlobal(h_input, countX, countY, i+1, j+1);
			h_outputCpu[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
			float angle = std::atan2(Gx, Gy);  
            angle = angle * (360.0 / (2.0 * PI));  
            if (angle < 0)
				grad[getIndexGlobal(countX, i, j)] = angle+180;
			else
				grad[getIndexGlobal(countX, i, j)] = angle;

		}
	}
}

void create_filter(float kernal[][5])
{
	// Source:https://www.geeksforgeeks.org/gaussian-filter-generation-c/
    float sigma = 1.0;
    float r, s = 2.0 * sigma * sigma; 
    float sum = 0.0;
 
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            r = sqrt(x * x + y * y);
            kernal[x + 2][y + 2] = (exp(-(r * r) / s)) / (PI * s);
            sum += kernal[x + 2][y + 2];
        }
    }
 
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            kernal[i][j] /= sum;
            

}

void correctness(std::vector<float>& h_outputCpu,std::vector<float>& h_outputGpu, std::size_t countX, std::size_t countY)
{

	std::size_t errorCount = 0;
		for (size_t i = 0; i < countX; i = i + 1) { //loop in the x-grad
			for (size_t j = 0; j < countY; j = j + 1) { //loop in the y-grad
				size_t index = i + j * countX;
				// Allow small differences between CPU and GPU results (due to different rounding behavior)
				if (!(std::abs (h_outputCpu[index] - h_outputGpu[index]) <= 1e-5)) {
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
			return;
		}

		std::cout << std::endl;

	std::cout << "Success" << std::endl;
}

void guassianHost(const std::vector<float>& h_input, std::vector<float>& h_outputCpu, float *GKernel,int filter_size, int countX, int countY){
	
    float sum=0.0;
    for (int i = 0; i < countX; i++) {
		for (int j = 0; j <  countY; j++) {
		size_t pos = getIndexGlobal(countX, i, j);
		 for (int i_ = 0; i_ < filter_size; i_++)
            for (int j_ = 0; j_ < filter_size; j_++)
				sum += GKernel[i_*filter_size+j_] * getValueGlobal(h_input,countX,countY,(i_+i+-1), (j_+j+-1));
		if (sum < 0.0)
			h_outputCpu[getIndexGlobal(countX, i, j)] = 0.0;
		else if (sum>255.0)
			h_outputCpu[getIndexGlobal(countX, i, j)] = 255.0;
		else
			h_outputCpu[getIndexGlobal(countX, i, j)] = sum;
		sum=0.0;

	}}
}

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


	// Load the source code..............................................................................................................................................................................................
	cl::Program program = OpenCL::loadProgramSource(context, "../../../src/canny.cl");
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	// Declare some values
	std::size_t wgSizeX = 16; // Number of work items per work group in X grad
	std::size_t wgSizeY = 16;
	std::size_t countX = wgSizeX * 40; // Overall number of work items in X grad = Number of elements in X grad
	std::size_t countY = wgSizeY * 30;
	std::size_t count = countX * countY; // Overall number of elements
	std::size_t size = count * sizeof (float); // Size of data in bytes

	// Allocate space for output data from CPU and GPU on the host......................................................................................................................................................
	Core::TimeSpan gaussCPU(0.0),gaussGPU(0.0),sobelCPU(0),sobelGPU(0),non_max_CPU(0),non_max_GPU(0),double_threshold_CPU(0),double_threshold_GPU(0),hysteris_CPU(0.0),hysteris_GPU(0);
	Core::TimeSpan time_gaussian(0.0), time_sobel(0.0), time_non_max(0.0),time_double_threshold(0.0), time_hystereis(0.0);

	std::vector<float> h_input (count);
	std::vector<float> h_outputCpu (count);
	std::vector<float> h_outputGpu (count);
	std::vector<float>sobel_output(count);
	std::vector<float>gauss_kernal_output(count);
	

	std::vector<float>sobel_grad_cpu(count);
	std::vector<float>non_max(count);
	
	cl::Buffer max_const(context,CL_MEM_READ_ONLY, 4);
	cl::Buffer d_gaussian(context,CL_MEM_WRITE_ONLY, (25 * sizeof(float))); // allocating 5x5 filter 

	cl::Image2D d_image(context,CL_MEM_READ_ONLY, cl::ImageFormat(CL_R,CL_FLOAT),countX, countY);
	
	cl::Buffer buffer1(context,CL_MEM_READ_WRITE, size);
	cl::Buffer buffer2(context,CL_MEM_WRITE_ONLY, size);
	cl::Buffer buffer3(context,CL_MEM_WRITE_ONLY, size);
	
	cl::size_t<3> origin;
	origin[0] = origin[1] = origin[2] = 0;
	cl::size_t<3> region;
	region[0] = countX;
	region[1] = countY;
	region[2] = 1;
	memset(h_input.data(), 255, size);
	memset(h_outputGpu.data(), 255, size);
	
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
	
	
	Core::TimeSpan beg = Core::getCurrentTime();
	Core::TimeSpan end = Core::getCurrentTime();
	
    //#################################               #############################
    //#################################   GPU PORTION #############################
    //#################################               #############################


    
    memset(h_outputGpu.data(), 255, size);
    memset(h_outputGpu.data(), 255, size);

	beg = Core::getCurrentTime();

    int rows = 5, cols = 5;

    float GKernel[5][5];
    create_filter(GKernel);
	float * to_1D = GKernel[0];
    guassianHost(h_input, gauss_kernal_output, to_1D, rows,countX, countY);
	//Core::TimeSpan 
	end = Core::getCurrentTime();
	gaussCPU = end-beg;
	
	cl::Event cl_stream_time;
	queue.enqueueWriteBuffer(buffer1, true, 0, size, h_input.data(),NULL,&cl_stream_time);
	cl_stream_time.wait();
	time_gaussian = OpenCL::getElapsedTime(cl_stream_time);
	queue.enqueueWriteBuffer(d_gaussian, true, 0, (rows*cols)*sizeof(float), to_1D, NULL,&cl_stream_time);
	cl_stream_time.wait();
	Core::TimeSpan temp = OpenCL::getElapsedTime(cl_stream_time);
	time_gaussian = time_gaussian + temp;


	cl::Kernel gaussian_kernal(program, "gaussiankernal_device");
	gaussian_kernal.setArg(0, buffer1);
	gaussian_kernal.setArg(1, buffer2);
	gaussian_kernal.setArg(2, d_gaussian);

	cl::Event kernalExcTime;

	queue.enqueueNDRangeKernel(gaussian_kernal,0, cl::NDRange(countX, countY),cl::NDRange(wgSizeX, wgSizeY),NULL,&kernalExcTime);
	queue.finish();
	gaussGPU = OpenCL::getElapsedTime(kernalExcTime);

	queue.enqueueReadBuffer(buffer2,true, 0,  size, h_outputGpu.data(),NULL,&cl_stream_time);
	cl_stream_time.wait();
	temp = OpenCL::getElapsedTime(cl_stream_time);
	time_gaussian = time_gaussian + temp;
	
    Core::writeImagePGM("gaussian_gpu.pgm", h_outputGpu, countX, countY);

	correctness(gauss_kernal_output,h_outputGpu,countX, countY);
   
	
	beg = Core::getCurrentTime();
	sobelHost(gauss_kernal_output, sobel_output, sobel_grad_cpu, countX, countY);
	end = Core::getCurrentTime();

	sobelCPU = end-beg;

	Core::writeImagePGM("sobel_with_gaussian.pgm", sobel_output, countX, countY);
	cl::Kernel sobel_kernal(program, "sobel_kernal");
	time_sobel = time_sobel + OpenCL::getElapsedTime(cl_stream_time);
	sobel_kernal.setArg(0, buffer2); 
	sobel_kernal.setArg(1, buffer1); 
	sobel_kernal.setArg(2, buffer3);
	
	queue.enqueueNDRangeKernel(sobel_kernal,0, cl::NDRange(countX, countY),cl::NDRange(wgSizeX, wgSizeY),NULL,&kernalExcTime);
	queue.finish();

	sobelGPU = OpenCL::getElapsedTime(kernalExcTime);

	queue.enqueueReadBuffer(buffer1,true, 0,  size, h_outputGpu.data(),NULL, &cl_stream_time);
	cl_stream_time.wait();
	time_sobel = time_sobel + OpenCL::getElapsedTime(cl_stream_time);
	time_sobel = time_sobel + OpenCL::getElapsedTime(cl_stream_time);
	correctness(sobel_output,h_outputGpu,countX, countY);
    Core::writeImagePGM("sobel_gpu.pgm", h_outputGpu, countX, countY);




	beg = Core::getCurrentTime();
	non_max_suppression(sobel_output,sobel_grad_cpu, non_max, countX, countY);
	Core::writeImagePGM("non_max_suppression.pgm", non_max, countX, countY);
	end = Core::getCurrentTime();
	non_max_CPU = end-beg;
	cl::Kernel non_max_supp(program, "non_max_suppression");

	
	non_max_supp.setArg(0, buffer1);
	non_max_supp.setArg(1, buffer3);
	non_max_supp.setArg(2, buffer2);
	
	queue.enqueueNDRangeKernel(non_max_supp,0, cl::NDRange(countX, countY),cl::NDRange(wgSizeX, wgSizeY),NULL,&kernalExcTime);
	queue.finish();
	
	non_max_GPU = OpenCL::getElapsedTime(kernalExcTime);
	queue.enqueueReadBuffer(buffer2,true, 0,  size, h_outputGpu.data(),NULL, &cl_stream_time);
	cl_stream_time.wait();
	time_non_max = time_non_max + OpenCL::getElapsedTime(cl_stream_time);
	
	Core::writeImagePGM("non_max_supp_gpu.pgm", h_outputGpu, countX, countY);
	correctness(non_max,h_outputGpu,countX, countY);


	memset(h_outputCpu.data(), 0, size);	
	beg = Core::getCurrentTime();
	double_theshold(non_max,h_outputCpu,0.02,0.07, countX, countY);
	end = Core::getCurrentTime();
	double_threshold_CPU = end-beg;
	Core::writeImagePGM("double_threshold.pgm", h_outputCpu, countX, countY);
	cl::Kernel double_thre(program, "double_threshold");
	time_double_threshold = time_double_threshold + OpenCL::getElapsedTime(cl_stream_time);
	cl_float max_ = *max_element(non_max.begin(),non_max.end());
	cl_float low = 0.02;
	cl_float high = 0.07;
	double_thre.setArg(0, buffer2);
	double_thre.setArg(1, buffer3);
	double_thre.setArg(2, max_);
	double_thre.setArg(3,low );
	double_thre.setArg(4, high);
	queue.enqueueNDRangeKernel(double_thre,0, cl::NDRange(countX, countY),cl::NDRange(wgSizeX, wgSizeY),NULL,&kernalExcTime);
	queue.finish();
	double_threshold_GPU = OpenCL::getElapsedTime(kernalExcTime);
	queue.enqueueReadBuffer(buffer3,true, 0,  size, h_outputGpu.data(),NULL, &cl_stream_time);
	time_double_threshold = time_double_threshold + OpenCL::getElapsedTime(cl_stream_time);
	Core::writeImagePGM("double_threshold_gpu.pgm", h_outputGpu, countX, countY);
	correctness(h_outputCpu,h_outputGpu,countX, countY);
	
	
	
	beg = Core::getCurrentTime();
	hysteris(h_outputCpu,0,0.004, countX, countY);
	Core::writeImagePGM("hysteris_cpu.pgm", h_outputCpu, countX, countY);
	end = Core::getCurrentTime();
	hysteris_CPU = end-beg;
	cl::Kernel hysteris_gpu(program, "hysterisis");
	time_hystereis = OpenCL::getElapsedTime(cl_stream_time);
	cl_float weak = 75;
	cl_float strong = 255;
	hysteris_gpu.setArg(0, buffer3);
	hysteris_gpu.setArg(1, weak);
	hysteris_gpu.setArg(2, strong);
	queue.enqueueNDRangeKernel(hysteris_gpu,0, cl::NDRange(countX, countY),cl::NDRange(wgSizeX, wgSizeY),NULL,&kernalExcTime);
	queue.finish();
	hysteris_GPU = OpenCL::getElapsedTime(kernalExcTime);
	queue.enqueueReadBuffer(buffer3,true, 0,  size, h_outputGpu.data(),NULL, &cl_stream_time);
	cl_stream_time.wait();
	time_hystereis = time_hystereis + OpenCL::getElapsedTime(cl_stream_time);
	Core::writeImagePGM("hysteris_gpu.pgm", h_outputGpu, countX, countY);
	correctness(h_outputCpu,h_outputGpu,countX, countY);    
    
	

	std::cout << "gaussian cpu time: " << gaussCPU.toString() << std::endl;
	std::cout << "gaussian gpu time  without i/o: " << gaussGPU.toString() << std::endl;
	std::cout << "gaussian gpu time  with i/o: " << (time_gaussian+gaussGPU).toString() << std::endl;
	std::cout << "sobel cpu time:" << sobelCPU.toString() << std::endl;
	std::cout << "sobel gpu time  without i/o: " << sobelGPU.toString() << std::endl;
	std::cout << "sobel gpu time  with i/o: " << (time_sobel+sobelGPU).toString() << std::endl;
	std::cout << "non-max cpu time:" << non_max_CPU.toString() << std::endl;
	std::cout << "non-max gpu time without i/o:" << non_max_GPU.toString() << std::endl;
	std::cout << "non-max gpu time with i/o: " << (time_non_max+non_max_GPU).toString() << std::endl;
	std::cout << "double-threshold cpu time:" << double_threshold_CPU.toString() << std::endl;
	std::cout << "double-threshold gpu time  without i/o:" << double_threshold_GPU.toString() << std::endl;
	std::cout << "double-threshold gpu time  with i/o:" << (time_double_threshold+double_threshold_GPU).toString() << std::endl;
	std::cout << "hysteris cpu time:" << hysteris_CPU.toString() << std::endl;
	std::cout << "hysteris gpu time without i/o:" << hysteris_GPU.toString() << std::endl;
	std::cout << "hysteris gpu time with i/o:" << (time_hystereis+hysteris_GPU).toString() << std::endl;
	Core::TimeSpan total_cpu  = gaussCPU+sobelCPU+non_max_CPU+double_threshold_CPU+hysteris_CPU;
	Core::TimeSpan total_gpu = gaussGPU+sobelGPU+non_max_GPU+double_threshold_GPU+hysteris_GPU;
	Core::TimeSpan total_gpu_io = total_gpu + time_gaussian + time_sobel + time_non_max + time_double_threshold + time_hystereis;
	std::cout<<"Canny Edge Detection CPU Time:"<<total_cpu<<endl;
	std::cout<<"Canny Edge Detection GPU Time without i/o:"<<total_gpu<<endl;
	std::cout<<"Canny Edge Detection GPU Time with i/o:"<<total_gpu_io<<endl;
	std::cout << "GPU speedup without i/o = " << (total_cpu.getSeconds() / total_gpu.getSeconds()) << std::endl;
	std::cout << "GPU speedup with i/o = " << (total_cpu.getSeconds() / total_gpu_io.getSeconds()) << std::endl;

	}