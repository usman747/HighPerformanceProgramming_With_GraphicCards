#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

//TODO
__kernel void mandelbrotKernel(__global uint*d_output, uint niter, float xmin, float xmax, float ymin, float ymax)
{
  
//void mandelbrotHost (std::vector<cl_uint>& h_output, size_t countX, size_t countY, cl_uint niter, float xmin, float xmax, float ymin, float ymax) // in C++ CPU, this was converted
   
//"size_t countX" will not be a parameter here because there is no loop here (in the x-direction) is parralelized 
//"size_t countY" will not be a parameter here because there is no loop here (in the y-direction) is parralelized


    size_t countX = get_global_size(0);   //Size of the work item = Limit of the x-dir loop converted in GPU
    size_t countY = get_global_size(1);   //Size of the work item = Limit of the y-dir loop converted in GPU

    size_t i = get_global_id(0);        // Get global index of the current work item in the x-direction //Implementation of loop in the x-direction when in GPU   
    size_t j = get_global_id(1);        // Get global index of the current work item in the y-direction //Implementation of loop in the y-direction when in GPU

    float xc = xmin + (xmax - xmin) / (countX - 1) * i; //xc=real(c)
    float yc = ymin + (ymax - ymin) / (countY - 1) * j; //yc=imag(c)
    float x = 0.0; //x=real(z_k)
    float y = 0.0; //y=imag(z_k)
			for (size_t k = 0; k < niter; k = k + 1) //iteration loop
			{ 
				float tempx = x * x - y * y + xc; //z_{n+1}=(z_n)^2+c;
				float tempy = 2 * x * y + yc;
				x = tempx;
				y= tempy;
				float r2 = x * x + y * y; //r2=|z_k|^2

				if ((r2 > 4) || k == niter - 1) //divergence condition
				{ 
					d_output[i + j * countX] = k;
					break;
		        }
			}
}

//-----------------------------------------------------------------------------------//
//OpenCL C	C++				Info
//char		cl_char			signed 8-bit integer
//uchar		cl_uchar		unsigned 8-bit integer
//short		cl_short		signed 16-bit integer
//ushort	cl_ushort		unsigned 16-bit integer
//int		cl_int			signed 32-bit integer
//uint		cl_uint			unsigned 32-bit integer
//long		cl_long			signed 64-bit integer
//ulong		cl_ulong		unsigned 64-bit integer
//float		cl_float		32-bit float
//double	cl_double		64-bit float
//bool 		-				boolean value (true or false)
//size_t 	-				pointer-sized unsigned integer
//__global T* cl::Buffer	pointer to data in global memory
//-----------------------------------------------------------------------------------//