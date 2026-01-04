//------------------------------------------------------------------------------
//
// Name:       vadd.c
//
// Purpose:    Elementwise addition of two vectors (c = a + b)
//
// HISTORY:    Written by Tim Mattson, December 2009
//             Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated by Tom Deakin, July 2013
//             Updated by Tom Deakin, October 2014
//
//------------------------------------------------------------------------------

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"

// pick up device type from compiler command line or from
// the default type
#define CPU CL_DEVICE_TYPE_CPU
#define GPU CL_DEVICE_TYPE_GPU

extern double wtime(); // returns time since some fixed past point (wtime.c)
extern int    output_device_info(cl_device_id);

//------------------------------------------------------------------------------

#define TOL (0.001)     // tolerance used in floating point comparisons
#define LENGTH (102400) // length of vectors a, b, and c

#define CPU_PERCENTAGE 0.001

#define GPU_LEN (size_t)((double)LENGTH * (double)CPU_PERCENTAGE)
#define CPU_LEN (LENGTH - GPU_LEN)

_Static_assert(GPU_LEN <= LENGTH, "cannot set a higher value than len");

//------------------------------------------------------------------------------
//
// kernel:  vadd
//
// Purpose: Compute the elementwise sum c = a+b
//
// input: a and b float vectors of length count
//
// output: c float vector of length count holding the sum a + b
//

const char* KernelSource =
    "\n"
    "__kernel void vadd(                                                 \n"
    "   __global float* a,                                                  \n"
    "   __global float* b,                                                  \n"
    "   __global float* c,                                                  \n"
    "   const unsigned int count)                                           \n"
    "{                                                                      \n"
    "   int i = get_global_id(0);                                           \n"
    "   if(i < count)                                                       \n"
    "       c[i] = a[i] + b[i];                                             \n"
    "}                                                                      \n"
    "\n";

//------------------------------------------------------------------------------

int main(int argc, char** argv)
{

    int err; // error code returned from OpenCL calls

    float* h_a = (float*)calloc(LENGTH, sizeof(float)); // a vector
    float* h_b = (float*)calloc(LENGTH, sizeof(float)); // b vector
    float* h_c =
        (float*)calloc(LENGTH, sizeof(float)); // c vector (a+b) returned from the compute device

    unsigned int correct; // number of correct results

    cl_event ev_cpu, ev_gpu;

    cl_context context0;
    cl_context context1;

    cl_command_queue queue_0;
    cl_command_queue queue_1;
    cl_program       program0; // compute program
    cl_program       program1; // compute program

    cl_kernel ko_vadd0; // compute kernel
    cl_kernel ko_vadd1; // compute kernel

    cl_mem d_a0, d_a1;
    cl_mem d_b0, d_b1;
    cl_mem d_c0, d_c1;

    // Fill vectors a and b with random float values
    int i = 0;
    for (i = 0; i < LENGTH; i++)
    {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // Set up platform and GPU device
    cl_uint num_platforms;

    // Find number of platforms
    err = clGetPlatformIDs(0, NULL, &num_platforms);

    checkError(err, "Finding platforms");
    if (num_platforms == 0)
    {
        printf("Found 0 platforms!\n");
        return EXIT_FAILURE;
    }

    // Get all platforms
    cl_platform_id platforms[num_platforms];
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    checkError(err, "Getting platforms");

    // Secure a GPU
    cl_device_id   gpu          = NULL;
    cl_platform_id gpu_platform = NULL;

    for (cl_uint i = 0; i < num_platforms; i++)
    {
        cl_int err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &gpu, NULL);

        if (err == CL_SUCCESS)
        {
            gpu_platform = platforms[i];
            break;
        }
    }
    // Secure a CPU
    cl_device_id   cpu          = NULL;
    cl_platform_id cpu_platform = NULL;

    for (cl_uint i = 0; i < num_platforms; i++)
    {
        cl_int err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 1, &cpu, NULL);

        if (err == CL_SUCCESS)
        {
            cpu_platform = platforms[i];
            break;
        }
    }

    if (!gpu || !cpu)
    {
        fprintf(stderr, "no cpu or gpu devices found 2 devices needed for this scheduling\n");
        exit(1);
    }

    err = output_device_info(gpu);
    checkError(err, "Printing device output");
    err = output_device_info(cpu);
    checkError(err, "Printing device output");

    // Create a compute context

    context0 = clCreateContext(NULL, 1, &cpu, NULL, NULL, &err);
    checkError(err, "Creating context");
    context1 = clCreateContext(NULL, 1, &gpu, NULL, NULL, &err);
    checkError(err, "Creating context");

    // Create both queues
    queue_0 = clCreateCommandQueue(context0, cpu, CL_QUEUE_PROFILING_ENABLE, &err);
    checkError(err, "Creating command queue0");
    queue_1 = clCreateCommandQueue(context1, gpu, CL_QUEUE_PROFILING_ENABLE, &err);
    checkError(err, "Creating command queue1");

    // Create the compute program from the source buffer
    program0 = clCreateProgramWithSource(context0, 1, (const char**)&KernelSource, NULL, &err);
    checkError(err, "Creating program");

    program1 = clCreateProgramWithSource(context1, 1, (const char**)&KernelSource, NULL, &err);
    checkError(err, "Creating program");

    // Build the program
    err = clBuildProgram(program0, 1, &cpu, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char   buffer[2048];

        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program0, cpu, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }
    err = clBuildProgram(program1, 1, &gpu, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char   buffer[2048];

        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program1, gpu, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    // Create the compute kernel from the program
    ko_vadd0 = clCreateKernel(program0, "vadd", &err);
    checkError(err, "Creating kernel");
    ko_vadd1 = clCreateKernel(program1, "vadd", &err);
    checkError(err, "Creating kernel");

    // Create the input (a, b) and output (c) arrays in device memory
    // DEVICE 0
    size_t len0  = CPU_LEN;
    size_t size0 = sizeof(float) * CPU_LEN;
    d_a0         = clCreateBuffer(context0, CL_MEM_READ_ONLY, size0, NULL, &err);
    checkError(err, "Creating buffer d_a0");
    d_b0 = clCreateBuffer(context0, CL_MEM_READ_ONLY, size0, NULL, &err);
    checkError(err, "Creating buffer d_b0");
    d_c0 = clCreateBuffer(context0, CL_MEM_WRITE_ONLY, size0, NULL, &err);
    checkError(err, "Creating buffer d_c0");

    err = clEnqueueWriteBuffer(queue_0, d_a0, CL_TRUE, 0, size0, h_a, 0, NULL, NULL);
    checkError(err, "Copying h_a to device at d_a");
    err = clEnqueueWriteBuffer(queue_0, d_b0, CL_TRUE, 0, size0, h_b, 0, NULL, NULL);
    checkError(err, "Copying h_b to device at d_b");

    err = clSetKernelArg(ko_vadd0, 0, sizeof(cl_mem), &d_a0);
    err |= clSetKernelArg(ko_vadd0, 1, sizeof(cl_mem), &d_b0);
    err |= clSetKernelArg(ko_vadd0, 2, sizeof(cl_mem), &d_c0);
    err |= clSetKernelArg(ko_vadd0, 3, sizeof(unsigned int), &len0);

    // DEVICE 1
    size_t len1  = GPU_LEN;
    size_t size1 = sizeof(float) * GPU_LEN;
    d_a1         = clCreateBuffer(context1, CL_MEM_READ_ONLY, size1, NULL, &err);
    checkError(err, "Creating buffer d_a1");
    d_b1 = clCreateBuffer(context1, CL_MEM_READ_ONLY, size1, NULL, &err);
    checkError(err, "Creating buffer d_b1");
    d_c1 = clCreateBuffer(context1, CL_MEM_WRITE_ONLY, size1, NULL, &err);
    checkError(err, "Creating buffer d_c1");

    // I get it from the part where the cpu ends
    float* ha_1_start = h_a + len0;
    float* hb_1_start = h_b + len0;
    err = clEnqueueWriteBuffer(queue_1, d_a1, CL_TRUE, 0, size1, ha_1_start, 0, NULL, NULL);
    checkError(err, "Copying h_a to device at d_a");
    err = clEnqueueWriteBuffer(queue_1, d_b1, CL_TRUE, 0, size1, hb_1_start, 0, NULL, NULL);
    checkError(err, "Copying h_b to device at d_b");

    err = clSetKernelArg(ko_vadd1, 0, sizeof(cl_mem), &d_a1);
    err |= clSetKernelArg(ko_vadd1, 1, sizeof(cl_mem), &d_b1);
    err |= clSetKernelArg(ko_vadd1, 2, sizeof(cl_mem), &d_c1);
    err |= clSetKernelArg(ko_vadd1, 3, sizeof(unsigned int), &len1);

    checkError(err, "Setting kernel arguments");

    // Execute the kernel over the entire range of our 1d input data set
    // letting the OpenCL runtime choose the work-group size
    size_t global0 = len0;
    size_t global1 = len1;

    err = clEnqueueNDRangeKernel(queue_0, ko_vadd0, 1, NULL, &global0, NULL, 0, NULL, &ev_cpu);
    checkError(err, "Enqueueing kernels");

    err = clEnqueueNDRangeKernel(queue_1, ko_vadd1, 1, NULL, &global1, NULL, 0, NULL, &ev_gpu);
    checkError(err, "Enqueueing kernels");

    // Wait for the commands to complete before stopping the timer
    err = clWaitForEvents(1, &ev_cpu);
    checkError(err, "Waiting for kernel events to finish");
    err = clWaitForEvents(1, &ev_gpu);
    checkError(err, "Waiting for kernel events to finish");

    err = clFinish(queue_0);
    checkError(err, "Waiting for kernel to finish");
    err = clFinish(queue_1);
    checkError(err, "Waiting for kernel to finish");

    size_t cpu_start, cpu_end, gpu_start, gpu_end;

    err = clGetEventProfilingInfo(ev_gpu, CL_PROFILING_COMMAND_START, sizeof(gpu_start), &gpu_start,
                                  NULL);
    checkError(err, "Getting event info");
    err =
        clGetEventProfilingInfo(ev_gpu, CL_PROFILING_COMMAND_END, sizeof(gpu_end), &gpu_end, NULL);
    checkError(err, "Getting event info");

    err = clGetEventProfilingInfo(ev_cpu, CL_PROFILING_COMMAND_START, sizeof(cpu_start), &cpu_start,
                                  NULL);
    checkError(err, "Getting event info");
    err =
        clGetEventProfilingInfo(ev_cpu, CL_PROFILING_COMMAND_END, sizeof(cpu_end), &cpu_end, NULL);
    checkError(err, "Getting event info");

    double cpu_time = (double)(cpu_end - cpu_start) * 1e-6;
    double gpu_time = (double)(gpu_end - gpu_start) * 1e-6;

    printf("CPU(%ld): %fms\nGPU(%ld): %fms\n", CPU_LEN, cpu_time, GPU_LEN, gpu_time);

    // Read back the results from the compute device

    err = clEnqueueReadBuffer(queue_0, d_c0, CL_TRUE, 0, size0, h_c, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array!\n%s\n", err_code(err));
        exit(1);
    }

    float* hc_1_start = h_c + len0;
    err = clEnqueueReadBuffer(queue_1, d_c1, CL_TRUE, 0, size1, hc_1_start, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array!\n%s\n", err_code(err));
        exit(1);
    }

    // Test the results
    correct = 0;
    float tmp;

    for (i = 0; i < LENGTH; i++)
    {
        tmp = h_a[i] + h_b[i];     // assign element i of a+b to tmp
        tmp -= h_c[i];             // compute deviation of expected and output result
        if (tmp * tmp < TOL * TOL) // correct if square deviation is less than tolerance squared
            correct++;
        else
        {
            printf(" tmp %f h_a %f h_b %f h_c %f \n", tmp, h_a[i], h_b[i], h_c[i]);
        }
    }

    // summarise results
    printf("C = A+B:  %d out of %d results were correct.\n", correct, LENGTH);

    // cleanup then shutdown
    clReleaseMemObject(d_a0);
    clReleaseMemObject(d_b0);
    clReleaseMemObject(d_c0);
    clReleaseProgram(program0);
    clReleaseKernel(ko_vadd0);
    clReleaseCommandQueue(queue_0);
    clReleaseContext(context0);

    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}
