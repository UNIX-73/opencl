/*

This program will numerically compute the integral of

                  4/(1+x*x)

from 0 to 1.  The value of this integral is pi -- which
is great since it gives us an easy way to check the answer.

The is the original sequential program.  It uses the timer
from the OpenMP runtime library

History: Written by Tim Mattson, 11/99.

*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif
#include <err_code.h>

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_GPU
#endif

#define N_STEPS 100000000

#define STEPS_PER_THREAD 1000
#define N_THREADS (N_STEPS / STEPS_PER_THREAD)
#define LOCAL_THREADS 100

extern int   output_device_info(cl_device_id);
extern char* load_kernel_source(const char* filename, size_t* size);
char*        getKernelSource(char* filename);

double        step;
extern double wtime();

#define assert(condition, msg)                                                                     \
    do                                                                                             \
    {                                                                                              \
        if (!(condition))                                                                          \
        {                                                                                          \
            printf(msg);                                                                           \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

int main()
{

    unsigned int i          = 0;
    size_t       num_groups = N_THREADS / LOCAL_THREADS;

    double* h_reduction = (double*)calloc(num_groups, sizeof(double));
    cl_mem  d_reduction; // device memory used for the reduction

    step                      = 1.0 / (double)N_STEPS;
    unsigned int steps_per_th = STEPS_PER_THREAD;
    const size_t local_th     = LOCAL_THREADS;

    cl_int err;

    cl_device_id     device_id;
    cl_context       context;
    cl_command_queue commands;
    cl_program       program;
    cl_kernel        ko_vadd;

    cl_uint numPlatforms;

    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkError(err, "Finding platforms");
    assert(numPlatforms != 0, "no platforms found");

    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    checkError(err, "Getting platforms");

    for (i = 0; i < numPlatforms; i++)
    {
        err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
        if (err == CL_SUCCESS)
            break;
    }
    assert(device_id != NULL, "no device could be selected");

    size_t max_local_size;
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_local_size),
                          &max_local_size, NULL);
    checkError(err, "getting max local size");
    assert(local_th < max_local_size, "reduce max local threads, not supported by the device");

    err = output_device_info(device_id);
    checkError(err, "Printing device output");

    // Create a compute context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    checkError(err, "Creating context");

    // Create a command queue
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    checkError(err, "Creating command queue");

    // Create the compute program from the source buffer
    size_t _size;
    char*  source = getKernelSource("pi.cl");
    program       = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &err);
    checkError(err, "Creating program");

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char   buffer[4096];

        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer,
                              &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    ko_vadd = clCreateKernel(program, "pi", &err);
    checkError(err, "Creating kernel");

    d_reduction =
        clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * num_groups, NULL, &err);
    checkError(err, "Creating buffer d_reduction");

    // TODO: checkear memcpy

    int arg_index = 0;

    err = clSetKernelArg(ko_vadd, arg_index++, sizeof(double), &step);
    err |= clSetKernelArg(ko_vadd, arg_index++, sizeof(unsigned int), &steps_per_th);
    err |= clSetKernelArg(ko_vadd, arg_index++, sizeof(double) * local_th, NULL);
    err |= clSetKernelArg(ko_vadd, arg_index++, sizeof(cl_mem), &d_reduction);
    checkError(err, "Setting kernel arguments");

    double              rtime     = wtime();
    const unsigned long n_threads = N_THREADS;
    err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &n_threads, &local_th, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel");

    // Wait for the commands to complete before stopping the timer
    err = clFinish(commands);
    checkError(err, "Waiting for kernel to finish");

    rtime = wtime() - rtime;
    printf("\nThe kernel ran in %lf seconds\n", rtime);

    err = clEnqueueReadBuffer(commands, d_reduction, CL_TRUE, 0, sizeof(double) * num_groups,
                              h_reduction, 0, NULL, NULL);
    assert(err == CL_SUCCESS, "Error: Failed to read output array!");

    double sum = 0.0;
    for (size_t i = 0; i < num_groups; i++)
    {
        sum += h_reduction[i];
    }
    sum /= N_THREADS * STEPS_PER_THREAD;

    double error = fabs(M_PI - sum);
    printf("Result:\tpi = %.12f, error = %.3e\n", sum, error);
}
