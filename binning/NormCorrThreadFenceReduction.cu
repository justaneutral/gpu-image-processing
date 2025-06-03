#include "prototypes.h"
#define VERSION_MAJOR (CUDART_VERSION/1000)
#define VERSION_MINOR (CUDART_VERSION%100)/10
#if CUDART_VERSION >= 2020
#include "NormCorrThreadFenceReduction_kernel.cuh"
#else
#pragma comment(user, "CUDA 2.2 is required to build for threadFenceReduction")
#endif
////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(pimholder pref);
void mainreduce(pimholder pref)
{
    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev;
	dev = findCudaDevice(1,NULL);
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
	printf("GPU Device supports SM %d.%d compute capability\n\n", deviceProp.major, deviceProp.minor);
	bool bTestResult = runTest(pref);
}

////////////////////////////////////////////////////////////////////////////////
//! Compute sum reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//!
//! @param data       pointer to input data
//! @param size       number of input data elements
////////////////////////////////////////////////////////////////////////////////
float ncreduceCPU(pimholder pref)
{
    float sum = (float)(pref->img.at(0)); //((float)(pref->img.at(0))-127.5)/8.0;
    float c = 0.0;

    for (int i = 1; i < pref->img.size(); i++)
    {
        float y = (float)(pref->img.at(i)); //((float)(pref->img.at(i))-127.5)/8.0; - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sum;
}

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}


////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the reduction
// We set threads / block to the minimum of maxThreads and n/2.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
    if (n == 1)
    {
        threads = 1;
        blocks = 1;
    }
    else
    {
        threads = (n < maxThreads*2) ? nextPow2(n / 2) : maxThreads;
        blocks = max(1, n / (threads * 2));
    }

    blocks = min(maxBlocks, blocks);
}

////////////////////////////////////////////////////////////////////////////////
// This function performs a reduction of the input data multiple times and
// measures the average reduction time.
////////////////////////////////////////////////////////////////////////////////
float ncbenchmarkReduce(pimholder pref,
                      int  numThreads,
                      int  numBlocks,
                      int  maxThreads,
                      int  maxBlocks,
                      StopWatchInterface *timer,
                      float *h_odata,
                      float *d_odata,
					  unsigned char *d_idata)
{
    float gpu_result = 0;
    cudaError_t error = (cudaError_t)0;
    gpu_result = 0;
    unsigned int retCnt = 0;
    error = setRetirementCount(retCnt);
    checkCudaErrors(error);
    cudaDeviceSynchronize();
    sdkStartTimer(&timer);
	getLastCudaError("Kernel execution failed");
	// execute the kernel
    reduceSinglePass(pref->img.width(),pref->img.height(), numThreads, numBlocks, d_odata,d_idata);
	// copy final sum from device to host
    error = cudaMemcpy(&gpu_result, d_odata, sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaErrors(error);
    // check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    return gpu_result;
}

////////////////////////////////////////////////////////////////////////////////
// The main function which runs the reduction test.
////////////////////////////////////////////////////////////////////////////////
bool runTest(pimholder pref)
{
    int size = pref->img.size(); // number of elements to reduce

    int maxThreads = 512;  // number of threads per block
    int maxBlocks = 1024;
    bool bTestResult = false;

    unsigned int bytes = size * sizeof(unsigned char);

    unsigned char *h_idata = (unsigned char *)(&(pref->img.at(0)));

    int numBlocks = 0;
    int numThreads = 0;
    getNumBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks, numThreads);

    // allocate mem for the result on host side
    float *h_odata = (float *) malloc(numBlocks*sizeof(float));

    printf("%d blocks\n", numBlocks);

    // allocate device memory and data
    unsigned char *d_idata = NULL; ////pd.debug - will take from texref
    float *d_odata = NULL;

    checkCudaErrors(cudaMalloc((void **) &d_idata, bytes));
    checkCudaErrors(cudaMalloc((void **) &d_odata, numBlocks*sizeof(float)));

    // copy data directly to device memory
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_odata, h_idata, numBlocks*sizeof(float), cudaMemcpyHostToDevice));
	////checkCudaErrors(cudaMemset(d_odata, 0, numBlocks*sizeof(float)));


    // warm-up
    ////reduce(size, numThreads, numBlocks, d_idata, d_odata);

    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);

    float gpu_result = 0;

    gpu_result = ncbenchmarkReduce(pref, numThreads, numBlocks, maxThreads, maxBlocks, timer, h_odata, d_odata, d_idata);

    float reduceTime = sdkGetAverageTimerValue(&timer);
    printf("Average time: %f ms\n", reduceTime);
    printf("Bandwidth:    %f GB/s\n\n", (size * sizeof(int)) / (reduceTime * 1.0e6));

    // compute reference solution
    float cpu_result = ncreduceCPU(pref);

    printf("GPU result = %0.12f\n", gpu_result);
    printf("CPU result = %0.12f\n", cpu_result);

    double threshold = 1e-8 * size;
    double diff = abs((double)gpu_result - (double)cpu_result);
    bTestResult = (diff < threshold);

    // cleanup
    sdkDeleteTimer(&timer);

    free(h_odata);
    //cudaFree(d_idata);
    cudaFree(d_odata);


    return bTestResult;
}
