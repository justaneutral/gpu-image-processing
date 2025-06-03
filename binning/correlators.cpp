// 2D convolutions with very large kernel sizes can be efficiently implemented using FFT transformations.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

// Helper functions for CUDA
#include "nvidiaCUDA90common\inc\helper_functions.h"
#include "nvidiaCUDA90common\inc\helper_cuda.h"

#include "convolutionFFT2D_common.h"

#include "prototypes.h"


CImg<ivt> correlators(CImg<ivt> *Data, CImg<ivt> *Kernel, const int printlevel, double& gpuTime,double& datatransfertime)
{
	ivt *h_Data = (ivt *)(&(Data->at(0)));
	const int dataH = Data->height();
	const int dataW = Data->width();
	ivt *h_Kernel = (ivt *)(&(Kernel->at(0)));
	const int kernelH = Kernel->height();
	const int kernelW = Kernel->width();
	const int kernelY = kernelH;/////-2;
	const int kernelX = kernelW;/////-2;
	const int fftH = snapTransformSize(dataH + kernelH - 1);
	const int fftW = snapTransformSize(dataW + kernelW - 1);
	CImg<ivt> ResultGPU;
	ResultGPU.assign(fftH,fftW);
	ivt *h_ResultGPU = (ivt *)(&(ResultGPU.at(0)));

	ivt *d_Data, *d_PaddedData, *d_Kernel, *d_PaddedKernel;
	fComplex *d_DataSpectrum, *d_KernelSpectrum;
	cufftHandle fftPlanFwd, fftPlanInv;

    StopWatchInterface *hTimer = NULL;
    sdkCreateTimer(&hTimer);

	if(printlevel>1)
	{
		printf("Testing built-in R2C / C2R FFT-based convolution\n");
		printf(" checking and allocating memory\n");
	}
	if(!h_Data || !h_Kernel || !h_ResultGPU)
		return ResultGPU;

    checkCudaErrors(cudaMalloc((void **)&d_Data,   dataH   * dataW   * sizeof(ivt)));
    checkCudaErrors(cudaMalloc((void **)&d_Kernel, kernelH * kernelW * sizeof(ivt)));
	checkCudaErrors(cudaMalloc((void **)&d_PaddedData,   fftH * fftW * sizeof(ivt)));
    checkCudaErrors(cudaMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(ivt)));
	checkCudaErrors(cudaMalloc((void **)&d_DataSpectrum,   fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    checkCudaErrors(cudaMalloc((void **)&d_KernelSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
	if(printlevel>1) printf("...creating R2C & C2R FFT plans for %i x %i\n", fftH, fftW);
    checkCudaErrors(cufftPlan2d(&fftPlanFwd, fftH, fftW, CUFFT_R2C));
    checkCudaErrors(cufftPlan2d(&fftPlanInv, fftH, fftW, CUFFT_C2R));
    checkCudaErrors(cudaMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(ivt)));
    checkCudaErrors(cudaMemset(d_PaddedData,   0, fftH * fftW * sizeof(ivt)));

    if(printlevel>1) printf("...uploading to GPU and padding convolution kernel and input data\n");
    checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel, kernelH * kernelW * sizeof(ivt), cudaMemcpyHostToDevice));
    padKernel(d_PaddedKernel,d_Kernel,fftH,fftW,kernelH,kernelW,kernelY,kernelX);
	
	

    //Not including kernel transformation into time measurement,
    //since convolution kernel is not changed very frequently
    if(printlevel>1) printf("...transforming convolution kernel\n");
    checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedKernel, (cufftComplex *)d_KernelSpectrum));

    if(printlevel>1) printf("...running GPU FFT convolution: ");
    checkCudaErrors(cudaDeviceSynchronize());

	size_t datatransfersize = dataH*dataW*sizeof(ivt);

    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
	checkCudaErrors(cudaMemcpy(d_Data, h_Data, datatransfersize, cudaMemcpyHostToDevice));
	sdkStopTimer(&hTimer);
    datatransfertime = sdkGetTimerValue(&hTimer);

    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
	padDataClampToBorder(d_PaddedData,d_Data,fftH,fftW,dataH,dataW,kernelH,kernelW,kernelY,kernelX);
	checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedData, (cufftComplex *)d_DataSpectrum));
    modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftH, fftW, 1);
    checkCudaErrors(cufftExecC2R(fftPlanInv, (cufftComplex *)d_DataSpectrum, (cufftReal *)d_PaddedData));

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer);
    if(printlevel>1)
	{
		printf("%f MPix/s (%f ms)\n", (double)dataH * (double)dataW * 1e-6 / (gpuTime * 0.001), gpuTime);
		printf("...reading back GPU convolution results\n");
	}

	size_t transfersize = fftH * fftW * sizeof(ivt);
	sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    checkCudaErrors(cudaMemcpy(h_ResultGPU, d_PaddedData, transfersize, cudaMemcpyDeviceToHost));
	sdkStopTimer(&hTimer);
    //datatransfertime = sdkGetTimerValue(&hTimer);
	
    sdkDeleteTimer(&hTimer);

    checkCudaErrors(cufftDestroy(fftPlanInv));
    checkCudaErrors(cufftDestroy(fftPlanFwd));

    checkCudaErrors(cudaFree(d_DataSpectrum));
    checkCudaErrors(cudaFree(d_KernelSpectrum));
    checkCudaErrors(cudaFree(d_PaddedData));
    checkCudaErrors(cudaFree(d_PaddedKernel));
    checkCudaErrors(cudaFree(d_Data));
    checkCudaErrors(cudaFree(d_Kernel));

    return ResultGPU;
}
