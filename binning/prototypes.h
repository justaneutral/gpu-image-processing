#pragma once


#define NumSums (3)
#define NumCorrsW (1)
#define NumCorrsH (1)
#define NumCorrs (NumCorrsW*NumCorrsH)
#define NumSumsTotal (NumSums*NumCorrs)

#define MAXSTEPVAL (1)


#include <iostream>
#include "listfiles.h"

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned char ivti;
typedef float ivt;

// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

// Helper functions for CUDA
#include "nvidiaCUDA90common\inc\helper_functions.h"
#include "nvidiaCUDA90common\inc\helper_cuda.h"

//#include "convolutionFFT2D_common.h"

//#include <omp.h>
//#define cimg_use_openmp
#include "Cimg.h"

using namespace std;
using namespace cimg_library;

//template<typename T> int calculate_gradmag(CImg<T> &dst, CImg<T> &src);
//int calculate_gradmag(CImg<ivt> &dst, CImg<ivt> &src);

#include "chError.h"
#include "chAssert.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#define INTCEIL(a,b) ( ((a)+(b)-1) / (b) )

//int convolutionalFFT2Dmain(int argc, char **argv);

extern "C" void gpucode(ivti *dd,ivti *kd, ivt *cd, int h, int w, int ch, int cw);
//__global__ void gpucode_kernel(ivt *dd,ivt *kd, ivt *cd, int h, int w, int ch, int cw);
//inline __device__ ivt gpucorrcoef(const ivt *dd,const ivt *kd, const int h, const int w, const int ch, const int cw, const int chi, const int cwi);

CImg<ivt> reverceimage(CImg<ivt> *im);
bool assignaperture(CImg<ivti> *im,const int apertureH,const int apertureW,const int centeroffsetH, const int centeroffsetW);
CImg<ivt>calculateimagedisplacement(ivt& offsety,ivt& offsetx,CImg<ivti>*img,CImg<ivti>*templateimage,const int H,const int W,const int printlevel,const bool fractional,CImgDisplay& disp_res_gpu,bool showdisplay,float& gpuTime,float& datatransfertime);
bool calcconvolution(ivt *h_Data, ivt *h_Kernel, ivt *h_ResultCPU, ivt *h_ResultGPU,const int kernelH, const int kernelW, const int kernelY, const int kernelX, const int dataH, const int dataW, const int fftH, const int fftW, const int printlevel);
CImg<ivt> calcconvolutiononcimg(CImg<ivti> *Data, CImg<ivti> *Kernel, const int H, const int W, const int printlevel, double& gpuTime,double& datatransfertime);
int imcorr(ivti *d,ivti *k, ivt *c, int h, int w, int ch, int cw);
CImg<float> mainncTT(CImg<unsigned  char> *idata,CImg<unsigned  char> *tdata,float& gpuTime,float& datatransfertime);
CImg<float> mainncTR(CImg<unsigned  char> *idata,CImg<unsigned  char> *tdata,float& gpuTime,float& datatransfertime);


typedef struct imholder_str
{
	CImgDisplay disp;
	CImg<unsigned char> img;
	//unsigned char *devdata;
	//unsigned char *hidata;
    //unsigned char *didata;
    //unsigned int HostPitch;
	//size_t DevicePitch;
    //int w;
	//int h;

	cudaArray *pArrayImage;
    //texture<unsigned char,2> *ptex;
	//cudaChannelFormatDesc desc;

} imholder, *pimholder;


//extern "C" void reduceSinglePass(pimholder pref, int threads, int blocks, float *d_odata, unsigned char *d_idata);
//extern "C" void reduceSinglePass(unsigned int sizex, unsigned int sizey, int threads, int blocks, float *d_odata);
//template <unsigned int blockSize, bool nIsPow2> __global__ void reduceSinglePass(const unsigned char *g_idata, float *g_odata, unsigned int n);
//extern "C" void reduce(int size, int threads, int blocks, const unsigned char *d_idata, float *d_odata);
//bool runTest(pimholder pref);


//void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads);
//extern "C" void reduceSinglePass(unsigned int sizex,unsigned int sizey, int threads, int blocks, float *d_odata, unsigned char *d_idata);