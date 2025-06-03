#pragma ones
#include "prototypes.h"

#include "chError.h"
#include "chAssert.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#define INTCEIL(a,b) ( ((a)+(b)-1) / (b) )

texture<unsigned char, 2> texImage;
texture<unsigned char, 2> texTemplate;

const int maxTemplatePixels = 3072;
__constant__ int g_xOffset[maxTemplatePixels];
__constant__ int g_yOffset[maxTemplatePixels];
__constant__ unsigned char g_Tpix[maxTemplatePixels];
__constant__ float g_cPixels, g_SumT, g_fDenomExp;
//unsigned int g_cpuSumT, g_cpuSumTSq;

//const float fThreshold = 1e-3f;

__device__ __host__ inline float CorrelationValue( float SumI, float SumISq, float SumIT, float SumT, float cPixels, float fDenomExp )
{
    float Numerator = cPixels*SumIT - SumI*SumT;
    float Denominator = rsqrtf( (cPixels*SumISq - SumI*SumI)*fDenomExp );
    return Numerator * Denominator;
}

///#include "corrTexTexSums.cuh"
///#include "corrTexTex.cuh"

///#include "corrTexConstantSums.cuh"
///#include "corrTexConstant.cuh"


///extern __shared__ unsigned char LocalBlock[];

//#include "corrSharedSMSums.cuh"
//#include "corrSharedSM.cuh"

//#include "corrSharedSums.cuh"
//#include "corrShared.cuh"

//#include "corrShared4Sums.cuh"
//#include "corrShared4.cuh"

//int poffsetx[maxTemplatePixels];
//int poffsety[maxTemplatePixels];

#define TC
#ifdef TC
											 
__global__ void corrTexConstantSums_kernelTC(float *pCorr,size_t CorrPitch,int *pI,int *pISq,int *pIT,float cPixels,float& fDenomExp,int w,int h,int wTemplate,int hTemplate)
{
    size_t row = blockIdx.y*blockDim.y + threadIdx.y;
    size_t col = blockIdx.x*blockDim.x + threadIdx.x;

    // adjust pointers to row
    pCorr = (float *) ((char *) pCorr+row*CorrPitch);
    pI    = (int *) ((char *) pI  +row*CorrPitch);
    pISq  = (int *) ((char *) pISq+row*CorrPitch);
    pIT   = (int *) ((char *) pIT +row*CorrPitch);

    // No __syncthreads in this kernel, so we can early-out
    // without worrying about the effects of divergence.
    if ( col >= w || row >= h )
        return;

    int SumI = 0;
    int SumISq = 0;
    int SumIT = 0;

    int inx = 0;

    for ( int j = 0; j < hTemplate; j++ )
	{
        for ( int i = 0; i < wTemplate; i++ )
		{
            unsigned char I = tex2D(texImage,(float)col+i,(float)row+j);
            unsigned char T = g_Tpix[inx++];
            SumI += I;
            SumISq += I*I;
            SumIT += I*T;
        }
    }
    pCorr[col] = CorrelationValue(SumI,SumISq,SumIT,g_SumT,cPixels,fDenomExp);
    pI[col] = SumI;
    pISq[col] = SumISq;
    pIT[col] = SumIT;
}

void corrTexConstantSumsTC(float *dCorr, int CorrPitch,int *dSumI,int *dSumISq,int *dSumIT,int wTile,int wTemplate,int hTemplate,float cPixels,float& fDenomExp,int sharedPitch,int xTemplate,int yTemplate,int w,int h,dim3 threads,dim3 blocks,int sharedMem)
{
    corrTexConstantSums_kernelTC<<<blocks, threads>>>(dCorr,CorrPitch,dSumI,dSumISq,dSumIT,cPixels,fDenomExp,w,h,wTemplate,hTemplate );
}


__global__ void corrTexConstant_kernelTC(float *pCorr,size_t CorrPitch,float cPixels,float fDenomExp,int w,int h,int wTemplate,int hTemplate)
{
    size_t row = blockIdx.y*blockDim.y + threadIdx.y;
    size_t col = blockIdx.x*blockDim.x + threadIdx.x;

    // adjust pointers to row
    pCorr = (float *) ((char *) pCorr+row*CorrPitch);

    // No __syncthreads in this kernel, so we can early-out
    // without worrying about the effects of divergence.
    if ( col >= w || row >= h )
        return;

    int SumI = 0;
    int SumISq = 0;
    int SumIT = 0;
    int inx = 0;

    for ( int j = 0; j < hTemplate; j++ ) 
	{
        for ( int i = 0; i < wTemplate; i++ )
		{
            unsigned char I = tex2D(texImage,(float)col+i,(float)row+j);
            unsigned char T = g_Tpix[inx++];
            SumI += I;
            SumISq += I*I;
            SumIT += I*T;
        }
    }
    pCorr[col] = CorrelationValue(SumI,SumISq,SumIT,g_SumT,cPixels,fDenomExp);
}


void corrTexConstantTC(float *dCorr,int CorrPitch,int wTile,int wTemplate,int hTemplate,float cPixels,float fDenomExp,int sharedPitch,int w,int h,dim3 threads,dim3 blocks,int sharedMem)
{
    corrTexConstant_kernelTC<<<blocks,threads>>>(dCorr,CorrPitch,cPixels,fDenomExp,w,h,wTemplate,hTemplate);
}


#endif

bool DoCorrelationTC(float *hCorr, int w,int h,// width and height of output
    int wTemplate, int hTemplate,int wTile,// width of image tile
    int sharedPitch, int sharedMem,dim3 threads,dim3 blocks,int cIterations,float& gputime)
{
    cudaError_t status;
    bool ret = false;
    size_t CorrPitch;

    float cPixels = (float) wTemplate*hTemplate;
    float fDenomExp;// = float((double) cPixels*g_cpuSumTSq - (double) g_cpuSumT*g_cpuSumT);

    ///hCorr = NULL;
	float *dCorr = NULL;
    int *hSumI = NULL, *dSumI = NULL;
    int *hSumISq = NULL, *dSumISq = NULL;
    int *hSumIT = NULL, *dSumIT = NULL;

    cudaEvent_t start = 0, stop = 0;

    ///hCorr = (float *) malloc( w*sizeof(float)*h );
    hSumI = (int *) malloc( w*sizeof(int)*h );
    hSumISq = (int *) malloc( w*sizeof(int)*h );
    hSumIT = (int *) malloc( w*sizeof(int)*h );
    if ( NULL == hCorr || NULL == hSumI || NULL == hSumISq || NULL == hSumIT )
        goto Error;

    cuda(MallocPitch( (void **) &dCorr, &CorrPitch, w*sizeof(float), h ) );
    cuda(MallocPitch( (void **) &dSumI, &CorrPitch, w*sizeof(int), h ) );
    cuda(MallocPitch( (void **) &dSumISq, &CorrPitch, w*sizeof(int), h ) );
    cuda(MallocPitch( (void **) &dSumIT, &CorrPitch, w*sizeof(int), h ) );

    cuda(Memset( dCorr, 0, CorrPitch*h ) );
    cuda(Memset( dSumI, 0, CorrPitch*h ) );
    cuda(Memset( dSumISq, 0, CorrPitch*h ) );
    cuda(Memset( dSumIT, 0, CorrPitch*h ) );

	corrTexConstantSumsTC(dCorr,CorrPitch,dSumI,dSumISq,dSumIT,wTile,wTemplate,hTemplate,cPixels,fDenomExp,sharedPitch,w,h,threads,blocks,sharedMem);

    cuda(Memcpy2D( hSumI, w*sizeof(int), dSumI, CorrPitch, w*sizeof(int), h, cudaMemcpyDeviceToHost ) );
    cuda(Memcpy2D( hSumISq, w*sizeof(int), dSumISq, CorrPitch, w*sizeof(int), h, cudaMemcpyDeviceToHost ) );
    cuda(Memcpy2D( hSumIT, w*sizeof(int), dSumIT, CorrPitch, w*sizeof(int), h, cudaMemcpyDeviceToHost ) );
    cuda(Memcpy2D( hCorr, w*sizeof(float), dCorr, CorrPitch, w*sizeof(float), h, cudaMemcpyDeviceToHost ) );

    cuda(Memset2D( dCorr, CorrPitch, 0, w*sizeof(float), h ) );
    cuda(DeviceSynchronize());
    
	
	cuda(EventCreate( &start, 0 ) );
    cuda(EventCreate( &stop, 0 ) );
	cuda(EventRecord(start,0));

    for ( int i = 0; i < cIterations; i++ )
	{
        corrTexConstantTC(dCorr,CorrPitch,wTile,wTemplate,hTemplate,cPixels,fDenomExp,sharedPitch,w,h,threads,blocks,sharedMem);
    }

    cuda(EventRecord(stop,0));

    cuda(Memcpy2D( hCorr, w*sizeof(float), dCorr, CorrPitch, w*sizeof(float), h, cudaMemcpyDeviceToHost ) );
    
    cuda(EventElapsedTime( &gputime, start, stop ) );

    ret = true;
Error:
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    //free( hCorr );
    free( hSumI );
    free( hSumISq );
    free( hSumIT );
    if ( dCorr ) cudaFree( dCorr );
    if ( dSumI ) cudaFree( dSumI );
    if ( dSumI ) cudaFree( dSumISq );
    if ( dSumI ) cudaFree( dSumIT );
    return ret;
}

cudaError_t CopyToTemplate(unsigned char *img,size_t imgPitch,int xTemplate,int yTemplate,int wTemplate,int hTemplate,int OffsetX,int OffsetY)
{
    cudaError_t status;
    unsigned char pixels[maxTemplatePixels];
    int inx = 0;
    int SumT = 0;
    int SumTSq = 0;
    int cPixels = wTemplate*hTemplate;
    size_t sizeOffsets = cPixels*sizeof(int);
    float fSumT, fDenomExp, fcPixels;
    cuda(Memcpy2D(pixels,wTemplate,img+yTemplate*imgPitch+xTemplate,imgPitch,wTemplate,hTemplate,cudaMemcpyDeviceToHost));
    cuda(MemcpyToSymbol(g_Tpix,pixels,cPixels));
    for ( int i = OffsetY; i < OffsetY+hTemplate; i++ )
	{
        for ( int j = OffsetX; j < OffsetX+wTemplate; j++)
		{
            SumT += pixels[inx];
            SumTSq += pixels[inx]*pixels[inx];
            poffsetx[inx] = j;
            poffsety[inx] = i;
            inx += 1;
        }
    }
    g_cpuSumT = SumT;
    g_cpuSumTSq = SumTSq;

    cuda(MemcpyToSymbol(g_xOffset, poffsetx, sizeOffsets) );
    cuda(MemcpyToSymbol(g_yOffset, poffsety, sizeOffsets) );

    fSumT = (float) SumT;
    cuda(MemcpyToSymbol(g_SumT, &fSumT, sizeof(float)) );

    fDenomExp = float( (double)cPixels*SumTSq - (double) SumT*SumT);
    cuda(MemcpyToSymbol(g_fDenomExp, &fDenomExp, sizeof(float)) );

    fcPixels = (float) cPixels;
    cuda(MemcpyToSymbol(g_cPixels, &fcPixels, sizeof(float)) );
Error:
    return status;
}


CImg<float> mainncTC(CImg<unsigned  char> *idata,CImg<unsigned  char> *tdata,float& gpuTime,float& datatransfertime)
{
    cudaError_t status;

	const unsigned char *hidata;
    static unsigned char *didata;
    static unsigned char *htdata;
    static unsigned char *dtdata;
    static unsigned int HostPitch;
	static size_t DevicePitch;
	static unsigned int HostPitcht;
	static size_t DevicePitcht;
    static int w;
	static int h;
    static int wTemplate;
    static int hTemplate;
	static int wc;
	static int hc;
	static CImg<float> cdata;
    static int wTile;
    static dim3 threads;
    static dim3 blocks;
    static int sharedPitch;
    static int sharedMem;
    static cudaArray *pArrayImage;
    static cudaArray *pArrayTemplate;
    static cudaChannelFormatDesc desc;
	static cudaChannelFormatDesc desct;
	static int cIterations;
	

	if(idata && !tdata)
	{
		hidata = &(idata->at(0));

		cudaEvent_t start,stop;cudaEventCreate(&start);cudaEventCreate(&stop);cudaEventRecord(start,0);

		cuda(MemcpyToArray(pArrayImage,0,0,hidata,w*h,cudaMemcpyHostToDevice));
		cuda(BindTextureToArray(texImage,pArrayImage));

		cudaEventRecord(stop,0);cudaEventSynchronize(stop);cudaEventElapsedTime(&datatransfertime,start,stop);cudaEventDestroy(start);cudaEventDestroy(stop);
	}

	if(idata && tdata)
	{
		hidata = &(idata->at(0));
		didata = NULL;
		htdata = &(tdata->at(0));
		dtdata = NULL;
		HostPitch = idata->width()*sizeof(unsigned char);
		HostPitcht = tdata->width()*sizeof(unsigned char);
		w = idata->width();
		h = idata->height();
		wTemplate = tdata->width();
		hTemplate = tdata->height();
		wc = w-wTemplate;
		hc = h-hTemplate;
		cdata.assign(hc,wc);
		pArrayImage = NULL;
		pArrayTemplate = NULL;
		desc = cudaCreateChannelDesc<unsigned char>();
		desct = cudaCreateChannelDesc<unsigned char>();
		cuda(SetDeviceFlags( cudaDeviceMapHost ) );
		cuda(DeviceSetCacheConfig( cudaFuncCachePreferShared ) );
		if (cudaSuccess != cudaMallocPitch((void**)&didata,&DevicePitch,w,h))
		    goto Error;
	    cudaMemcpy2D( didata, DevicePitch, hidata, w, w, h, cudaMemcpyHostToDevice );
		if (cudaSuccess != cudaMallocPitch((void**)&dtdata,&DevicePitcht,wTemplate,hTemplate))
			goto Error;
		cudaMemcpy2D( dtdata, DevicePitcht, htdata, wTemplate, wTemplate, hTemplate, cudaMemcpyHostToDevice );
		cuda(MallocArray( &pArrayImage, &desc, w, h ) );
		cuda(MallocArray( &pArrayTemplate, &desct, wTemplate, hTemplate ) );
		cuda(MemcpyToArray( pArrayImage, 0, 0, hidata, w*h, cudaMemcpyHostToDevice ) );
		cuda(MemcpyToArray( pArrayTemplate, 0, 0, htdata, wTemplate*hTemplate, cudaMemcpyHostToDevice ) );
	    cuda(BindTextureToArray( texImage, pArrayImage ) );
	    cuda(BindTextureToArray( texTemplate, pArrayTemplate ) );

		//threads.x = 16; threads.y = 8; threads.z = 1;
		//blocks.x = INTCEIL(w,threads.x); blocks.y = INTCEIL(h,threads.y); blocks.z = 1;
		cIterations = 1;

		CopyToTemplate(didata,DevicePitch,0,0,wTemplate,hTemplate,0,0);
	    // height of thread block must be >= hTemplate
		wTile = 32;
		threads = dim3(32,8);
		blocks = dim3(w/wTile+(0!=w%wTile),h/threads.y+(0!=h%threads.y));
		sharedPitch = ~63&(wTile+wTemplate+63);
		sharedMem = sharedPitch*(threads.y+hTemplate);
	}


#if 0    
	// height of thread block must be >= hTemplate
    wTile = 32;
    threads = dim3(32,8);
    blocks = dim3(w/wTile+(0!=w%wTile),h/threads.y+(0!=h%threads.y));

    sharedPitch = ~63&(wTile+wTemplate+63);
    sharedMem = sharedPitch*(threads.y+hTemplate);

    TEST_VECTOR( corrShared, false, 100, NULL );
#endif

#if 0	
	// height of thread block must be >= hTemplate
    wTile = 32;
    threads = dim3(32,8);
    blocks = dim3(w/wTile+(0!=w%wTile),h/threads.y+(0!=h%threads.y));

    sharedPitch = ~63&(((wTile+wTemplate)+63));
    sharedMem = sharedPitch*(threads.y+hTemplate);

    TEST_VECTOR( corrSharedSM, false, 100, NULL );
#endif

#if 0
    TEST_VECTOR( corrShared4, false, 100, NULL );

    // set up blocking parameters for 2D tex-constant formulation
    threads.x = 32; threads.y = 16; threads.z = 1;
    blocks.x = INTCEIL(w,threads.x); blocks.y = INTCEIL(h,threads.y); blocks.z = 1;
    TEST_VECTOR( corrTexConstant, false, 100, NULL );
#endif


	// set up blocking parameters for 2D tex-tex formulation
    //DoCorrelation(xOffset,yOffset,w,h,xTemplate-xOffset,yTemplate-yOffset,wTemplate,hTemplate,wTile,sharedPitch,sharedMem,threads,blocks,cIterations,outputFilename);
	if(idata)
	{
		///cudaEvent_t start,stop;cudaEventCreate(&start);cudaEventCreate(&stop);cudaEventRecord(start,0);
		
		DoCorrelationTC(&(cdata.at(0)),wc,hc,wTemplate,hTemplate,wTile,sharedPitch,sharedMem,threads,blocks,cIterations,gpuTime);

		///cudaEventRecord(stop,0);cudaEventSynchronize(stop);cudaEventElapsedTime(&gpuTime,start,stop);cudaEventDestroy(start);cudaEventDestroy(stop);
	}
	if(idata)
		return cdata;
Error:
	//free( hidata );
    cudaFree(didata); 
    //free( htdata );
    cudaFree(dtdata); 
    cudaFreeArray(pArrayImage);
    cudaFreeArray(pArrayTemplate);
    return cdata;
}
