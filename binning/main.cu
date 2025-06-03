#if 0
//#pragma ones
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

__device__ __host__ inline float CorrelationValue( float SumI, float SumISq, float SumIT, float SumT, float cPixels, float fDenomExp )
{
    float Numerator = cPixels*SumIT - SumI*SumT;
    float Denominator = rsqrtf( (cPixels*SumISq - SumI*SumI)*fDenomExp );
    return Numerator * Denominator;
}

__global__ void corrTexTex_kernelTT(float *pCorr,size_t CorrPitch,float cPixels,size_t wTemplate,size_t hTemplate,int w,int h,int step)
{
    //size_t row = blockIdx.y*blockDim.y + threadIdx.y;
    //size_t col = blockIdx.x*blockDim.x + threadIdx.x;
	
	//size_t offset = blockIdx.z*blockDim.z + threadIdx.z;
	//float fDenom;
	//__shared__ float cv;
	
	size_t row = blockIdx.y;
	size_t col = blockIdx.x;

	size_t offsety = threadIdx.y;
	size_t offsetx = threadIdx.x;

	
    if(col>=w || row>=h)
        return;
    __shared__ int SumI;
    __shared__ int SumT;
    __shared__ int SumISq;
    __shared__ int SumTSq;
    __shared__ int SumIT;

    for(size_t y=offsety;y<hTemplate;y+=step)
	{
        for (size_t x=offsetx;x<wTemplate;x+=step)
		{
            unsigned char I = tex2D(texImageTT,(float)col+x,(float)row+y);
            unsigned char T = tex2D(texTemplateTT,(float)x,(float)y);
			
			if(x==0 && y==0 && offsetx==0 && offsety == 0)
			{
				SumI = 0;
				SumT = 0;
				SumISq = 0;
				SumTSq = 0;
				SumIT = 0;
			}
			else
			{
				atomicAdd(&SumI,I);
				atomicAdd(&SumT,T);
				atomicAdd(&SumISq,I*I);
				atomicAdd(&SumTSq,T*T);
				atomicAdd(&SumIT,I*T);
			}
		}
	}

	if((offsety==0) && (offsetx==0) )
	{
		//cv = cvl;
		// adjust pCorr to point to row
		float *pC = (float *)((char *)pCorr+row*CorrPitch);
		__syncthreads();
		float fDenom = float( (double) cPixels*SumTSq - (double) SumT*SumT);
		float cvl = CorrelationValue( SumI, SumISq, SumIT, SumT, cPixels, fDenom );
		pC[col] = cvl;
	}
}

void corrTexTexTT(float *dCorr, int CorrPitch,int wTile,int wTemplate,int hTemplate,float cPixels,int sharedPitch,int w,int h,dim3 threads,dim3 blocks,int step,int sharedMem)
{
    
	corrTexTex_kernelTT<<<blocks, threads>>>(dCorr,CorrPitch,cPixels,wTemplate,hTemplate,w,h,step);
}


bool DoCorrelationTT(float *hCorr, int w,int h,// width and height of output
    int wTemplate, int hTemplate,int wTile,// width of image tile
    int sharedPitch, int sharedMem,dim3 threads,dim3 blocks,int step,float& gputime)
{
    cudaError_t status;
    bool ret = false;
    size_t CorrPitch;

    float cPixels = (float) wTemplate*hTemplate;
    
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
    cuda(Memset( dCorr, 0, CorrPitch*h ) );
    cuda(Memset2D( dCorr, CorrPitch, 0, w*sizeof(float), h ) );
    cuda(DeviceSynchronize());
	cuda(EventCreate( &start, 0 ) );
    cuda(EventCreate( &stop, 0 ) );
	cuda(EventRecord(start,0));

    corrTexTexTT(dCorr,CorrPitch,wTile,wTemplate,hTemplate,cPixels,sharedPitch,w,h,threads,blocks,step,sharedMem);

    cuda(EventRecord(stop,0));

    cuda(Memcpy2D( hCorr, w*sizeof(float), dCorr, CorrPitch, w*sizeof(float), h, cudaMemcpyDeviceToHost ) );
    
    cuda(EventElapsedTime( &gputime, start, stop ) );

    ret = true;
Error:
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return ret;
}


CImg<float> mainncTT(CImg<unsigned  char> *idata,CImg<unsigned  char> *tdata,float& gpuTime,float& datatransfertime)
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
	static int step;

	if(idata && !tdata)
	{
		hidata = &(idata->at(0));

		cudaEvent_t start,stop;cudaEventCreate(&start);cudaEventCreate(&stop);cudaEventRecord(start,0);

		cuda(MemcpyToArray(pArrayImage,0,0,hidata,w*h,cudaMemcpyHostToDevice));
		cuda(BindTextureToArray(texImageTT,pArrayImage));

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
	    cuda(BindTextureToArray( texImageTT, pArrayImage ) );
	    cuda(BindTextureToArray( texTemplateTT, pArrayTemplate ) );

		step = 16;
		threads.x = step; threads.y = step; threads.z = 1;
		blocks.x = INTCEIL(wc,INTCEIL(threads.x,step)); blocks.y = INTCEIL(hc*step/step,INTCEIL(threads.y,step)); blocks.z = 1;

		cIterations = 1;
	}

	// set up blocking parameters for 2D tex-tex formulation
    //DoCorrelation(xOffset,yOffset,w,h,xTemplate-xOffset,yTemplate-yOffset,wTemplate,hTemplate,wTile,sharedPitch,sharedMem,threads,blocks,cIterations,outputFilename);
	if(idata)
	{
		///cudaEvent_t start,stop;cudaEventCreate(&start);cudaEventCreate(&stop);cudaEventRecord(start,0);
		DoCorrelationTT(&(cdata.at(0)),wc,hc,wTemplate,hTemplate,wTile,sharedPitch,sharedMem,threads,blocks,step,gpuTime);
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


#endif //#if 0





