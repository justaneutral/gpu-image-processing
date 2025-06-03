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

texture<unsigned char, 2> texImageTT;
texture<unsigned char, 2> texTemplateTT;


__device__ __host__ inline float CorrelationValue( float SumI, float SumISq, float SumIT, float SumT, float cPixels, float fDenomExp )
{
    float Numerator = cPixels*SumIT - SumI*SumT;
    float Denominator = rsqrtf( (cPixels*SumISq - SumI*SumI)*fDenomExp );
    return Numerator * Denominator;
}


__global__ void corrTexTex_kernelTTR(float *pCorr,int *gp, size_t CorrPitch,float cPixels,size_t wTemplate,size_t hTemplate,int w,int h,int step)
{

	size_t row = blockIdx.y*blockDim.y + threadIdx.y;
    size_t col = blockIdx.x*blockDim.x + threadIdx.x;
    
	if(row>=hTemplate || col>=wTemplate)
        return;

	int *browsbcolsdone=gp;
	int *gI = &gp[1];
    int *gT = &gp[2];
    int *gISq = &gp[3];
    int *gTSq = &gp[4];
    int *gIT = &gp[5];

    __shared__ int sI;
    __shared__ int sT;
    __shared__ int sISq;
    __shared__ int sTSq;
    __shared__ int sIT;

	size_t brow = blockIdx.y;
    size_t bcol = blockIdx.x;
	size_t bRC = brow*bcol;

	size_t trow = threadIdx.y;
    size_t tcol = threadIdx.x;

	int offsety = w;
	int offsetx = h;

	unsigned char I = tex2D(texImageTT,(float)col+offsetx,(float)row+offsety);
    unsigned char T = tex2D(texTemplateTT,(float)col,(float)row);

	int mT = (int)T;
	int mI = (int)I;
	int mTSq = mT*mT;
	int mISq = mI*mI;
	int mIT = mI*mT;

	//float *pc = (float *)((char *)pCorr+offsetx*CorrPitch);
	*pCorr = 100.0;
	//pc[offsety] = 100;//offsetx+offsety; //cvl;

#if 0
	if(trow==0 && tcol==0)
	{
		sT = mT;
		sI = mI;
		sISq = mISq;
		sTSq = mTSq;
		sIT = mIT;
		//__syncthreads();
		if(brow==0 && bcol==0)
		{
			*gT = sT;
			*gI = sI;
			*gISq = sISq;
			*gTSq = sTSq;
			*gIT = sIT;

			//__syncthreads();
			//atomicAdd(browsbcolsdone,1);
			//while(*browsbcolsdone<bRC);
			mT = *gT;
			mI = *gI;
			mISq = *gISq;
			mTSq = *gTSq;
			mIT = *gIT;

			float fDenom = float((double)cPixels*mTSq-(double)mT*mT);
			float cvl = CorrelationValue(mI,mISq,mIT,mT,cPixels,fDenom );
			*pc = offsetx+offsety; //cvl;

		}
		else
		{
			//__syncthreads();
			atomicAdd(gT,sT);
			atomicAdd(gI,sI);
			atomicAdd(gISq,sISq);
			atomicAdd(gTSq,sTSq);
			atomicAdd(gIT,sIT);
			//__syncthreads();
			atomicAdd(browsbcolsdone,1);
		}

	}
	else
	{
		//__syncthreads();
		atomicAdd(&sT,mT);
		atomicAdd(&sI,mI);
		atomicAdd(&sISq,mISq);
		atomicAdd(&sTSq,mTSq);
		atomicAdd(&sIT,mIT);
		__syncthreads();
	}
#endif
}

void corrTexTexTTR(float *dCorr,int *gp,int CorrPitch,int wTile,int wTemplate,int hTemplate,float cPixels,int sharedPitch,int w,int h,dim3 threads,dim3 blocks,int step,int sharedMem)
{
	corrTexTex_kernelTTR<<<blocks,threads>>>(dCorr,gp,CorrPitch,cPixels,wTemplate,hTemplate,w,h,step);
}


bool DoCorrelationTTR(float *hCorr, int w,int h,// width and height of output
    int wTemplate, int hTemplate,int wTile,// width of image tile
    int sharedPitch, int sharedMem,dim3 threads,dim3 blocks,int step,float& gputime)
{
    cudaError_t status;
    bool ret = false;
    size_t CorrPitch;

    float cPixels = (float) wTemplate*hTemplate;
    
    ///hCorr = NULL;
	float *dCorr = NULL;
	int *gp = NULL;
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

    cuda(MallocPitch((void**)&dCorr,&CorrPitch,w*sizeof(float),h));
	cuda(Malloc((void**)&gp,6*sizeof(int)));

#ifdef __sumsTT__
    cuda(MallocPitch( (void **) &dSumI, &CorrPitch, w*sizeof(int), h ) );
    cuda(MallocPitch( (void **) &dSumISq, &CorrPitch, w*sizeof(int), h ) );
    cuda(MallocPitch( (void **) &dSumIT, &CorrPitch, w*sizeof(int), h ) );
    cuda(Memset( dSumI, 0, CorrPitch*h ) );
    cuda(Memset( dSumISq, 0, CorrPitch*h ) );
    cuda(Memset( dSumIT, 0, CorrPitch*h ) );
#endif
    cuda(Memset(dCorr,0,CorrPitch*h));
	cuda(Memset(gp,0,6*sizeof(int)));

#ifdef __sumsTT__
    corrTexTexSumsTT(dCorr,CorrPitch,dSumI,dSumISq,dSumIT,wTile,wTemplate,hTemplate,cPixels,sharedPitch,w,h,threads,blocks,sharedMem);
    cuda(Memcpy2D( hSumI, w*sizeof(int), dSumI, CorrPitch, w*sizeof(int), h, cudaMemcpyDeviceToHost ) );
    cuda(Memcpy2D( hSumISq, w*sizeof(int), dSumISq, CorrPitch, w*sizeof(int), h, cudaMemcpyDeviceToHost ) );
    cuda(Memcpy2D( hSumIT, w*sizeof(int), dSumIT, CorrPitch, w*sizeof(int), h, cudaMemcpyDeviceToHost ) );
    cuda(Memcpy2D( hCorr, w*sizeof(float), dCorr, CorrPitch, w*sizeof(float), h, cudaMemcpyDeviceToHost ) );
#endif


    cuda(Memset2D( dCorr, CorrPitch, 0, w*sizeof(float), h ) );
    cuda(DeviceSynchronize());
  
	
	cuda(EventCreate( &start, 0 ) );
    cuda(EventCreate( &stop, 0 ) );
	cuda(EventRecord(start,0));

    for(int wp=0;wp<w;wp++)
	{
		for(int hp=0;hp<h;hp++)
		{
			//cuda(Memset(gp,0,6*sizeof(int)));
			corrTexTexTTR(dCorr,gp,CorrPitch,wTile,wTemplate,hTemplate,cPixels,sharedPitch,wp,hp,threads,blocks,step,sharedMem);
			cudaDeviceSynchronize();
		}
	}

    cuda(EventRecord(stop,0));
	int tstgp[6];
	cudaMemcpy(tstgp,gp,6*sizeof(int),cudaMemcpyDeviceToHost);

    cuda(Memcpy2D( hCorr, w*sizeof(float), dCorr, CorrPitch, w*sizeof(float), h, cudaMemcpyDeviceToHost ) );
    
    cuda(EventElapsedTime( &gputime, start, stop ) );

    ret = true;
Error:
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    //free( hCorr );
	if(gp) cudaFree(gp);
	if (dCorr) cudaFree(dCorr);
#ifdef __sumsTT__
    free( hSumI );
    free( hSumISq );
    free( hSumIT );

    if ( dSumI ) cudaFree( dSumI );
    if ( dSumI ) cudaFree( dSumISq );
    if ( dSumI ) cudaFree( dSumIT );
#endif
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

		//cudaEvent_t start,stop;
		//cudaEventCreate(&start);
		//cudaEventCreate(&stop);
		//cudaEventRecord(start,0);

		cuda(MemcpyToArray(pArrayImage,0,0,hidata,w*h,cudaMemcpyHostToDevice));
		cuda(BindTextureToArray(texImageTT,pArrayImage));

		//cudaEventRecord(stop,0);
		//cudaEventSynchronize(stop);
		//cudaEventElapsedTime(&datatransfertime,start,stop);
		//cudaEventDestroy(start);
		//cudaEventDestroy(stop);
		datatransfertime = 0;
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
		//wTile = 32;
	    //threads = dim3(32,8);
		//blocks = dim3(w/wTile+(0!=w%wTile),h/threads.y+(0!=h%threads.y));
		//sharedPitch = ~63&(wTile+wTemplate+63);
		//sharedMem = sharedPitch*(threads.y+hTemplate);
		
		///threads.x = 16; threads.y = 16; threads.z = 1;
		///blocks.x = INTCEIL(wc,threads.x); blocks.y = INTCEIL(hc,threads.y); blocks.z = 1;

		//step = 4;
		//threads.x = step; threads.y = step; threads.z = 1;
		//blocks.x = INTCEIL(wc,INTCEIL(threads.x,step)); blocks.y = INTCEIL(hc*step/step,INTCEIL(threads.y,step)); blocks.z = 1;

		step = 8; //8 is max
		threads.x = INTCEIL(wTemplate,step); threads.y = INTCEIL(hTemplate,step); threads.z = 1;
		blocks.x = INTCEIL(wTemplate,threads.x); blocks.y = INTCEIL(hTemplate,threads.y); blocks.z = 1;


		cIterations = 1;
	}


	// set up blocking parameters for 2D tex-tex formulation
    //DoCorrelation(xOffset,yOffset,w,h,xTemplate-xOffset,yTemplate-yOffset,wTemplate,hTemplate,wTile,sharedPitch,sharedMem,threads,blocks,cIterations,outputFilename);
	if(idata)
	{
		///cudaEvent_t start,stop;cudaEventCreate(&start);cudaEventCreate(&stop);cudaEventRecord(start,0);
		
		DoCorrelationTTR(&(cdata.at(0)),wc,hc,wTemplate,hTemplate,wTile,sharedPitch,sharedMem,threads,blocks,step,gpuTime);

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
