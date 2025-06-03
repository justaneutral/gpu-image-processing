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

//const int maxTemplatePixels = 3072;
//__constant__ int g_xOffset[maxTemplatePixels];
//__constant__ int g_yOffset[maxTemplatePixels];
//__constant__ unsigned char g_Tpix[maxTemplatePixels];
//__constant__ float g_cPixels, g_SumT, g_fDenomExp;
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


__global__ void corrTexTexSums_kernelTR(float *pCorr,size_t CorrPitch,int *pI,int *pISq,int *pIT,float cPixels,int wTemplate,int hTemplate,int w,int h)
{
	int step = 16;

    size_t row = blockIdx.y*blockDim.y + threadIdx.y;
    size_t col = blockIdx.x*blockDim.x + threadIdx.x;
	size_t offs = blockIdx.z*blockDim.z + threadIdx.z;

    // adjust pCorr to point to row
    pCorr = (float *)((char *)pCorr+row*CorrPitch);
    pI    = (int *) ((char *) pI  +row*CorrPitch);
    pISq  = (int *) ((char *) pISq+row*CorrPitch);
    pIT   = (int *) ((char *) pIT +row*CorrPitch);

    // No __syncthreads in this kernel, so we can early-out
    // without worrying about the effects of divergence.
    if(col>=w || row>=h)
        return;
    int SumI = 0;
    int SumT = 0;
    int SumISq = 0;
    int SumTSq = 0;
    int SumIT = 0;
    for(int y=offs;y<hTemplate;y+=step)
	{
        for (int x=0;x<wTemplate;x++)
		{
            unsigned char I = tex2D(texImageTT,(float)col+x+offs,(float)row+y);
            unsigned char T = tex2D(texTemplateTT,(float)x+offs,(float)y);
            SumI += I;
            SumT += T;
            SumISq += I*I;
            SumTSq += T*T;
            SumIT += I*T;
        }
        float fDenomExp = float( (double) cPixels*SumTSq - (double) SumT*SumT);
		if(offs==0)
		{
			pI[col] = SumI;
			pISq[col] = SumISq;
			pIT[col] = SumIT;
			pCorr[col] = CorrelationValue( SumI, SumISq, SumIT, SumT, cPixels, fDenomExp );
		}
		else
		{
			atomicAdd(&(pI[col]),SumI);
			atomicAdd(&(pISq[col]),SumISq);
			atomicAdd(&(pIT[col]),SumIT);
			atomicAdd(&(pCorr[col]),CorrelationValue( SumI, SumISq, SumIT, SumT, cPixels, fDenomExp ));
		}
    }
}

void corrTexTexSumsTR(float *dCorr,int CorrPitch,int *pI,int *pISq,int *pIT,int wTile,int wTemplate,int hTemplate,float cPixels,int sharedPitch,int w,int h,dim3 threads,dim3 blocks,int sharedMem)
{
    corrTexTexSums_kernelTR<<<blocks,threads>>>(dCorr,CorrPitch,pI,pISq,pIT,cPixels,wTemplate,hTemplate,w,h);
}


/////////////////do GPU proessing here ///////////////////
__global__ void corrTexTex_kernelTR(float *pCorr,size_t CorrPitch,int *pI,int SumIPitch, int *pISq,int SumISqPitch,int *pIT,int SumITPitch, int *pT,int SumTPitch,int *pTSq,int SumTSqPitch,float cPixels,int wTemplate,int hTemplate,int w,int h,int wi,int hi)
{
	size_t row = hi; //correlation function vertical offset
    size_t col = wi; //correlation function horizontal offset
	
	size_t yt = blockIdx.y*blockDim.y + threadIdx.y; //template image pixel vertical position
	size_t xt = blockIdx.x*blockDim.x + threadIdx.x;  //template image pixel horizontal position

	if(col>=w || row>=h || yt>=hTemplate || xt>=wTemplate) return;
	
	size_t yi = yt + row; //source image pixel vertical position
	size_t xi = xt + col; //source image pixel horizontal position

    // adjust global pointers to point to the current correlation function row and column
    pCorr = (float*)((float *)((char *)pCorr+row*CorrPitch)+col);
    pI    = (int*)((int*)((char*)pI+row*SumIPitch)+col); 
    pISq  = (int*)((int*)((char*)pISq+row*SumISqPitch)+col); 
    pIT   = (int*)((int*)((char*)pIT+row*SumITPitch)+col); 
    pT    = (int*)((int*)((char*)pT+row*SumTPitch)+col); 
    pTSq  = (int*)((int*)((char*)pTSq+row*SumTSqPitch)+col);

	//get template and source pixels
	unsigned char I = tex2D(texImageTT,(float)xi,(float)yi);
    unsigned char T = tex2D(texTemplateTT,(float)xt,(float)yt);

	//calculate pixel values
	int mI = I;
    int mT = T;
    int mISq = I*I;
    int mTSq = T*T;
    int mIT = I*T;

	//create accumulators for all threads in each block
	__shared__ int sI;
    __shared__ int sT;
    __shared__ int sISq;
    __shared__ int sTSq;
    __shared__ int sIT;

	__shared__ bool done;

	if(threadIdx.y==0 && threadIdx.x==0)
	{
		sI = 1;
		sT = 1;
		sISq = 1;
		sTSq = 1;
		sIT = 1;
	}
	//__syncthreads();

	if(threadIdx.y!=0 || threadIdx.x!=0)
	{
		atomicAdd(&sI,mI);
		atomicAdd(&sT,mT);
		atomicAdd(&sISq,mISq);
		atomicAdd(&sTSq,mTSq);
		atomicAdd(&sIT,mIT);
	}
	
	__syncthreads();

	if(blockIdx.y==0 && blockIdx.x==0 && threadIdx.y==0 && threadIdx.x==0)
	{
		*pI=sI;
		*pISq=sISq;
		*pT=sT;
		*pTSq=sTSq;
		*pIT=sIT;
		*pCorr=1234567.0;
	}

	if((blockIdx.y!=0 || blockIdx.x!=0) && threadIdx.y==0 && threadIdx.x==0)
	{
		//while(*pCorr<1234566);
		atomicAdd(pI,sI);
		atomicAdd(pISq,sISq);
		atomicAdd(pT,sT);
		atomicAdd(pTSq,sTSq);
		atomicAdd(pIT,sIT);
		atomicAdd(pCorr,1.0);
	}
}


void corrTexTexTR(float *dCorr,int CorrPitch,int *dSumI,int SumIPitch,int *dSumISq,int SumISqPitch,int *dSumIT,int SumITPitch,int *dSumT,int SumTPitch,int *dSumTSq,int SumTSqPitch,int wTile,int wTemplate,int hTemplate,float cPixels,int sharedPitch,int w,int h,dim3 threads,dim3 blocks,int sharedMem,int wi,int hi)
{
    corrTexTex_kernelTR<<<blocks,threads>>>(dCorr,CorrPitch,dSumI,SumIPitch,dSumISq,SumISqPitch,dSumIT,SumITPitch,dSumT,SumTPitch,dSumTSq,SumTSqPitch,cPixels,wTemplate,hTemplate,w,h,wi,hi);
}


bool DoCorrelationTR(float *hCorr, int w,int h,// width and height of output
    int wTemplate, int hTemplate,int wTile,// width of image tile
    int sharedPitch, int sharedMem,dim3 threads,dim3 blocks,int cIterations,float& gputime)
{
    cudaError_t status;
    bool ret = false;
    size_t CorrPitch,SumIPitch,SumISqPitch,SumITPitch,SumTPitch,SumTSqPitch;

    float cPixels = (float) wTemplate*hTemplate;
    
    ///hCorr = NULL;
	float *dCorr = NULL;
    int *hSumI = NULL, *dSumI = NULL;
    int *hSumISq = NULL, *dSumISq = NULL;
    int *hSumIT = NULL, *dSumIT = NULL;
	int *hSumT = NULL, *dSumT = NULL;
	int *hSumTSq = NULL, *dSumTSq = NULL;

    cudaEvent_t start = 0, stop = 0;

    ///hCorr = (float *) malloc( w*sizeof(float)*h );
    hSumI = (int *) malloc( w*sizeof(int)*h );
	hSumT = (int *) malloc( w*sizeof(int)*h );
    hSumISq = (int *) malloc( w*sizeof(int)*h );
	hSumTSq = (int *) malloc( w*sizeof(int)*h );
    hSumIT = (int *) malloc( w*sizeof(int)*h );

    if ( NULL == hCorr || NULL == hSumI || NULL == hSumISq || NULL == hSumIT || NULL == hSumT || NULL == hSumTSq)
        goto Error;

    cuda(MallocPitch( (void **) &dCorr, &CorrPitch, w*sizeof(float), h ) );
    cuda(MallocPitch( (void **) &dSumI, &SumIPitch, w*sizeof(int), h ) );
    cuda(MallocPitch( (void **) &dSumISq, &SumISqPitch, w*sizeof(int), h ) );
    cuda(MallocPitch( (void **) &dSumIT, &SumITPitch, w*sizeof(int), h ) );
	cuda(MallocPitch( (void **) &dSumT, &SumTPitch, w*sizeof(int), h ) );
    cuda(MallocPitch( (void **) &dSumTSq, &SumTSqPitch, w*sizeof(int), h ) );

    cuda(Memset( dCorr, 0, CorrPitch*h ) );
    cuda(Memset( dSumI, 0, SumIPitch*h ) );
    cuda(Memset( dSumISq, 0, SumISqPitch*h ) );
    cuda(Memset( dSumIT, 0, SumITPitch*h ) );
	cuda(Memset( dSumT, 0, SumTPitch*h ) );
    cuda(Memset( dSumTSq, 0, SumTSqPitch*h ) );


	//corrTexTexSumsTR(dCorr,CorrPitch,dSumI,dSumISq,dSumIT,wTile,wTemplate,hTemplate,cPixels,sharedPitch,w,h,threads,blocks,sharedMem);

    	
	cuda(EventCreate( &start, 0 ) );
    cuda(EventCreate( &stop, 0 ) );
	cuda(EventRecord(start,0));
    
	cuda(Memset2D( dCorr, CorrPitch, 0, w*sizeof(float), h ) );
    cuda(DeviceSynchronize());

    for ( int i = 0; i < cIterations; i++ )
	{
		//now wi and hi are indices of a single corelation coefficient
		for(int hi=0;hi<h;hi++)
		{
			for(int wi=0;wi<w;wi++)
			{
				corrTexTexTR(dCorr,CorrPitch,dSumI,SumIPitch,dSumISq,SumISqPitch,dSumIT,SumITPitch,dSumT,SumTPitch,dSumTSq,SumTSqPitch,wTile,wTemplate,hTemplate,cPixels,sharedPitch,w,h,threads,blocks,sharedMem,wi,hi);
				//cuda(DeviceSynchronize());
			}
		}
    }

    cuda(EventRecord(stop,0));

    cuda(Memcpy2D( hSumT, w*sizeof(int), dSumT, SumTPitch, w*sizeof(int), h, cudaMemcpyDeviceToHost ) );
    cuda(Memcpy2D( hSumTSq, w*sizeof(int), dSumTSq, SumTSqPitch, w*sizeof(int), h, cudaMemcpyDeviceToHost ) );
    cuda(Memcpy2D( hSumI, w*sizeof(int), dSumI, SumIPitch, w*sizeof(int), h, cudaMemcpyDeviceToHost ) );
    cuda(Memcpy2D( hSumISq, w*sizeof(int), dSumISq, SumISqPitch, w*sizeof(int), h, cudaMemcpyDeviceToHost ) );
    cuda(Memcpy2D( hSumIT, w*sizeof(int), dSumIT, SumITPitch, w*sizeof(int), h, cudaMemcpyDeviceToHost ) );
    cuda(Memcpy2D( hCorr, w*sizeof(float), dCorr, CorrPitch, w*sizeof(float), h, cudaMemcpyDeviceToHost ) );

	
	/////////////combine after GPU processing ///////////////
	for(int hi=0;hi<h;hi++)
	{
		for(int wi=0;wi<w;wi++)
		{
			float fDenomExp = float((double)cPixels*hSumTSq[hi*w+wi]-(double)hSumT[hi*w+wi]*hSumT[hi*w+wi]);
			hCorr[hi*w+wi] = CorrelationValue((float)hSumI[hi*w+wi],(float)hSumISq[hi*w+wi],(float)hSumIT[hi*w+wi],(float)hSumTSq[hi*w+wi],cPixels,fDenomExp);
		}
	}
	
    cuda(EventElapsedTime( &gputime, start, stop ) );

    ret = true;
Error:
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    //free( hCorr );
    free( hSumI );
    free( hSumISq );
	free( hSumT );
    free( hSumTSq );
    free( hSumIT );
    if ( dCorr ) cudaFree( dCorr );
    if ( dSumI ) cudaFree( dSumI );
    if ( dSumI ) cudaFree( dSumISq );
    if ( dSumI ) cudaFree( dSumIT );
	if ( dSumT ) cudaFree( dSumT );
    if ( dSumT ) cudaFree( dSumTSq );
    return ret;
}


CImg<float> mainncTR(CImg<unsigned  char> *idata,CImg<unsigned  char> *tdata,float& gpuTime,float& datatransfertime)
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
    static cudaArray *pArrayImage=NULL;
    static cudaArray *pArrayTemplate;
    static cudaChannelFormatDesc desc;
	static cudaChannelFormatDesc desct;
	static int cIterations;
	

	if(idata && !tdata)
	{
		hidata = &(idata->at(0));

		//cudaEvent_t start,stop;cudaEventCreate(&start);cudaEventCreate(&stop);cudaEventRecord(start,0);
		if(pArrayImage)
		{
			cuda(MemcpyToArray(pArrayImage,0,0,hidata,w*h,cudaMemcpyHostToDevice));
			cuda(BindTextureToArray(texImageTT,pArrayImage));
		}
		//cudaEventRecord(stop,0);cudaEventSynchronize(stop);cudaEventElapsedTime(&datatransfertime,start,stop);cudaEventDestroy(start);cudaEventDestroy(stop);
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
		wTile = 32;
	    //threads = dim3(32,8);
		blocks = dim3(w/wTile+(0!=w%wTile),h/threads.y+(0!=h%threads.y));
		sharedPitch = ~63&(wTile+wTemplate+63);
		sharedMem = sharedPitch*(threads.y+hTemplate);

		threads.x = 16; threads.y = 16; threads.z = 1;
		blocks.x = INTCEIL(wTemplate,threads.x); blocks.y = INTCEIL(hTemplate,threads.y); blocks.z = 1;
		cIterations = 1;
	}


	if(idata)
	{
		cudaEvent_t start,stop;cudaEventCreate(&start);cudaEventCreate(&stop);cudaEventRecord(start,0);
		
		DoCorrelationTR(&(cdata.at(0)),wc,hc,wTemplate,hTemplate,wTile,sharedPitch,sharedMem,threads,blocks,cIterations,gpuTime);

		cudaEventRecord(stop,0);cudaEventSynchronize(stop);cudaEventElapsedTime(&gpuTime,start,stop);cudaEventDestroy(start);cudaEventDestroy(stop);
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
