
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define INTCEIL(a,b) (((a)+(b)-1)/(b))

#include <time.h>

#include <stdio.h>
//#include <Windows.h>

//#define EXPLICIT_PRINT_OUT 1

typedef unsigned char uchar;
typedef unsigned int uint;

texture<uchar,2> texImageTT;
texture<uchar,2> texTemplateTT;


#define gpuerchk(ans) { gpuassert((ans), __FILE__, __LINE__, retcnt); }
inline float gpuassert(cudaError_t gpuerrorcode, const char *file, int line, float& retcnt)
{
   retcnt-=1.0;
   if (gpuerrorcode != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(gpuerrorcode), file, line);
      return retcnt;
   }
}

__device__ __host__ inline float CorrelationValue( float SumI, float SumISq, float SumIT, float SumT, float cPixels, float fDenomExp )
{
    float Numerator = cPixels*SumIT - SumI*SumT;
    float Denominator = rsqrtf( (cPixels*SumISq - SumI*SumI)*fDenomExp );
    return Numerator * Denominator;
}

__global__ void corrTexTex_kernelTT(float *pCorr,size_t CorrPitch,float cPixels,size_t wTemplate,size_t hTemplate,int w,int h,int step)
{
    size_t row = blockIdx.y*blockDim.y + threadIdx.y;
    size_t col = blockIdx.x*blockDim.x + threadIdx.x;
	
	//size_t sz = blockIdx.x*blockDim.x + threadIdx.x;
	//size_t col = sz % w;
	//size_t row = (sz-col)/h;
	

    if(col>=w || row>=h)
        return;
	
    unsigned char Ic;
    unsigned char Tc;
	float I,T;

	float SumI = 0.0;
    float SumT = 0.0;
    float SumISq = 0.0;
    float SumTSq = 0.0;
    float SumIT = 0.0;

    for(size_t y=0;y<hTemplate;y+=step)
	{	

        for (size_t x=0;x<wTemplate;x+=step)
		{
            Ic = tex2D(texImageTT,(float)col+x,(float)row+y);
            Tc = tex2D(texTemplateTT,(float)x,(float)y);
			I = ((float)Ic)*.125;
			T = ((float)Tc)*.125;
			SumI += (float)I;
			SumT += (float)T;
			SumISq += (float)I*I;
			SumTSq += (float)T*T;
			SumIT += (float)I*T;
		}
	}

	float *pC = (float *)((char *)pCorr+row*CorrPitch);
	float fDenom = float( (double) cPixels*SumTSq - (double) SumT*SumT);
	float cvl = CorrelationValue( SumI, SumISq, SumIT, SumT, cPixels, fDenom );
	pC[col] = cvl;

	//if(col==(w-1) && row==(h-1))
	//{
	//	printf("cvl=%f, last Ic,Tc = %u,%u\n",cvl,Ic,Tc);
	//}

}

//int cscntr = 0;
//CRITICAL_SECTION CriticalSection; 
float bfa_kernel_main(uint& x,uint& y,uchar *refimgptr, uint refimgwidth, uint refimgheight, uchar *srcimgptr, uint srcimgwidth, uint srcimgheight,uint corrh, uint corrw,uint step)
{

//	InitializeCriticalSectionAndSpinCount(&CriticalSection,0x00000400);

	static int state = 0;

	float retcnt = 0.0;
	float correlation[400];

	float milliseconds = -1.0;

	//cudaError_t status;

	//define grid parameters
	dim3 threads, blocks; 
	threads.z = 1;
	threads.x = 28;
	threads.y = 2;
	blocks.x = INTCEIL(corrw,threads.x); 
	blocks.y = INTCEIL(corrh,threads.y); 
	blocks.z = 1;

	cudaEvent_t start, stop;
	size_t correlationpitch; 
	float *dcorrelation = NULL;
	cudaArray *pArrayImage;
	cudaArray *pArrayTemplate;


	size_t freemem,totalmem;
	
		cudaDeviceReset();
		cudaDeviceSynchronize();

//		EnterCriticalSection(&CriticalSection);
//		cudaMemGetInfo(&freemem, &totalmem );
//		LeaveCriticalSection(&CriticalSection);

//#ifdef EXPLICIT_PRINT_OUT
//		printf("1)freemem=%u,totalmem=%u\n",freemem,totalmem);
//#endif
		//if(freemem<1032690073)
		//	return -8;

		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		//creare correlation function array in GPU
		cudaMallocPitch((void**)&dcorrelation,&correlationpitch,corrw*sizeof(float),corrh);

		//create template and image in GPU
		cudaChannelFormatDesc desc= cudaCreateChannelDesc<unsigned char>();
		cudaChannelFormatDesc desct = cudaCreateChannelDesc<unsigned char>();
		cudaMallocArray(&pArrayImage,&desc,srcimgwidth,srcimgheight);
		cudaMallocArray(&pArrayTemplate,&desct,refimgwidth,refimgheight);

//	EnterCriticalSection(&CriticalSection);
//	cudaMemGetInfo(&freemem, &totalmem);
//	LeaveCriticalSection(&CriticalSection);

//#ifdef EXPLICIT_PRINT_OUT
//	printf("2)freemem=%u,totalmem=%u\n",freemem,totalmem);
//#endif

		cudaMemcpyToArray(pArrayImage,0,0,srcimgptr,srcimgwidth*srcimgheight,cudaMemcpyHostToDevice);
		cudaMemcpyToArray(pArrayTemplate,0,0,refimgptr,refimgwidth*refimgheight,cudaMemcpyHostToDevice);

		cudaBindTextureToArray(texImageTT,pArrayImage);
		cudaBindTextureToArray(texTemplateTT,pArrayTemplate);
	
		//call correlation using texture memory
		cudaMemset2D(dcorrelation,correlationpitch,0,corrw*sizeof(float),corrh);
		cudaEventRecord(start);
	
		corrTexTex_kernelTT<<<blocks,threads>>>(dcorrelation,correlationpitch,refimgwidth*refimgheight,refimgwidth,refimgheight,corrw,corrh,step);
		cudaMemGetInfo(&freemem, &totalmem );
//#ifdef EXPLICIT_PRINT_OUT
//		printf("3)freemem=%u,totalmem=%u\n",freemem,totalmem);
//#endif
		//gpuerchk(cudaGetLastError());

		cudaEventRecord(stop);
		cudaMemcpy2D(correlation,corrw*sizeof(float),dcorrelation,correlationpitch,corrw*sizeof(float),corrh,cudaMemcpyDeviceToHost);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds,start,stop);

		unsigned int posx=0,posy=0;
		float maxcor=0.0;
		for(int posx = 0; posx<corrw; posx++)
		{
			for(int posy = 0;posy<corrh; posy++)
			{
				float t = correlation[posy*corrw+posx];
				if(maxcor<t)
				{
					maxcor=t;
					x = posx;
					y = posy;
				}
			}
		}



		cudaFree(dcorrelation);
		cudaFreeArray(pArrayImage);
		cudaFreeArray(pArrayTemplate);

//	EnterCriticalSection(&CriticalSection);
//	cudaMemGetInfo(&freemem, &totalmem );
//	LeaveCriticalSection(&CriticalSection);

//#ifdef EXPLICIT_PRINT_OUT
//	printf("4)freemem=%u,totalmem=%u\n",freemem,totalmem);
//#endif

//	if(1032690073>freemem)
//	{
//		printf("returning.\n");
//		return -1.0;
//	}

	return milliseconds;
}


#define P3000COEF (1229000)
#define TitanXpCOEF (1895000)

__global__ void msdelay_kernel(uint delaymilliseconds,clock_t *global_now)
{
    size_t row = blockIdx.y*blockDim.y + threadIdx.y;
    size_t col = blockIdx.x*blockDim.x + threadIdx.x;

	volatile long x = 0;

    if(col>0 || row>0)
        return;

	clock_t start = clock();
	clock_t now;
	
	for (;;) 
	{
	  now = clock();
	  clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
	  if (cycles >= TitanXpCOEF*delaymilliseconds)
	  {
		break;
	  }
	}
	*global_now = now;
	

	//for(long i = 0;i<1000;i++)
	//	for(long j=0;j<1000;j++)
	//		(*global_now)++;

}

uint bfa_kernel_msdelay(uint delaymilliseconds)
{
	uint numiter = delaymilliseconds/1000;
	uint lasdelay = delaymilliseconds%1000;

	clock_t *global_now = NULL;
	clock_t cpu_now;
	float milliseconds;
	dim3 threads, blocks; 
	threads.z = 1;
	threads.x = 1;
	threads.y = 1;
	blocks.x = 1; 
	blocks.y = 1; 
	blocks.z = 1;

	cudaEvent_t start, stop;
	cudaDeviceReset();
	cudaDeviceSynchronize();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	cudaMalloc((void**)&global_now,sizeof(clock_t));
	cudaMemset((void*)&global_now,0,1);

	if(numiter)
	{
		for(uint i=0;i<numiter;i++)
		{
			msdelay_kernel<<<blocks,threads>>>(1000,global_now);
			cudaDeviceSynchronize();
		}
	}

	if(lasdelay)
	{
		msdelay_kernel<<<blocks,threads>>>(lasdelay,global_now);
		cudaDeviceSynchronize();
	}

	cudaMemcpy((void*)&cpu_now,(const void*)&global_now,sizeof(clock_t),cudaMemcpyDeviceToHost);
	cudaFree(global_now);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds,start,stop);
	return (uint)milliseconds;
}





__global__ void add_and_msdelay_kernel(uint delaymilliseconds,uint *gpubuffer)
{
    size_t row = blockIdx.y*blockDim.y + threadIdx.y;
    size_t col = blockIdx.x*blockDim.x + threadIdx.x;

    if(col>0 || row>0)
        return;

	clock_t start = clock();
	clock_t now;
	
	for (;;) 
	{
	  now = clock();
	  clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
	  if (cycles >= TitanXpCOEF*delaymilliseconds)
	  {
		break;
	  }
	}
	//printf("\n\n\n ============>>>>>gpubuffer = %u,%u\n",gpubuffer[0],gpubuffer[1]);
	gpubuffer[0] += gpubuffer[1];
	gpubuffer[1] = (uint)now;
}


uint bfa_kernel_add_and_msdelay(uint sumval,uint inval,uint delaymilliseconds)
{
	uint numiter = delaymilliseconds/1000;
	uint lasdelay = delaymilliseconds%1000;

	uint *gpubuffer = NULL;
	uint cpubuffer[2] = {sumval,inval};
	float milliseconds;
	dim3 threads, blocks; 
	threads.z = 1;
	threads.x = 1;
	threads.y = 1;
	blocks.x = 1; 
	blocks.y = 1; 
	blocks.z = 1;

	cudaEvent_t start, stop;
	cudaDeviceReset();
	cudaDeviceSynchronize();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	cudaMalloc((void**)&gpubuffer,2*sizeof(uint));
	//cudaMemset((void*)&gpubuffer,0,2);
	//cpubuffer[0] = 5;cpubuffer[1]=6;
	//printf("===|===|===>>> %u,%u\n",cpubuffer[0],cpubuffer[1]);
	cudaMemcpy((void*)gpubuffer,(const void*)cpubuffer,2*sizeof(uint),cudaMemcpyHostToDevice);
	//cudaDeviceSynchronize();
	//cpubuffer[0] = 7;cpubuffer[1]=8;
	//cudaMemcpy((void*)cpubuffer,(const void*)gpubuffer,2*sizeof(uint),cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();
	//printf("===X===X===>>> %u,%u\n",cpubuffer[0],cpubuffer[1]);
	if(numiter)
	{
		for(uint i=0;i<numiter;i++)
		{
			add_and_msdelay_kernel<<<blocks,threads>>>(1000,gpubuffer);
			cudaDeviceSynchronize();
		}
	}

	if(lasdelay)
	{
		add_and_msdelay_kernel<<<blocks,threads>>>(lasdelay,gpubuffer);
		cudaDeviceSynchronize();
	}

	cudaMemcpy((void*)cpubuffer,(const void*)gpubuffer,2*sizeof(clock_t),cudaMemcpyDeviceToHost);
	cudaFree(gpubuffer);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds,start,stop);

	if(cpubuffer[0] > sumval)
		return (cpubuffer[0]);
	else
		return sumval;
}
