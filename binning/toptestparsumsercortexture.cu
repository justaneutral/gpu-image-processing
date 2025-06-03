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

void deviceinfo(pimholder pref)
{
    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev;
	dev = findCudaDevice(1,NULL);
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
	printf("GPU Device supports SM %d.%d compute capability\n\n", deviceProp.major, deviceProp.minor);
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







void imcreate(CImg<unsigned char>*pim,int w,int h)
{
	pim->assign(w,h).rand(0,255).blur(0.001);
	//cimg_forXY((*pim),x,y)
	//{
	//	(*pim)(x,y) = 255*8.0*(double)(w-x)*(double)x*(double)(w-y)*(double)y/(double)w/(double)h/(double)w/(double)h;
	//}
	//pim->blur(0.01);
}

void imclone(CImg<unsigned char>*pim,CImg<unsigned char>*psrc, int x, int y, int w,int h)
{
	pim->assign(psrc->get_crop(x,y,x+w-1,y+h-1));
}

//texture<unsigned char, 2> texImageTT;
//texture<unsigned char, 2> texTemplateTT;

__device__ __host__ inline float CorrelationValue( float SumI, float SumISq, float SumIT, float SumT, float cPixels, float fDenomExp )
{
    float Numerator = cPixels*SumIT - SumI*SumT;
    float Denominator = rsqrtf( (cPixels*SumISq - SumI*SumI)*fDenomExp );
    return Numerator * Denominator;
}

__global__ void corr_kernel(float *pCorr,size_t CorrPitch,float cPixels,size_t wTemplate,size_t hTemplate,int w,int h,int step,unsigned char *dref,unsigned char *dsrc)
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
            Ic = dsrc[col+x + (w+wTemplate)*(row+y)];
            Tc = dref[x + wTemplate*y];
			I = ((float)Ic)/8;
			T = ((float)Tc)/8;
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

void imcorrelateCPU(CImg<float> *pim,CImg<unsigned char> *psrc,CImg<unsigned char> *pref)
{
	const int Wr = pref->width(), Hr = pref->height(),W = psrc->width()-Wr, H = psrc->height()-Hr;
	int x,y,w,h;
	long cPixels = Wr*Hr;
	unsigned char i,t;
	double vi,vt,si,st,si2,st2,sit;
	
	//printf("Correlation function:");
	pim->assign(W,H);
	for(h=0;h<H;h++)//loop over vertical offset
	{
		printf("\n");
		for(w=0;w<W;w++)//loop over horizontal offset
		{
			si=0;st=0;si2=0;st2=0;sit=0;
			for(y=0;y<Hr;y++)//loop over vertical location
			{
				for(x=0;x<Wr;x++)//loop over horizontal location
				{
					i = psrc->atXY(x+w,y+h);
					t = pref->atXY(x,y);
					vi = (double)i;
					vt = (double)t;
					si+=vi;
					st+=vt;
					si2+=vi*vi;
					st2+=vt*vt;
					sit+=vi*vt;
				}
			}
			float fDenom = float((double)cPixels*st2-(float)st*st);
			float Numerator = cPixels*sit-si*st;
			float Denominator = rsqrt((cPixels*si2-si*si)*fDenom);
			float corrval = Numerator*Denominator;
			//printf(" %+1.4lf",corrval);
			//unsigned char corrvalu = corrval>0.0?(unsigned char)(32*255*(corval)):0;
			(*pim)(w,h) = corrval;
		}
	}
	//printf("\n");
}




int main(int argc, char **argv)
{
	int retcode = -7;
	float milliseconds = 0;

	int xmax = NumCorrsW, ymax = NumCorrsH;
	int wds = 3000-NumCorrsW, hts = 3000-NumCorrsH;

	bool binning = false;

	int ws=wds+xmax,hs=hts+ymax,wr=ws-xmax,hr=hs-ymax,x=(ws-wr)/2,y=(hs-hr)/2;

	int maxThreads = 512;  // number of threads per block
    int maxBlocks = 1024;
    int numBlocks = 0;
    int numThreads = 0;
	float *d_odata = NULL;	

    unsigned int laststep=0, retCnt = 0;

	CImgDisplay , binnedsrcdisp;
	CImg<unsigned char> src,ref;
	imcreate(&src,ws,hs);
	imclone(&ref,&src,x,y,wr,hr);
	CImg<float> correlation(src.width()-ref.width(),src.height()-ref.height());
	
	CImg<unsigned char> binnedsrcimg(src.width(),src.height());

	float T[MAXSTEPVAL];
	float TT[MAXSTEPVAL];
	
	float I=0, II=0, IT=0;

	double cpixels = (double)ref.width()*ref.height();

	float fDenom=0, Numerator=0, Denominator=0, corrval=0, corrval_gpu[NumCorrs];


	//test in CPU
	//imcorrelateCPU(&correlation,&src,&ref);
	//correlationdisp.display(correlation).set_title("correlation (CPU)");

	cudaError_t status;

	cuda(DeviceReset());
	//cuda(SetDeviceFlags( cudaDeviceMapHost ) );
	//cuda(DeviceSetCacheConfig( cudaFuncCachePreferShared ) );
	cudaEvent_t start, stop; cuda(EventCreate(&start)); cuda(EventCreate(&stop)); cuda(DeviceSynchronize());

	//creare correlation function array in GPU
	size_t correlationpitch,srclen=src.width()*src.height()*sizeof(unsigned char),reflen=ref.width()*ref.height()*sizeof(unsigned char); 
	float *dcorrelation = NULL;
	unsigned char *dsrc = NULL,*dref = NULL;
	cuda(MallocPitch((void**)&dcorrelation,&correlationpitch,correlation.width()*sizeof(float),correlation.height()));
	////////cuda(Malloc((void**)&dsrc,srclen));
	////////cuda(Malloc((void**)&dref,reflen));
	////////cuda(Memcpy(dref,&(ref.at(0)),reflen,cudaMemcpyHostToDevice));

	//plase textures in GPU memory
	initTexture(&(ref.at(0)),ref.width(),ref.height(),&(src.at(0)),src.width(),src.height());



#if 1
	//binning
	#define MAXBINS 33
	uchar* binnedref = NULL, *binnedsrc = NULL;
	cuda(Malloc((void**)&binnedref,sizeof(uchar)*ref.size())); //allocatw for mnimal binning 2x2
	cuda(Malloc((void**)&binnedsrc,sizeof(uchar)*src.size())); //allocatw for mnimal binning 2x2

	uint optntx[MAXBINS], optnty[MAXBINS], optgsx[MAXBINS], optgsy[MAXBINS];
	float mintime[MAXBINS];
	for(unsigned int binsize=2;binsize<MAXBINS;binsize++)
	{
		mintime[binsize] = 1000.0;
		uint width = src.width()/binsize;
		uint height = src.height()/binsize;
		for(uint ntx = 1;ntx<=32;ntx++)
			for(uint nty=1;nty<=32;nty++)
			{
				uint gsx = (width+ntx-1)/ntx;
				uint gsy = (height+nty-1)/nty;
				dim3 block_size(ntx,nty);
				dim3 grid_size(gsx,gsy);
				//grid_size.x = gsx;
				//grid_size.y = (height + block_size.y - 1)/block_size.y;
				//binning_ref_kernel<<<grid_size,block_size>>>(binnedref,ref.width(),ref.height(), binsize);
				binning_src_kernel<<<grid_size,block_size>>>(binnedsrc,width,height,binsize); //warm
				cuda(EventRecord(start));
				for(uint bcnt = 0;bcnt<10;bcnt++)
					binning_src_kernel<<<grid_size,block_size>>>(binnedsrc,width,height,binsize);
				cuda(EventRecord(stop));
				cuda(EventSynchronize(stop));
				cuda(EventElapsedTime(&milliseconds,start,stop));
				milliseconds/=10;
				printf("binning size = %u, <<<(%u,%u),(%u,%u)>>>, time = %f milliseconds\n", binsize, gsx,gsy,ntx,nty,milliseconds);
				cuda(Memcpy(&(binnedsrcimg.at(0)),binnedsrc,sizeof(uchar)*binnedsrcimg.width()*binnedsrcimg.height(),cudaMemcpyDeviceToHost));
				binnedsrcdisp.display(binnedsrcimg).set_title("binned src. image");
		
				if(milliseconds<mintime[binsize])
				{
					mintime[binsize] = milliseconds;
					optntx[binsize]=ntx;
					optnty[binsize]=nty;
					optgsx[binsize]=gsx;
					optgsy[binsize]=gsy;
				}
		
				/*
				for(int i=0;i<binnedsrcimg.height();i++)
				{
					std::cout<<endl;
					for(int j=0;j<binnedsrcimg.width();j++)
						std::cout << (char)('0'+(binnedsrcimg.atXY(j,i)>>4));
				}
				*/

				/*uint ind = 0;
				printf("biny = %u\n",binnedsrcimg[ind++]);
				printf("binx = %u\n",binnedsrcimg[ind++]);
				printf("boy = %u\n",binnedsrcimg[ind++]);
				printf("box = %u\n",binnedsrcimg[ind++]);
				printf("numsubblocks = %u\n",binnedsrcimg[ind++]);
				printf("binnedwidth = %u\n",binnedsrcimg[ind++]);
				printf("blockIdx.x = %u\n",binnedsrcimg[ind++]);
				printf("blockDim.x = %u\n",binnedsrcimg[ind++]);
				printf("threadIdx.x = %u\n",binnedsrcimg[ind++]);
				printf("blockIdx.y = %u\n",binnedsrcimg[ind++]);
				printf("blockDim.y = %u\n",binnedsrcimg[ind++]);
				printf("threadIdx.y = %u\n",binnedsrcimg[ind++]);
				printf("numsubblocksy = %u\n",binnedsrcimg[ind++]);*/
			}
	}

	FILE *fly = NULL;
	if(NULL!=(fly=fopen("binning_results.txt","w")))
	{
		char ostr[256];
		for(uint i=2;i<MAXBINS;i++)
		{
			sprintf(ostr,"binsize = %u, time = %u us, dims:<<<(%u,%u),(%u,%u)>>>\n",i,optgsx[i],(uint)(1000.0*mintime[i]),optgsy[i],optntx[i],optnty[i]);
			cout << ostr;
			fprintf(fly,ostr);
		}
		fclose(fly);
	}

#endif	//binning
	
	//refine GPU configuration for maximum number of blocks
	getNumBlocksAndThreads(ref.width()*ref.height(), maxBlocks, maxThreads, numBlocks, numThreads);
	//allocate requared scratch memory in GPU for maximum number of blocks
	cuda(Malloc((void**)&d_odata,NumSumsTotal*numBlocks*sizeof(float)));
	float *h_odata=(float*)malloc(NumSumsTotal*numBlocks*sizeof(float));

	
	//obtain coefficients for reference autocorrelation - T,TT
	for(unsigned int step=1;step<=MAXSTEPVAL;step++)
	{
		T[step-1] = 0;
		TT[step-1] = 0;
		cuda(Memset(d_odata,0,NumSumsTotal*numBlocks*sizeof(float)));
		checkCudaErrors(setRetirementCount(retCnt));
		cuda(DeviceSynchronize());
		getLastCudaError("Kernel execution failed");
		reduceSinglePass(-1.0,-1.0,ref.width(),ref.height(),numThreads,numBlocks,dref,dref,d_odata,step,binning,0,0);
		cuda(DeviceSynchronize());
		getLastCudaError("Kernel execution failed");
		cuda(Memcpy(h_odata,d_odata,NumSumsTotal*numBlocks*sizeof(float),cudaMemcpyDeviceToHost));
		T[step-1]=h_odata[0];
		TT[step-1]=h_odata[1];
		printf("step=%u, T=%f, TT=%f\n",step,T[step-1],TT[step-1]);
		cudaDeviceSynchronize();
	}

	//load next source image and see how long it takes to load another source image
	cuda(EventRecord(start));
	//cuda(Memcpy(dsrc,&(src.at(0)),srclen,cudaMemcpyHostToDevice));
	reloadTexture(&(src.at(0)),src.width(),src.height());
	cuda(EventRecord(stop));
	cuda(EventSynchronize(stop));
	cuda(EventElapsedTime(&milliseconds,start,stop));
	printf("image size %u pixels x % u pixels = %u bytes, memory load time = %f milliseconds\n", ref.width(), ref.height(), ref.width()*ref.height()*sizeof(unsigned char), milliseconds);


	unsigned int prevpositionx=MAXSTEPVAL*NumCorrsW, prevpositiony=MAXSTEPVAL*NumCorrsH, postpositionx=0, postpositiony=0;
	float corrvalmax=0;
	cuda(DeviceSynchronize());
	getLastCudaError("Kernel execution failed");
	cuda(EventRecord(start));
	//zooming:
	unsigned int step = MAXSTEPVAL;
	//refine GPU configuration for current step
	getNumBlocksAndThreads((ref.width()/step)*(ref.height()/step), maxBlocks, maxThreads, numBlocks, numThreads);
	for(y=0;y<=correlation.height()-NumCorrsH; y+=NumCorrsH)
	{//3
		for(x=0;x<=correlation.width()-NumCorrsW; x+=NumCorrsW)
		{//4
			//obtain coefficients for normcorrelation - I,II,IT
			cuda(Memset(d_odata,0,NumSumsTotal*numBlocks*sizeof(float)));
			checkCudaErrors(setRetirementCount(retCnt));
			//cuda(DeviceSynchronize());
			//getLastCudaError("Kernel execution failed");
			//cuda(EventRecord(start));
			reduceSinglePass(T[step-1],TT[step-1],ref.width(),ref.height(),numThreads,numBlocks,dsrc,dref,d_odata,step,binning,x,y);
			//cuda(EventRecord(stop));
			cuda(Memcpy(corrval_gpu,d_odata,NumCorrs*sizeof(float),cudaMemcpyDeviceToHost));
			//cuda(EventSynchronize(stop));
			//cuda(EventElapsedTime(&milliseconds,start,stop));
			for(unsigned int cn=0,unsigned int cny=0;cny<NumCorrsH;cny++)
			{//5
				unsigned int curpositiony = y+cny;
				for(unsigned int cnx=0;cnx<NumCorrsW;cnx++,cn++)
				{//6
					unsigned int curpositionx = x+cnx;
					//printf("step=%u, c(%u,%u)=%1.6lf\n",step,curpositiony,curpositionx,corrval_gpu[cn]);
					if(corrvalmax<corrval_gpu[cn])
					{//7
						postpositiony = curpositiony;
						postpositionx = curpositionx;
						corrvalmax = corrval_gpu[cn];
						laststep = step;
						//printf("step=%u, c(%u,%u)=%1.6lf\n",step,postpositiony,postpositionx,corrvalmax);
					}//7
				}//6
			}//5
		}//4
	}//3
	prevpositiony=postpositiony;
	prevpositionx=postpositionx;
	// check if kernel execution generated an error
	//cudaDeviceSynchronize();

	goto Done;


	for(unsigned int step = MAXSTEPVAL; step>0; step>>=1) //step between pixels in correlation
	{//1
		{//2
			//refine GPU configuration for current step
			getNumBlocksAndThreads((ref.width()/step)*(ref.height()/step), maxBlocks, maxThreads, numBlocks, numThreads);
			for(y=prevpositiony-((step*NumCorrsH)/2);y<((step<MAXSTEPVAL)?(((step*NumCorrsH)/2)+prevpositiony):ymax);y+=NumCorrsH)
			{//3
				for(x=prevpositionx-((step*NumCorrsW)/2);x<((step<MAXSTEPVAL)?(((step*NumCorrsW)/2)+prevpositionx):xmax);x+=NumCorrsW)
				{//4
					//obtain coefficients for normcorrelation - I,II,IT
					cuda(Memset(d_odata,0,NumSumsTotal*numBlocks*sizeof(float)));
					checkCudaErrors(setRetirementCount(retCnt));
					//cuda(DeviceSynchronize());
					//getLastCudaError("Kernel execution failed");
					//cuda(EventRecord(start));
					reduceSinglePass(T[step-1],TT[step-1],ref.width(),ref.height(),numThreads,numBlocks,dsrc,dref,d_odata,step,binning,x,y);
					//cuda(EventRecord(stop));
					cuda(Memcpy(corrval_gpu,d_odata,NumCorrs*sizeof(float),cudaMemcpyDeviceToHost));
					//cuda(EventSynchronize(stop));
					//cuda(EventElapsedTime(&milliseconds,start,stop));
					for(unsigned int cn=0, unsigned int cny=0;cny<NumCorrsH;cny++)
					{//5
						unsigned int curpositiony = y+cny;
						for(unsigned int cnx=0;cnx<NumCorrsW;cnx++,cn++)
						{//6
							unsigned int curpositionx = x+cnx;
							//printf("step=%u, c(%u,%u)=%1.6lf\n",step,curpositiony,curpositionx,corrval_gpu[cn]);
							if(corrvalmax<corrval_gpu[cn])
							{//7
								if(postpositiony == curpositiony && postpositionx == curpositionx)
									goto Done;
								postpositiony = curpositiony;
								postpositionx = curpositionx;
								corrvalmax = corrval_gpu[cn];
								laststep = step;
								//printf("%f milliseconds, step=%u, c(%u,%u)=%1.6lf\n",milliseconds,step,postpositiony,postpositionx,corrvalmax);
							}//7
						}//6
					}//5
				}//4
			}//3
			prevpositiony=postpositiony;
			prevpositionx=postpositionx;
			// check if kernel execution generated an error
			//cudaDeviceSynchronize();
		}//2
	}//1
Done:
	cuda(EventRecord(stop));
	cuda(EventSynchronize(stop));
	cuda(EventElapsedTime(&milliseconds,start,stop));
	printf("%f milliseconds, step=%u, c(%u,%u)=%1.6lf\n",milliseconds,laststep,postpositiony,postpositionx,corrvalmax);
	std::cout<<'>';
	std::cin.get();
	retcode = 0;
Error:
	freeTexture();
	free(h_odata);
	cudaFree(d_odata);
	d_odata=NULL;
    cudaFree(dsrc);
	dsrc = NULL;
    cudaFree(dref);
	dref = NULL;
	cudaFree(dcorrelation);
	dcorrelation = NULL;
#if 0 //binning
	cudaFree(binnedref);
	cudaFree(binnedsrc);
#endif
	cudaEventDestroy(stop);
	cudaEventDestroy(start);
	return retcode;
}
